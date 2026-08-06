import os
import fitz
import hashlib
import pickle
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    OnlinePDFLoader
)

# ---------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
TRACKING_FILE = "processed_files.json"
MAX_WORKERS = os.cpu_count() or 4

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ===================================================
#  LOCAL PDF INGESTION & LINK EXTRACTION
# ===================================================

def file_hash(filepath):
    """Generate SHA256 hash for duplicate detection."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def extract_pdf(pdf_path):
    """Extracts text AND hidden/raw URLs from a single PDF."""
    documents = []
    found_links = set() 
    
    # A RegEx pattern that detects any web address in a block of text
    URL_PATTERN = r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?"
    
    try:
        pdf = fitz.open(pdf_path)
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            
            # Method 1: Look for interactive, embedded clickable boxes
            for link in page.get_links():
                if "uri" in link and link["uri"].startswith("http"):
                    found_links.add(link["uri"])
                    
            # Method 2: Scan the raw text for typed-out URLs
            text_urls = re.findall(URL_PATTERN, text)
            for url in text_urls:
                clean_url = url.rstrip(').,;\"\'')
                found_links.add(clean_url)
            
            if not text.strip():
                continue
                
            metadata = {
                "source": str(pdf_path),
                "filename": Path(pdf_path).name,
                "page": page_num + 1,
                "total_pages": len(pdf),
                "file_size_MB": round(os.path.getsize(pdf_path)/(1024*1024),2),
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
        pdf.close()
        return documents, list(found_links)
    
    except Exception as e:
        return f"ERROR::{pdf_path}::{e}", []

def collect_pdf_files(folder):
    pdf_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root,file))
    return pdf_files

def ingest_local_pdfs(tracking_data):
    print("\n--- [1/3] INGESTING LOCAL PDFs ---")
    if not os.path.exists(DATA_FOLDER):
        print(f"Directory '{DATA_FOLDER}' not found. Skipping local PDFs.")
        return [], []

    pdf_files = collect_pdf_files(DATA_FOLDER)
    
    unique_files_to_process = []
    current_run_hashes = set()

    for pdf in pdf_files:
        filename = Path(pdf).name
        file_fingerprint = file_hash(pdf)

        # 1. Skip if there is a duplicate file in the folder right now
        if file_fingerprint in current_run_hashes:
            print(f"🗑️ Skipping folder duplicate: {filename}")
            continue
        current_run_hashes.add(file_fingerprint)

        # 2. Skip if we already processed this exact file in a previous run
        if filename in tracking_data and tracking_data[filename] == file_fingerprint:
            print(f"⏭️ SKIPPED (Already in database): {filename}")
            continue

        # 3. Queue up Brand New or Modified files
        print(f"🆕 NEW/MODIFIED FILE DETECTED: {filename}")
        unique_files_to_process.append(pdf)
        tracking_data[filename] = file_fingerprint

    documents = []
    failed = []
    all_deep_links = set() 

    if unique_files_to_process:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_pdf, pdf): pdf for pdf in unique_files_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
                result_docs, result_links = future.result() 
                
                if isinstance(result_docs, str):
                    failed.append(result_docs)
                else:
                    documents.extend(result_docs)
                    all_deep_links.update(result_links)

    if failed:
        with open(os.path.join(OUTPUT_FOLDER,"failed_local_files.txt"), "w") as f:
            for item in failed:
                f.write(item + "\n")
                
    return documents, list(all_deep_links)


# ===================================================
#  DEEP CRAWLING FUNCTION
# ===================================================

def scrape_extracted_links(url_list, tracking_data):
    if not url_list:
        return []
        
    print("\n--- [2/3] DEEP CRAWLING (Scraping links found in PDFs) ---")
    
    # Filter out URLs we have already scraped before
    if "scraped_urls" not in tracking_data:
        tracking_data["scraped_urls"] = []
        
    new_urls = [url for url in url_list if url not in tracking_data["scraped_urls"]]
    
    if not new_urls:
        print("⏭️ SKIPPED: All hidden URLs were already scraped in previous runs.")
        return []

    scraped_documents = []
    for url in tqdm(new_urls, desc="Scraping Hidden URLs"):
        try:
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            
            for doc in web_docs:
                doc.metadata["source_type"] = "deep_crawl"
                
            scraped_documents.extend(web_docs)
            tracking_data["scraped_urls"].append(url) # Save to memory
        except Exception as e:
            pass 
            
    print(f"Deep Crawling Complete! Added {len(scraped_documents)} new documents from the web.")
    return scraped_documents


# ===================================================
#  ONLINE DATA INGESTION (Standard)
# ===================================================

def extract_multiple_urls(url_list):
    all_web_documents = []
    for url in url_list:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_web_documents.extend(docs)
        except Exception:
            pass
    return all_web_documents

def ingest_web_data(tracking_data):
    print("\n--- [3/3] INGESTING STANDARD ONLINE DATA ---")
    web_docs = []

    if "scraped_urls" not in tracking_data:
        tracking_data["scraped_urls"] = []

    if os.path.exists("data/urls.txt"):
        with open("data/urls.txt", "r") as f:
            web_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        # Only scrape URLs that are brand new to the system
        new_urls = [url for url in web_urls if url not in tracking_data["scraped_urls"]]
        
        if new_urls:
            print(f"🌐 Found {len(new_urls)} NEW websites in urls.txt to scrape.")
            web_docs = extract_multiple_urls(new_urls)
            tracking_data["scraped_urls"].extend(new_urls) # Save to memory
        else:
            print("⏭️ SKIPPED: No new websites found in urls.txt")

    return web_docs


# ===================================================
#  THE JOINT MERGE: MASTER EXECUTION
# ===================================================

if __name__ == "__main__":
    print("==================================================")
    print(" 🚀 STARTING INCREMENTAL DATA PIPELINE")
    print("==================================================")
    
    # 1. Load System Memory
    tracking_data = {}
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            tracking_data = json.load(f)

    # 2. Load Existing Database (So we don't overwrite old data!)
    output_filepath = os.path.join(OUTPUT_FOLDER, "master_documents.pkl")
    if os.path.exists(output_filepath):
        with open(output_filepath, "rb") as f:
            master_document_list = pickle.load(f)
        print(f"📦 Loaded existing database containing {len(master_document_list)} pages.")
    else:
        master_document_list = []
        print("📦 No existing database found. Creating a new one...")
    
    # 3. Ingest Data (Only processes BRAND NEW files and links)
    new_local_data, deep_crawl_urls = ingest_local_pdfs(tracking_data)
    new_deep_crawl_data = scrape_extracted_links(deep_crawl_urls, tracking_data)
    new_online_data = ingest_web_data(tracking_data)
    
    # 4. Combine all the newly found data
    new_documents_total = new_local_data + new_deep_crawl_data + new_online_data
    
    # 5. Save everything if we found new data
    print("\n==================================================")
    if len(new_documents_total) > 0:
        master_document_list.extend(new_documents_total)
        
        # Save the updated database
        with open(output_filepath, "wb") as f:
            pickle.dump(master_document_list, f)
            
        # Save the updated memory file
        with open(TRACKING_FILE, "w") as f:
            json.dump(tracking_data, f, indent=4)
            
        print(f" ✅ SUCCESS: Added {len(new_documents_total)} NEW documents to the database!")
        print(f" 💾 Master Knowledge Base saved to: '{output_filepath}'")
    else:
        print(" ✅ SUCCESS: No new files or links to process. System is fully up to date.")
    print("==================================================")