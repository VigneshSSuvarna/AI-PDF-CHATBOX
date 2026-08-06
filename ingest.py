import os
import fitz
import hashlib
import pickle
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    RecursiveUrlLoader,
    SitemapLoader,
    OnlinePDFLoader
)

# ---------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------
DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"
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
    """Extracts text AND hidden URLs from a single PDF."""
    documents = []
    found_links = set() # [NEW: DEEP CRAWLING] Track links found in this specific PDF
    
    try:
        pdf = fitz.open(pdf_path)
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)
            text = page.get_text("text")
            
            # [NEW: DEEP CRAWLING] Grab all clickable links on this page
            for link in page.get_links():
                if "uri" in link and link["uri"].startswith("http"):
                    found_links.add(link["uri"])
            
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
        # [NEW: DEEP CRAWLING] Return both the text documents AND the links
        return documents, list(found_links)
    
    except Exception as e:
        return f"ERROR::{pdf_path}::{e}", []
def extract_pdf(pdf_path):
    """Extracts text AND hidden/raw URLs from a single PDF."""
    documents = []
    found_links = set() 
    
    # [NEW] A RegEx pattern that detects any web address in a block of text
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
                    
            # Method 2: [NEW FALLBACK] Scan the raw text for typed-out URLs
            text_urls = re.findall(URL_PATTERN, text)
            for url in text_urls:
                # Strip off any accidental trailing punctuation (like a period at the end of a sentence)
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

def ingest_local_pdfs():
    print("\n--- [1/3] INGESTING LOCAL PDFs ---")
    if not os.path.exists(DATA_FOLDER):
        print(f"Directory '{DATA_FOLDER}' not found. Skipping local PDFs.")
        return [], []

    pdf_files = collect_pdf_files(DATA_FOLDER)
    print(f"Total Local PDFs Found : {len(pdf_files)}")

    hashes, unique_files = set(), []
    for pdf in pdf_files:
        h = file_hash(pdf)
        if h not in hashes:
            hashes.add(h)
            unique_files.append(pdf)

    print(f"Unique PDFs to process : {len(unique_files)}")
    
    documents = []
    failed = []
    all_deep_links = set() # [NEW: DEEP CRAWLING] Master list of all links found

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_pdf, pdf): pdf for pdf in unique_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
            # [NEW: DEEP CRAWLING] Unpack the two return values
            result_docs, result_links = future.result() 
            
            if isinstance(result_docs, str):
                failed.append(result_docs)
            else:
                documents.extend(result_docs)
                all_deep_links.update(result_links) # Add new links to the master list

    if failed:
        with open(os.path.join(OUTPUT_FOLDER,"failed_local_files.txt"), "w") as f:
            for item in failed:
                f.write(item + "\n")
                
    print(f"Local Extraction Complete! Extracted {len(documents)} pages.")
    print(f"🔗 Deep Crawl: Found {len(all_deep_links)} unique websites hidden inside these PDFs.")
    
    # [NEW: DEEP CRAWLING] Return the links so the scraper can use them!
    return documents, list(all_deep_links)


# ===================================================
#  DEEP CRAWLING FUNCTION (NEW)
# ===================================================

def scrape_extracted_links(url_list):
    """Takes the URLs found in the PDFs and scrapes their website text."""
    if not url_list:
        return []
        
    print("\n--- [2/3] DEEP CRAWLING (Scraping links found in PDFs) ---")
    scraped_documents = []
    
    for url in tqdm(url_list, desc="Scraping Hidden URLs"):
        try:
            loader = WebBaseLoader(url)
            web_docs = loader.load()
            
            # Tag these documents so we know they came from the Deep Crawl
            for doc in web_docs:
                doc.metadata["source_type"] = "deep_crawl"
                
            scraped_documents.extend(web_docs)
        except Exception as e:
            # We silently pass here so one broken website doesn't crash the program
            pass 
            
    print(f"Deep Crawling Complete! Added {len(scraped_documents)} new documents from the web.")
    return scraped_documents


# ===================================================
#  ONLINE DATA INGESTION (Standard)
# ===================================================

def extract_online_pdfs(pdf_url_list):
    all_pdf_documents = []
    for url in pdf_url_list:
        print(f"Fetching online PDF -> {url}")
        try:
            loader = OnlinePDFLoader(url)
            docs = loader.load()
            all_pdf_documents.extend(docs)
        except Exception as e:
            print(f"Failed to load PDF {url}. Error: {e}")
    return all_pdf_documents

def extract_multiple_urls(url_list):
    all_web_documents = []
    for url in url_list:
        print(f"Scraping URL -> {url}")
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_web_documents.extend(docs)
        except Exception as e:
            print(f"Failed to load website {url}. Error: {e}")
    return all_web_documents

def ingest_web_data():
    print("\n--- [3/3] INGESTING STANDARD ONLINE DATA ---")
    web_docs, online_pdf_docs = [], []

    # 1. Standard Websites
    if os.path.exists("data/urls.txt"):
        print("Reading websites from 'data/urls.txt'...")
        with open("data/urls.txt", "r") as f:
            web_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        web_docs = extract_multiple_urls(web_urls)
    
    # 2. Online PDFs
    if os.path.exists("data/online_pdfs.txt"):
        print("Reading online PDFs from 'data/online_pdfs.txt'...")
        with open("data/online_pdfs.txt", "r") as f:
            pdf_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        online_pdf_docs = extract_online_pdfs(pdf_urls)

    total_online = web_docs + online_pdf_docs
    print(f"Online Extraction Complete! Extracted {len(total_online)} pages/documents.")
    return total_online


# ===================================================
#  THE JOINT MERGE: MASTER EXECUTION
# ===================================================

if __name__ == "__main__":
    print("==================================================")
    print(" 🚀 STARTING MASTER DATA INGESTION PIPELINE")
    print("==================================================")
    
    # 1. Collect Local PDFs AND their hidden URLs
    local_data, deep_crawl_urls = ingest_local_pdfs()
    
    # 2. Scrape the URLs found in Step 1 (The Deep Crawl)
    deep_crawl_data = scrape_extracted_links(deep_crawl_urls)
    
    # 3. Collect standard online data (urls.txt, etc.)
    online_data = ingest_web_data()
    
    # 4. Combine EVERYTHING into a single massive knowledge base
    master_document_list = local_data + deep_crawl_data + online_data
    
    # 5. Save the final output for Phase 2 (Data Cleaning / Chunking)
    output_filepath = os.path.join(OUTPUT_FOLDER, "master_documents.pkl")
    with open(output_filepath, "wb") as f:
        pickle.dump(master_document_list, f)

    print("\n==================================================")
    print(f" ✅ SUCCESS: {len(master_document_list)} Total Documents Ingested!")
    print(f" 💾 Master Knowledge Base saved to: '{output_filepath}'")
    print("==================================================")