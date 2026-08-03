from langchain_community.document_loaders import (
    WebBaseLoader,
    RecursiveUrlLoader,
    SitemapLoader
)
import os

def load_from_sitemap(sitemap_url):
    """
    Loads all documents listed in a website's official sitemap XML.
    """
    print(f"Parsing sitemap: {sitemap_url} ...")
    loader = SitemapLoader(web_path=sitemap_url)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from sitemap!")
    return documents

def crawl_entire_documentation(root_url):
    """
    Crawls an entire website/documentation tree recursively 
    for large-scale capstone projects.
    """
    print(f"Starting deep crawl on: {root_url}")
    loader = RecursiveUrlLoader(
        url=root_url, 
        max_depth=3, 
        prevent_outside=True  # Keeps it locked to your target domain
    )
    documents = loader.load()
    print(f"Successfully scraped {len(documents)} pages from the web crawl!")
    return documents

def extract_multiple_urls(url_list):
    """
    Takes a list of URLs, scrapes them using WebBaseLoader,
    and returns a combined list of LangChain Document objects.
    """
    all_web_documents = []
    for url in url_list:
        print(f"Scraping URL -> {url}")
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            all_web_documents.extend(docs)
            print(f"Successfully scraped: {url}")
        except Exception as e:
            print(f"Failed to load {url}. Error: {e}")
            
    return all_web_documents

# --- Member 2 Execution Block (Web Ingestion Only) ---
if __name__ == "__main__":
    print("--- WEB DATA INGESTION PIPELINE ---")
    
    # Ingest Web Data using urls.txt or default list
    if os.path.exists("urls.txt"):
        print("Reading URLs from 'urls.txt'...")
        with open("urls.txt", "r") as f:
            target_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        web_docs = extract_multiple_urls(target_urls)
    else:
        print("'urls.txt' not found. Running default web list.")
        target_urls = [
            "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
        ]
        web_docs = extract_multiple_urls(target_urls)

    # Master document list for web data
    master_document_list = web_docs
    
    print(f"\n==========================================")
    print(f"TOTAL WEB DOCUMENTS INGESTED: {len(master_document_list)}")
    print(f"==========================================")