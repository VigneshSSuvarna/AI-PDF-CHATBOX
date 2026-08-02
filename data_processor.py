# Run these commands in terminal before execution :
# python -m pip install PyMuPDF
# pip install pyplumber
# pip install pypdf

import fitz
import pdfplumber
from pypdf import PdfReader
from langchain_core.documents import Document

def extract_rag_text(file_path):
    """
    Extract text using fitz and package it for LangChain.
    """
    documents = []
    
    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():  # Only save pages that actually have text
                # Package it into a LangChain Document
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                )
                documents.append(doc)
                
    return documents

# --- Test it ---
if __name__ == "__main__":
    file_path = "AI CHAT BOT USING RAG.pdf"
    langchain_docs = extract_rag_text(file_path)
    
    print(f"Extracted {len(langchain_docs)} pages formatted for LangChain!")

def extract_text(file_path):
    """Extract all text from the PDF."""
    with fitz.open(file_path) as pdf:
        return "".join(page.get_text() for page in pdf)


def extract_links(file_path):
    """Extract all hyperlinks from the PDF."""
    links = []

    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            page_links = page.get_links()
            if page_links:
                links.append({
                    "page": page_num,
                    "links": page_links
                })

    return links


def extract_tables(file_path):
    """Extract all tables from the PDF."""
    tables = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()
            if page_tables:
                tables.append({
                    "page": page_num,
                    "tables": page_tables
                })

    return tables


def extract_images(file_path):
    """Extract all images from the PDF."""
    reader = PdfReader(file_path)
    saved_files = []

    for page_num, page in enumerate(reader.pages, start=1):
        for img_num, img in enumerate(page.images, start=1):
            filename = f"page{page_num}_img{img_num}.png"

            with open(filename, "wb") as fp:
                fp.write(img.data)

            saved_files.append(filename)

    return saved_files


# ---------------- Main ----------------

file_path = "AI CHAT BOT USING RAG.pdf"

text = extract_text(file_path)
links = extract_links(file_path)
tables = extract_tables(file_path)
images = extract_images(file_path)
document=extract_rag_text(file_path)

print(f"Characters extracted : {len(text)}")
print(f"Pages containing links : {len(links)}")
print(f"Pages containing tables : {len(tables)}")
print(f"Images extracted : {len(images)}")

#print(text)
#print(links)
#print(tables)
#print(images)
print(document)