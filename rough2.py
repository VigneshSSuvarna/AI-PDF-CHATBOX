import os
import fitz
import hashlib
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from tqdm import tqdm


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

DATA_FOLDER = "data"
OUTPUT_FOLDER = "output"

MAX_WORKERS = os.cpu_count()

SUPPORTED_EXTENSIONS = [".pdf"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ---------------------------------------------------
# Duplicate Detection
# ---------------------------------------------------

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


# ---------------------------------------------------
# PDF Extraction
# ---------------------------------------------------

def extract_pdf(pdf_path):

    documents = []

    try:

        pdf = fitz.open(pdf_path)

        for page_num in range(len(pdf)):

            page = pdf.load_page(page_num)

            text = page.get_text("text")

            if not text.strip():
                continue

            metadata = {

                "source": str(pdf_path),

                "filename": Path(pdf_path).name,

                "page": page_num + 1,

                "total_pages": len(pdf),

                "file_size_MB": round(os.path.getsize(pdf_path)/(1024*1024),2),

            }
            print(metadata)

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        pdf.close()

        return documents

    except Exception as e:

        return f"ERROR::{pdf_path}::{e}"


# ---------------------------------------------------
# Collect PDFs
# ---------------------------------------------------

def collect_pdf_files(folder):

    pdf_files = []

    for root, dirs, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(".pdf"):

                pdf_files.append(os.path.join(root,file))

    return pdf_files


# ---------------------------------------------------
# Main Loader
# ---------------------------------------------------

def ingest_documents():

    print("\nSearching for PDFs...\n")

    pdf_files = collect_pdf_files(DATA_FOLDER)

    print(f"Total PDFs Found : {len(pdf_files)}")

    hashes = set()

    unique_files = []

    for pdf in pdf_files:

        h = file_hash(pdf)

        if h not in hashes:

            hashes.add(h)

            unique_files.append(pdf)

    print(f"Unique PDFs : {len(unique_files)}")

    documents = []

    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(extract_pdf,pdf): pdf
            for pdf in unique_files
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading PDFs"):

            result = future.result()

            if isinstance(result,str):

                failed.append(result)

            else:

                documents.extend(result)

    print("\nExtraction Complete")

    print(f"Total Pages : {len(documents)}")

    print(f"Failed PDFs : {len(failed)}")

    with open(os.path.join(OUTPUT_FOLDER,"documents.pkl"),"wb") as f:

        pickle.dump(documents,f)

    with open(os.path.join(OUTPUT_FOLDER,"failed_files.txt"),"w") as f:

        for item in failed:
            f.write(item+"\n")

    print("\nDocuments Saved Successfully")

    return documents


# ---------------------------------------------------
# Run
# ---------------------------------------------------

if __name__ == "__main__":

    docs = ingest_documents()

    if docs:

        print("\nSample Metadata\n")

        print(docs[0].metadata)

        print("\nFirst 300 Characters\n")

        print(docs[0].page_content[:300])
        