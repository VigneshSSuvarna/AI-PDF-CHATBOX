import os
import pickle
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
INPUT_FILE = os.path.join("output", "cleaned_master_documents.pkl")
OUTPUT_FILE = os.path.join("output", "chunked_documents.pkl")

# We use 1000 characters per chunk, with a 200-character overlap 
# so that a sentence cut in half is continued in the next chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------
# THE CHUNKING ENGINE
# ---------------------------------------------------
def split_documents(documents):
    print(f"🔪 Initializing Text Splitter (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Splits by paragraph first, then line, then word
    )
    
    # Perform the splitting
    chunked_docs = text_splitter.split_documents(documents)
    
    print(f"✅ Successfully sliced {len(documents)} pages into {len(chunked_docs)} individual chunks.")
    return chunked_docs

# ---------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------
if __name__ == "__main__":
    print("==================================================")
    print("🧩 PHASE 3: TEXT CHUNKING PIPELINE")
    print("==================================================")
    
    # 1. Verify the Cleaned File exists
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Cannot find '{INPUT_FILE}'. Please run 'data_cleaner.py' first!")
        exit()

    # 2. Load the pristine data
    print("📥 Loading Cleaned Knowledge Base...")
    with open(INPUT_FILE, "rb") as f:
        cleaned_documents = pickle.load(f)

    # 3. Slice the data into chunks
    chunked_documents = split_documents(cleaned_documents)

    # 4. Save the chunked data for the Vector Database
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(chunked_documents, f)

    print(f"\n💾 Chunked Knowledge Base safely stored at: '{OUTPUT_FILE}'")
    
    # 5. Proof of Work (Check the first chunk)
    if chunked_documents:
        print("\n==================================================")
        print("🔍 QUALITY ASSURANCE PREVIEW (Chunk #1)")
        print("==================================================")
        print(f"Length: {len(chunked_documents[0].page_content)} characters")
        print(f"Source: {chunked_documents[0].metadata.get('source', 'Unknown')}")
        print("Content:")
        print(chunked_documents[0].page_content + "\n")