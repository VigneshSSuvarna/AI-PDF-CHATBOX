import os
import pickle
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
# We are grabbing the advanced Parent/Child chunks from Week 1!
INPUT_FILE = os.path.join("output", "parent_child_chunks.pkl")
DB_DIR = "chroma_db"

# Member 2's Choice: The absolute best free/fast model right now
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def build_vector_database():
    print("==================================================")
    print(" 🧠 PHASE 4: BUILDING THE LOCAL VECTOR DATABASE")
    print("==================================================")

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find '{INPUT_FILE}'. Please run the chunker first.")
        return

    # 1. Load the Chunks
    print("📂 Loading pre-processed chunks...")
    with open(INPUT_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks ready for embedding.")

    # 2. Member 2: Initialize the Embedding Engine
    # (The first time you run this, it will take ~10 seconds to download the model from HuggingFace)
    print(f"\n⚙️ Initializing Local AI Model: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        # Change 'cpu' to 'cuda' if you have an Nvidia GPU for 10x faster processing
        model_kwargs={'device': 'cpu'}, 
        # Normalizing helps the math work better during similarity searches
        encode_kwargs={'normalize_embeddings': True} 
    )

    # 3. Member 1: Build & Save the Chroma Database
    print("\n💾 Generating vectors and saving to ChromaDB...")
    print("(This might take a minute depending on your computer's speed)")

    # We process in batches of 100 to prevent your computer from crashing
    BATCH_SIZE = 100
    
    # Initialize an empty Chroma database pointing to a local folder
    vector_db = Chroma(
        collection_name="capstone_knowledge_base",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    # Convert text to numbers and push them into the database
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding Batches"):
        batch = chunks[i : i + BATCH_SIZE]
        vector_db.add_documents(documents=batch)

    print("\n==================================================")
    print(f" ✅ SUCCESS: Vector Database created safely!")
    print(f" 📁 All mathematical vectors are permanently saved in the '{DB_DIR}' folder.")
    print("==================================================")

if __name__ == "__main__":
    build_vector_database()