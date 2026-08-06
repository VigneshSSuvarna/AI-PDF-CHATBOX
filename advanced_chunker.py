import os
import pickle
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
INPUT_FILE = os.path.join("output", "cleaned_master_documents.pkl")
OUTPUT_FILE = os.path.join("output", "parent_child_chunks.pkl")

# We create TWO splitters now. 
# The Parent is large (context). The Child is small (precision).
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

def create_parent_child_chunks(documents):
    print("🧬 Initializing Parent-Child Chunking Engine...")
    
    all_advanced_chunks = []
    
    # 1. First, split the raw documents into large Parent Chunks
    parent_chunks = parent_splitter.split_documents(documents)
    print(f"📄 Created {len(parent_chunks)} Large Parent Chunks.")
    
    for parent in parent_chunks:
        # 2. Generate a unique ID for this specific Parent
        parent_id = str(uuid.uuid4())
        
        # Add the ID to the parent's metadata
        parent.metadata["doc_id"] = parent_id
        parent.metadata["chunk_type"] = "parent"
        
        # Save the parent to our final list
        all_advanced_chunks.append(parent)
        
        # 3. Now, split this specific Parent into smaller Child Chunks
        child_chunks = child_splitter.split_documents([parent])
        
        for child in child_chunks:
            # 4. Inject the Parent's ID into the Child's metadata so they are forever linked!
            child.metadata["parent_id"] = parent_id
            child.metadata["chunk_type"] = "child"
            
            # Save the child to our final list
            all_advanced_chunks.append(child)
            
    print(f"👶 Generated a total of {len(all_advanced_chunks)} combined Parent & Child chunks.")
    return all_advanced_chunks

if __name__ == "__main__":
    print("==================================================")
    print("🧩 PHASE 3: ADVANCED PARENT-CHILD CHUNKING")
    print("==================================================")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Cannot find '{INPUT_FILE}'.")
        exit()

    with open(INPUT_FILE, "rb") as f:
        cleaned_documents = pickle.load(f)

    # Run the advanced chunking engine
    advanced_chunks = create_parent_child_chunks(cleaned_documents)

    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(advanced_chunks, f)

    print(f"\n💾 Advanced Knowledge Base safely stored at: '{OUTPUT_FILE}'")