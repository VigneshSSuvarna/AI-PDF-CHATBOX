import os
import pickle

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
INPUT_FILE  = os.path.join("output", "ner_enriched_chunks.pkl")
OUTPUT_FILE = os.path.join("output", "embeddings.pkl")

# Free, local model: 384-dimensional vectors, no API key required.
# Swap for "text-embedding-3-small" + OpenAI client if you prefer API-based embeddings.
MODEL_NAME  = "all-MiniLM-L6-v2"

# Number of chunks to embed in a single forward pass.
# Larger batches are faster but use more RAM. 32 is a safe default.
BATCH_SIZE  = 32

# ---------------------------------------------------
# EMBEDDING ENGINE
# ---------------------------------------------------
def generate_embeddings(chunks, model):
    print(f"\n[EMBED] Encoding {len(chunks)} chunks with '{MODEL_NAME}'...")

    texts = [chunk.page_content for chunk in chunks]

    # Batch-encode all texts. show_progress_bar gives a live tqdm bar.
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,    # returns a numpy float32 array
        normalize_embeddings=True # L2-normalize so cosine sim == dot product
    )

    print(f"[SUCCESS] Generated {len(vectors)} embedding vectors.")
    print(f"          Vector dimensions : {vectors.shape[1]}")
    return vectors


def build_embedding_store(chunks, vectors):
    """
    Pairs every vector with its chunk text and metadata into a flat list
    of dicts that is easy to load into any vector database later.
    """
    store = []
    for chunk, vector in zip(chunks, vectors):
        store.append({
            "text"     : chunk.page_content,
            "vector"   : vector,           # numpy array, shape (384,)
            "metadata" : chunk.metadata,   # source, page, NER fields, etc.
        })
    return store


# ---------------------------------------------------
# EXECUTION BLOCK
# ---------------------------------------------------
if __name__ == "__main__":
    print("==================================================")
    print("PHASE 4: EMBEDDING GENERATION")
    print("==================================================")

    # 1. Verify input exists
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Cannot find '{INPUT_FILE}'.")
        print("        Run 'enrich_ner.py' first to generate enriched chunks.")
        raise SystemExit(1)

    # 2. Load enriched chunks
    print(f"Loading enriched chunks from '{INPUT_FILE}'...")
    with open(INPUT_FILE, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    # 3. Load the embedding model (downloads once, then cached locally)
    print(f"\nLoading embedding model '{MODEL_NAME}'...")
    print("(First run will download ~90 MB — subsequent runs use the local cache.)")
    model = SentenceTransformer(MODEL_NAME)

    # 4. Generate embeddings
    vectors = generate_embeddings(chunks, model)

    # 5. Pair vectors with metadata
    embedding_store = build_embedding_store(chunks, vectors)

    # 6. Save output
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(embedding_store, f)

    print(f"\nEmbedding store saved to: '{OUTPUT_FILE}'")

    # 7. Quality Assurance preview
    sample = embedding_store[0]
    print("\n==================================================")
    print("QA PREVIEW: EMBEDDING ENTRY #1")
    print("==================================================")
    print(f"Source   : {sample['metadata'].get('source')}")
    print(f"Page     : {sample['metadata'].get('page')}")
    print(f"Orgs     : {sample['metadata'].get('organizations', [])}")
    print(f"Vector   : [{sample['vector'][0]:.6f}, {sample['vector'][1]:.6f}, "
          f"{sample['vector'][2]:.6f}, ...]  (shape: {sample['vector'].shape})")
    print(f"Text     : {sample['text'][:120]}...")