import os
import pickle
import sys

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
INPUT_FILE = os.path.join("output", "chunked_documents.pkl")
OUTPUT_FILE = os.path.join("output", "ner_enriched_chunks.pkl")


def load_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        print("[INFO] Model 'en_core_web_sm' not found locally. Downloading model...")
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


# Words that are never part of a real person name.
# If any token in the entity matches one of these, it is discarded.
NON_NAME_WORDS = {
    # Technical / domain words
    "api", "use", "uses", "using", "write", "writes", "build", "builds",
    "load", "loads", "scrape", "scrapes", "implement", "implements",
    "feature", "focus", "technical", "assigned", "goal", "milestone",
    "data", "web", "pdf", "document", "loaders", "scrapers", "cleaner",
    "chunking", "metadata", "pipeline", "framework", "backend", "frontend",
    "server", "database", "model", "module", "script", "function", "class",
    # Common single generic words that get mis-tagged
    "half", "end", "start", "member", "team", "week", "day", "year",
    "best", "practice", "industry", "alternative", "choice", "version",
}


def is_valid_person_name(name: str) -> bool:
    """
    Returns True only if the entity looks like a genuine person name.

    Rules:
    - Must be 1-3 words (a name like 'John', 'Bill Gates', 'Mary Jane Watson')
    - Every word must start with an uppercase letter (Title Case)
    - No word in the name may appear in the NON_NAME_WORDS blacklist
    - No word may be all-uppercase acronym style (e.g. 'API', 'PDF')
    """
    words = name.strip().split()

    # Rule 1: Reasonable word count for a human name
    if not (1 <= len(words) <= 3):
        return False

    for word in words:
        # Rule 2: Every word must start uppercase
        if not word[0].isupper():
            return False

        # Rule 3: Discard if any word matches non-name blacklist
        if word.lower() in NON_NAME_WORDS:
            return False

        # Rule 4: Discard all-caps words (they are usually acronyms, not names)
        if word.isupper() and len(word) > 1:
            return False

    return True


def enrich_chunks_with_ner(chunks, nlp):
    print(f"\n[NER] Extracting Named Entities for {len(chunks)} chunks...")

    enriched_count = 0

    for idx, chunk in enumerate(chunks, 1):
        if not hasattr(chunk, "page_content") or not chunk.page_content:
            continue

        doc = nlp(chunk.page_content)

        orgs = set()
        people = set()
        dates = set()
        locations = set()
        all_entities = set()

        for ent in doc.ents:
            clean_text = ent.text.strip()
            # Filter out single-character noise
            if len(clean_text) < 2:
                continue

            if ent.label_ == "ORG":
                orgs.add(clean_text)
                all_entities.add(clean_text)
            elif ent.label_ == "PERSON":
                # Only add person if it passes the confidence heuristic
                if is_valid_person_name(clean_text):
                    people.add(clean_text)
                    all_entities.add(clean_text)
            elif ent.label_ in ["DATE", "TIME"]:
                dates.add(clean_text)
                all_entities.add(clean_text)
            elif ent.label_ in ["GPE", "LOC"]:
                locations.add(clean_text)
                all_entities.add(clean_text)

        # Inject entity metadata into the document chunk
        chunk.metadata["entities"] = sorted(list(all_entities))
        chunk.metadata["organizations"] = sorted(list(orgs))
        chunk.metadata["people"] = sorted(list(people))
        chunk.metadata["dates"] = sorted(list(dates))
        chunk.metadata["locations"] = sorted(list(locations))
        chunk.metadata["ner_enriched"] = True

        if all_entities:
            enriched_count += 1

    print(f"[SUCCESS] Successfully enriched {enriched_count} / {len(chunks)} chunks with Named Entities!")
    return chunks


if __name__ == "__main__":
    print("==================================================")
    print("PHASE 3.5: NAMED ENTITY RECOGNITION (NER)")
    print("==================================================")

    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Cannot find '{INPUT_FILE}'. Please run 'data_chunker.py' first!")
        sys.exit(1)

    print("Loading Chunked Documents...")
    with open(INPUT_FILE, "rb") as f:
        chunks = pickle.load(f)

    nlp = load_spacy_model()
    enriched_chunks = enrich_chunks_with_ner(chunks, nlp)

    # Save enriched chunks to output/ner_enriched_chunks.pkl
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(enriched_chunks, f)

    print(f"\nEnriched chunks stored at: '{OUTPUT_FILE}'")

    # Preview Chunk #1 Metadata
    if enriched_chunks:
        print("\n==================================================")
        print("QA PREVIEW: ENRICHED CHUNK #1 METADATA")
        print("==================================================")
        sample = enriched_chunks[0]
        print(f"Source       : {sample.metadata.get('source')}")
        print(f"Organizations: {sample.metadata.get('organizations')}")
        print(f"People       : {sample.metadata.get('people')}")
        print(f"Dates        : {sample.metadata.get('dates')}")
        print(f"Locations    : {sample.metadata.get('locations')}")
        print(f"All Entities : {sample.metadata.get('entities')}\n")
