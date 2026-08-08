import os
import hashlib
import json

TRACKING_FILE = "output/processed_files.json"
DATA_FOLDER = "data"

def get_file_hash(filepath):
    """Generates a SHA-256 digital fingerprint of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        # Read the file in chunks so it doesn't crash on huge PDFs
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_tracking_data():
    """Loads the 'notebook' of previously processed files."""
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    return {} # Return an empty dictionary if it's the first time running

def main():
    print("🔄 Starting Incremental Sync...")
    tracking_data = load_tracking_data()
    files_to_process = []
    
    # 1. Look at every PDF in the folder
    for filename in os.listdir(DATA_FOLDER):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(DATA_FOLDER, filename)
        current_hash = get_file_hash(filepath)
        
        # 2. Check if the file is new or has been modified
        if filename not in tracking_data:
            print(f"🆕 NEW FILE DETECTED: {filename}")
            files_to_process.append(filepath)
            tracking_data[filename] = current_hash
            
        elif tracking_data[filename] != current_hash:
            print(f"✏️ MODIFIED FILE DETECTED: {filename}")
            files_to_process.append(filepath)
            tracking_data[filename] = current_hash
            
        else:
            print(f"⏭️ SKIPPED (No changes): {filename}")

    # 3. Actually process the new/changed files (Simulated here)
    if files_to_process:
        print(f"\n Extracting text and chunking {len(files_to_process)} files...")
        # --> MEMBER 1 & 4's CODE GOES HERE <--
        # run_pdf_loader(files_to_process)
        # run_text_chunker(files_to_process)
        
        # 4. Save the updated tracking notebook
        with open(TRACKING_FILE, "w") as f:
            json.dump(tracking_data, f, indent=4)
        print("💾 Tracking file updated.")
    else:
        print("\n✅ System is fully up to date. Nothing to process!")

if __name__ == "__main__":
    main()