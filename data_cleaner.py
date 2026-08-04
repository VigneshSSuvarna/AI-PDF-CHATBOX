import os
import re
import unicodedata
import pickle


class DataCleaner:
    """
    A utility class for cleaning and normalizing raw text data.
    """

    DEFAULT_ALLOWED_PUNCTUATION = ".,!?():/'\"&%$-"

    def __init__(self, allowed_punctuation: str = None, lowercase: bool = False):
        self.allowed_punctuation = (
            allowed_punctuation
            if allowed_punctuation is not None
            else self.DEFAULT_ALLOWED_PUNCTUATION
        )

        self.lowercase = lowercase

        no_hyphen = self.allowed_punctuation.replace("-", "")
        escaped = re.escape(no_hyphen)

        self._special_char_pattern = re.compile(rf"[^\w\s{escaped}-]")
        self._whitespace_pattern = re.compile(r"\s+")

        # Fixed: Explicitly initialized so the attribute exists
        self._unicode_replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
        }

    def _normalize_unicode(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)

        for original, replacement in self._unicode_replacements.items():
            text = text.replace(original, replacement)

        return text

    def _remove_control_chars(self, text: str) -> str:
        return "".join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
        )

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)._name_}")

        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)
        text = self._special_char_pattern.sub("", text)
        text = self._whitespace_pattern.sub(" ", text)
        text = text.strip()

        if self.lowercase:
            text = text.lower()

        return text

    def clean_batch(self, texts: list) -> list:
        return [self.clean_text(t) for t in texts]


# ==========================
# Clean master_documents.pkl
# ==========================

if __name__ == "__main__":
    cleaner = DataCleaner()

    input_file = os.path.join("output", "master_documents.pkl")
    output_file = os.path.join("output", "cleaned_master_documents.pkl")

    if not os.path.exists(input_file):
        print(f"❌ Error: Cannot find '{input_file}'. Please run your data ingestion script first!")
        exit()

    print(f"📥 Loading documents from {input_file}...")
    with open(input_file, "rb") as f:
        documents = pickle.load(f)

    print(f"Total Loaded Documents: {len(documents)}")

    cleaned_documents = []
    dropped_count = 0

    for doc in documents:
        if hasattr(doc, "page_content") and isinstance(doc.page_content, str):
            cleaned_content = cleaner.clean_text(doc.page_content)
            
            if len(cleaned_content) > 5:
                doc.page_content = cleaned_content
                cleaned_documents.append(doc)
            else:
                dropped_count += 1
        else:
            dropped_count += 1

    os.makedirs("output", exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(cleaned_documents, f)

    print(f"\n✨ Cleaning Complete!")
    print(f"🗑️ Dropped {dropped_count} empty or invalid pages.")
    print(f"💾 Cleaned documents saved to {output_file} (Retained: {len(cleaned_documents)} pages)")