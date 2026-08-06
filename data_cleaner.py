import os
import re
import unicodedata
import pickle


class DataCleaner:
    """
    Advanced Data Cleaner for RAG Pipelines.
    Implements:
    1. Unicode & Control Character Normalization
    2. Web Navigation & Boilerplate Removal
    3. Repeated Header/Footer Detection
    4. Page Number Removal
    5. List Formatting Preservation & Standardisation
    6. Table / Multi-column Alignment Preservation
    7. Mid-sentence Line Merging & Paragraph Preservation
    8. OCR Garbage & Noise Token Removal
    9. Empty Line & Whitespace Cleanup
    10. Duplicate Paragraph Deduplication
    """

    WEB_BOILERPLATE_PATTERNS = [
        r"^jump to (content|navigation|search)$",
        r"^main menu$",
        r"^navigation$",
        r"^search$",
        r"^donate$",
        r"^log in$",
        r"^create account$",
        r"^toggle navigation$",
        r"^privacy policy$",
        r"^terms of (use|service)$",
        r"^cookie (policy|notice|settings)$",
        r"^all rights reserved\.?$",
        r"^copyright \u00a9?.*$",
        r"^related articles$",
        r"^share this:?$",
        r"^follow us:?$",
        r"^from wikipedia, the free encyclopedia$",
        r"^move to sidebar hide$",
        r"^toggle .* subsection$",
        r"^toggle the table of contents$",
        r"^edit links$",
        r"^personal tools$",
        r"^article$",
        r"^talk$",
        r"^read$",
        r"^edit$",
        r"^view history$",
        r"^tools$",
        r"^actions$",
        r"^contribute$",
        r"^\d+\s+languages$",
        r"^tools move to sidebar hide$",
        r"^main pagecontentscurrent events.*$",
        r"^helplearn to editcommunity portal.*$",
        r"^appearance$",
        r"^contents move to sidebar hide$",
        r"^\(?top\)?$",
        r"^general$",
        r"^print/export$",
        r"^in other projects$",
        r"^what links here.*$",
        r"^download as pdf.*$",
        r"^printable version$",
        r"^wikimedia commons.*$",
        r"^wikidata item$",
    ]

    PAGE_NUMBER_PATTERNS = [
        r"^(page\s+)?\d+\s*(of|/)\s*\d+$",
        r"^page\s+\d+$",
        r"^-\s*\d+\s*-$",
        r"^\[\d+\]$",
        r"^\d+$",  # Standalone page numbers on a line by themselves
    ]

    BULLET_PREFIX_PATTERN = re.compile(r"^[\s]*[•▪►\*o–—\-]\s+")
    OCR_GARBAGE_PATTERN = re.compile(r"([a-zA-Z0-9])\1{5,}")  # e.g., lllllllllll, IIIIIIII, 000O00

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

        self._unicode_replacements = {
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2013": "-",
            "\u2014": "-",
            "\u00a0": " ",
            "\u2022": "•",
            "\u25aa": "▪",
            "\u25ba": "►",
        }

        self._compiled_boilerplate = [
            re.compile(p, re.IGNORECASE) for p in self.WEB_BOILERPLATE_PATTERNS
        ]
        self._compiled_page_numbers = [
            re.compile(p, re.IGNORECASE) for p in self.PAGE_NUMBER_PATTERNS
        ]

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

    def _is_web_boilerplate(self, line: str) -> bool:
        line_clean = line.strip()
        if not line_clean:
            return False
        for pat in self._compiled_boilerplate:
            if pat.match(line_clean):
                return True
        return False

    def _is_page_number(self, line: str) -> bool:
        line_clean = line.strip()
        if not line_clean:
            return False
        for pat in self._compiled_page_numbers:
            if pat.match(line_clean):
                return True
        return False

    def _is_ocr_garbage(self, line: str) -> bool:
        # Check repeated character sequences like lllllllllll or IIIIIIII
        if self.OCR_GARBAGE_PATTERN.search(line):
            return True
        # Extremely high ratio of non-printable or symbol noise
        if len(line) > 10:
            alnum_ratio = sum(1 for c in line if c.isalnum() or c.isspace()) / len(line)
            if alnum_ratio < 0.4:
                return True
        return False

    def _format_lists(self, line: str) -> str:
        # Convert bullet points into unified markdown list format: "- Item"
        if self.BULLET_PREFIX_PATTERN.match(line):
            line = self.BULLET_PREFIX_PATTERN.sub("- ", line)
        return line

    def _is_table_row(self, line: str) -> bool:
        # Detect table rows: lines with pipe symbols or multiple multi-space separated columns
        if "|" in line and line.count("|") >= 2:
            return True
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 3 and any(p.replace(".", "").replace(",", "").isdigit() for p in parts):
            return True
        return False

    def _merge_broken_lines(self, lines: list) -> list:
        merged = []
        i = 0
        n = len(lines)

        while i < n:
            current = lines[i]

            # If current line is empty, bullet list, table row, or ends with terminal punctuation
            if (
                not current
                or self.BULLET_PREFIX_PATTERN.match(current)
                or current.startswith("- ")
                or self._is_table_row(current)
                or current[-1] in ".!?:;\"'%"
            ):
                merged.append(current)
                i += 1
                continue

            # Check if current line should merge with the next line
            if i + 1 < n:
                nxt = lines[i + 1]
                # Don't merge if next line is empty, a list item, table row, or boilerplate
                if (
                    not nxt
                    or self.BULLET_PREFIX_PATTERN.match(nxt)
                    or nxt.startswith("- ")
                    or self._is_table_row(nxt)
                ):
                    merged.append(current)
                    i += 1
                    continue

                # Handle hyphenated word at line end (e.g. "se-" + "mantic")
                if current.endswith("-") and len(current) > 1 and current[-2].isalpha():
                    merged_line = current[:-1] + nxt
                    lines[i + 1] = merged_line
                    i += 1
                    continue

                # Normal mid-sentence line break merge
                lines[i + 1] = current + " " + nxt
                i += 1
                continue

            merged.append(current)
            i += 1

        return merged

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")

        text = self._normalize_unicode(text)
        text = self._remove_control_chars(text)

        raw_lines = text.splitlines()
        processed_lines = []

        for line in raw_lines:
            # 6. Remove blank lines containing only spaces
            line_stripped = line.rstrip()

            if not line_stripped:
                processed_lines.append("")
                continue

            # 1 & 10. Web Boilerplate & Navigation Removal
            if self._is_web_boilerplate(line_stripped):
                continue

            # 1. Page Number Removal
            if self._is_page_number(line_stripped):
                continue

            # 7. OCR Garbage Removal
            if self._is_ocr_garbage(line_stripped):
                continue

            # 9. Table spacing vs normal space normalization
            if self._is_table_row(line_stripped):
                # Preserve alignment spacing for tables
                clean_line = line_stripped
            else:
                # Normalize multiple spaces per line
                clean_line = re.sub(r"[ \t]+", " ", line_stripped)

            # 8. List Formatting
            clean_line = self._format_lists(clean_line)

            processed_lines.append(clean_line)

        # 3. Merge broken lines while preserving paragraphs
        merged_lines = self._merge_broken_lines(processed_lines)

        # 6. Empty Line Cleanup (collapse 3+ newlines into max 2: \n\n)
        cleaned_text = "\n".join(merged_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

        if self.lowercase:
            cleaned_text = cleaned_text.lower()

        return cleaned_text

    def clean_documents(self, documents: list) -> list:
        """
        Cleans a list of LangChain Document objects:
        - Detects and removes repeated cross-page headers/footers.
        - De-duplicates identical paragraphs across documents.
        """
        # 2. Repeated Header/Footer Detection across pages
        header_counts = {}
        footer_counts = {}
        total_docs = len(documents)

        for doc in documents:
            if hasattr(doc, "page_content") and isinstance(doc.page_content, str):
                lines = [l.strip() for l in doc.page_content.splitlines() if l.strip()]
                if lines:
                    top_line = lines[0]
                    header_counts[top_line] = header_counts.get(top_line, 0) + 1
                    bottom_line = lines[-1]
                    footer_counts[bottom_line] = footer_counts.get(bottom_line, 0) + 1

        # repeated header/footer threshold: appears in >= 40% of docs (if docs >= 3)
        threshold = max(2, int(total_docs * 0.4))
        repeated_headers = {h for h, c in header_counts.items() if c >= threshold and len(h) > 3}
        repeated_footers = {f for f, c in footer_counts.items() if c >= threshold and len(f) > 3}

        cleaned_docs = []
        seen_paragraphs = set()

        for doc in documents:
            if not hasattr(doc, "page_content") or not isinstance(doc.page_content, str):
                continue

            content = doc.page_content

            # Strip detected repeated header / footer
            lines = content.splitlines()
            if lines and lines[0].strip() in repeated_headers:
                lines = lines[1:]
            if lines and lines[-1].strip() in repeated_footers:
                lines = lines[:-1]
            content = "\n".join(lines)

            # Apply single-text cleaner pipeline
            cleaned_text = self.clean_text(content)

            # 5. Duplicate Paragraph Removal within/across documents
            paragraphs = cleaned_text.split("\n\n")
            unique_paragraphs = []
            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue
                # Skip duplicate long paragraphs (> 30 chars) to avoid repetitive legal/footer blocks
                if len(p_clean) > 30 and p_clean.lower() in seen_paragraphs:
                    continue
                seen_paragraphs.add(p_clean.lower())
                unique_paragraphs.append(p_clean)

            final_content = "\n\n".join(unique_paragraphs)

            if len(final_content) > 5:
                doc.page_content = final_content
                cleaned_docs.append(doc)

        return cleaned_docs


# ==========================
# Clean master_documents.pkl
# ==========================

if __name__ == "__main__":
    cleaner = DataCleaner()

    input_file = os.path.join("output", "master_documents.pkl")
    output_file = os.path.join("output", "cleaned_master_documents.pkl")

    if not os.path.exists(input_file):
        print(f"[ERROR] Cannot find '{input_file}'. Please run your data ingestion script first!")
        exit()

    print(f"Loading documents from {input_file}...")
    with open(input_file, "rb") as f:
        documents = pickle.load(f)

    print(f"Total Loaded Documents: {len(documents)}")

    cleaned_documents = cleaner.clean_documents(documents)

    os.makedirs("output", exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(cleaned_documents, f)

    print(f"\nAdvanced Cleaning Complete!")
    print(f"Cleaned documents saved to {output_file} (Retained: {len(cleaned_documents)} pages)")