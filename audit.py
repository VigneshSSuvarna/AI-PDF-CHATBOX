"""
data_auditor.py — Phase 4: Data Quality Audit

Reads chunked documents, drops low-quality/duplicate chunks, and writes
an approved knowledge base plus a JSON audit report.
"""

import argparse
import hashlib
import json
import logging
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------
# CONFIGURATION (defaults — all overridable via CLI flags)
# ---------------------------------------------------
DEFAULT_INPUT_FILE = Path("output") / "chunked_documents.pkl"
DEFAULT_OUTPUT_FILE = Path("output") / "audited_documents.pkl"
DEFAULT_REPORT_FILE = Path("output") / "audit_report.json"

DEFAULT_MIN_WORD_COUNT = 10
DEFAULT_MAX_SPECIAL_CHAR_RATIO = 0.3

SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------
# STATS TRACKER
# ---------------------------------------------------
@dataclass
class AuditStats:
    total: int = 0
    kept: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=lambda: {
        "empty": 0,
        "too_short": 0,
        "high_special_char_ratio": 0,
        "duplicate": 0,
    })

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped_by_reason.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_processed": self.total,
            "total_kept": self.kept,
            "total_dropped": self.total_dropped,
            "dropped_by_reason": self.dropped_by_reason,
        }


# ---------------------------------------------------
# QUALITY CHECKS
# ---------------------------------------------------
def get_quality_issue(
    text: str,
    seen_hashes: set[str],
    min_word_count: int,
    max_special_char_ratio: float,
) -> str | None:
    """Returns a drop reason if the chunk fails a quality check, else None."""
    normalized = WHITESPACE_PATTERN.sub(" ", text).strip()

    if not normalized:
        return "empty"

    if len(normalized.split()) < min_word_count:
        return "too_short"

    special_ratio = len(SPECIAL_CHAR_PATTERN.findall(normalized)) / len(normalized)
    if special_ratio > max_special_char_ratio:
        return "high_special_char_ratio"

    content_hash = hashlib.md5(normalized.lower().encode()).hexdigest()
    if content_hash in seen_hashes:
        return "duplicate"
    seen_hashes.add(content_hash)

    return None


def audit_chunks(
    chunks: list,
    min_word_count: int = DEFAULT_MIN_WORD_COUNT,
    max_special_char_ratio: float = DEFAULT_MAX_SPECIAL_CHAR_RATIO,
) -> tuple[list, AuditStats]:
    """Filters chunks, returning (kept_chunks, stats)."""
    logger.info(
        "🔍 Auditing %d chunks (min words: %d, max special-char ratio: %.2f)...",
        len(chunks), min_word_count, max_special_char_ratio,
    )

    seen_hashes: set[str] = set()
    stats = AuditStats(total=len(chunks))
    kept = []

    for chunk in chunks:
        issue = get_quality_issue(
            chunk.page_content, seen_hashes, min_word_count, max_special_char_ratio
        )
        if issue:
            stats.dropped_by_reason[issue] += 1
        else:
            kept.append(chunk)

    stats.kept = len(kept)
    return kept, stats


def print_summary(stats: AuditStats) -> None:
    logger.info("\n==================================================")
    logger.info("📊 AUDIT SUMMARY")
    logger.info("==================================================")
    logger.info("Processed %d chunks. Dropped %d due to low quality.", stats.total, stats.total_dropped)
    if stats.total_dropped:
        logger.info("  🕳️  Empty/blank:            %d", stats.dropped_by_reason["empty"])
        logger.info("  ✂️  Too short:              %d", stats.dropped_by_reason["too_short"])
        logger.info("  🧩 High special-char ratio: %d", stats.dropped_by_reason["high_special_char_ratio"])
        logger.info("  🪞 Duplicate content:       %d", stats.dropped_by_reason["duplicate"])


# ---------------------------------------------------
# I/O HELPERS
# ---------------------------------------------------
def load_chunks(input_file: Path) -> list:
    if not input_file.exists():
        logger.error("❌ ERROR: Cannot find '%s'. Please run 'chunker.py' first!", input_file)
        sys.exit(1)

    logger.info("📥 Loading Chunked Documents...")
    try:
        with open(input_file, "rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError) as e:
        logger.error("❌ ERROR: '%s' is corrupted or unreadable (%s).", input_file, e)
        sys.exit(1)


def save_outputs(kept_chunks: list, stats: AuditStats, output_file: Path, report_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(kept_chunks, f)

    with open(report_file, "w") as f:
        json.dump(stats.as_dict(), f, indent=4)

    logger.info("\n💾 Approved Knowledge Base stored at: '%s' (%d chunks)", output_file, stats.kept)
    logger.info("📄 Audit report saved at: '%s'", report_file)


# ---------------------------------------------------
# CLI / EXECUTION
# ---------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit chunked documents for quality issues.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORD_COUNT)
    parser.add_argument("--max-special-ratio", type=float, default=DEFAULT_MAX_SPECIAL_CHAR_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("==================================================")
    logger.info("🧪 PHASE 4: DATA QUALITY AUDIT")
    logger.info("==================================================")

    chunks = load_chunks(args.input)
    kept_chunks, stats = audit_chunks(chunks, args.min_words, args.max_special_ratio)
    print_summary(stats)
    save_outputs(kept_chunks, stats, args.output, args.report)


if __name__ == "__main__":
    main()