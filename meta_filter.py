"""
member4_metadata_filter.py
--------------------------

Member 4: Metadata Filtering Engine

Responsibilities:
    1. Build validated metadata filters.
    2. Inspect the project's parent/child chunks.
    3. Show filtering statistics in the terminal.
    4. Save a JSON audit report.

Does NOT:
    - generate embeddings
    - create ChromaDB
    - perform similarity search

Project metadata:
    chunk_type  -> parent | child
    source_type -> deep_crawl
"""

from __future__ import annotations

import json
import pickle
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("output") / "parent_child_chunks.pkl"
REPORT_FILE = Path("output") / "filter_report.json"


# ============================================================
# METADATA TYPES
# ============================================================

class ChunkType(str, Enum):
    PARENT = "parent"
    CHILD = "child"


class SourceType(str, Enum):
    DEEP_CRAWL = "deep_crawl"


# ============================================================
# ERROR
# ============================================================

class MetadataFilterError(ValueError):
    """Raised when an invalid metadata filter is requested."""


_FIELD_ENUMS = {
    "chunk_type": ChunkType,
    "source_type": SourceType,
}


# ============================================================
# VALIDATION
# ============================================================

def _normalize(field: str, value: str) -> str:
    """Normalize and validate a metadata value."""

    if not isinstance(value, str):
        raise MetadataFilterError(f"{field} must be a string.")

    value = value.strip().lower()

    allowed = {member.value for member in _FIELD_ENUMS[field]}

    if value not in allowed:
        raise MetadataFilterError(
            f"Invalid {field}: '{value}'. Allowed: {sorted(allowed)}"
        )

    return value


# ============================================================
# METADATA FILTER ENGINE
# ============================================================

class MetadataFilter:
    """Build validated ChromaDB-compatible metadata filters."""

    @staticmethod
    def build(
        *,
        chunk_type: str | None = None,
        source_type: str | None = None,
        source_type_in: Iterable[str] | None = None,
    ) -> dict[str, Any]:

        conditions = []

        if chunk_type is not None:
            conditions.append({"chunk_type": _normalize("chunk_type", chunk_type)})

        if source_type is not None:
            conditions.append({"source_type": _normalize("source_type", source_type)})

        if source_type_in is not None:
            values = [_normalize("source_type", value) for value in source_type_in]
            if not values:
                raise MetadataFilterError("source_type_in requires at least one value.")
            conditions.append({"source_type": {"$in": values}})

        if not conditions:
            return {}

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    # --------------------------------------------------------
    # Project filters
    # --------------------------------------------------------

    @classmethod
    def parent_deep_crawl(cls) -> dict[str, Any]:
        """Main project filter."""
        return cls.build(chunk_type="parent", source_type="deep_crawl")

    @classmethod
    def parent_chunks(cls) -> dict[str, Any]:
        """Filter only parent chunks."""
        return cls.build(chunk_type="parent")

    @classmethod
    def deep_crawl(cls) -> dict[str, Any]:
        """Filter only deep-crawled sources."""
        return cls.build(source_type="deep_crawl")

    @staticmethod
    def none() -> dict[str, Any]:
        """Return an unfiltered search."""
        return {}


# ============================================================
# HUMAN-READABLE DESCRIPTION
# ============================================================

def describe_filter(metadata_filter: dict[str, Any]) -> str:
    """Convert a filter into readable text."""

    if not metadata_filter:
        return "No metadata filter applied."

    conditions = metadata_filter.get("$and", [metadata_filter])
    parts = []

    for condition in conditions:
        field, value = next(iter(condition.items()))
        if isinstance(value, dict):
            parts.append(f"{field} {value}")
        else:
            parts.append(f"{field}='{value}'")

    return " AND ".join(parts)


# ============================================================
# FILTER EVALUATION  (fixes: case-sensitive matching, duplicated filter logic)
# ============================================================

def _matches_condition(condition: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Check one {field: value} or {field: {"$in": [...]}} condition against
    a chunk's metadata dict. Comparison is case/whitespace-insensitive on
    both sides, since MetadataFilter.build() normalizes filter values but
    real chunk metadata may not be normalized the same way upstream."""

    field, expected = next(iter(condition.items()))
    actual = metadata.get(field)

    if actual is None:
        return False

    actual_normalized = str(actual).strip().lower()

    if isinstance(expected, dict) and "$in" in expected:
        allowed = {str(v).strip().lower() for v in expected["$in"]}
        return actual_normalized in allowed

    return actual_normalized == str(expected).strip().lower()


def filter_matches(metadata_filter: dict[str, Any], metadata: dict[str, Any]) -> bool:
    """Evaluate whether one chunk's metadata satisfies a MetadataFilter dict.

    Supports the exact shapes MetadataFilter.build() produces: a single
    condition, an implicit-AND {"$and": [...]}, and $in value lists.

    audit_metadata() calls this against the SAME filter object the retriever
    will receive, instead of re-implementing the parent/deep_crawl check as
    a separate hardcoded comparison that could silently drift out of sync.
    """

    if not metadata_filter:
        return True  # empty filter matches everything

    conditions = metadata_filter.get("$and", [metadata_filter])
    return all(_matches_condition(condition, metadata) for condition in conditions)


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[Any]:
    """Load chunks generated by the previous pipeline stage."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Run the parent/child chunking stage first."
        )

    with open(INPUT_FILE, "rb") as file:
        chunks = pickle.load(file)

    if not isinstance(chunks, list):
        raise ValueError("Input file does not contain a chunk list.")

    return chunks


def _get_chunk_metadata(chunk: Any) -> dict[str, Any]:
    """Return a chunk's metadata dict regardless of chunk shape.

    Handles both:
      - plain dicts, e.g. {"text": ..., "metadata": {...}}
      - objects with a `.metadata` attribute, e.g. LangChain Document

    getattr(chunk, "metadata", {}) silently returns {} for a dict-shaped
    chunk (dicts don't have attributes), which previously made every
    dict-shaped chunk look like it had no metadata at all. This checks
    the shape explicitly instead of assuming one.
    """

    if isinstance(chunk, dict):
        metadata = chunk.get("metadata", {})
    else:
        metadata = getattr(chunk, "metadata", {})

    return metadata if isinstance(metadata, dict) else {}


# ============================================================
# METADATA AUDIT
# ============================================================

def audit_metadata(
    chunks: list[Any],
    target_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Inspect metadata stored inside every chunk.

    Produces statistics useful for verifying that the metadata filtering
    layer is working correctly. `target_filter` defaults to the project's
    main filter (parent + deep_crawl) but any MetadataFilter.build() output
    can be passed in to audit against a different filter.
    """

    if target_filter is None:
        target_filter = MetadataFilter.parent_deep_crawl()

    chunk_types = Counter()
    source_types = Counter()

    missing_chunk_type = 0
    missing_source_type = 0

    matching_chunks = 0

    for chunk in chunks:
        metadata = _get_chunk_metadata(chunk)

        chunk_type = metadata.get("chunk_type")
        source_type = metadata.get("source_type")

        # Distribution counts use the RAW value (not normalized) on purpose:
        # if upstream data has "Parent" and "parent" mixed, that shows up
        # here as two separate buckets, which is a useful signal that the
        # chunking stage's casing is inconsistent.
        if chunk_type:
            chunk_types[str(chunk_type)] += 1
        else:
            missing_chunk_type += 1

        if source_type:
            source_types[str(source_type)] += 1
        else:
            missing_source_type += 1

        # Matching uses filter_matches(), which normalizes case/whitespace
        # on both sides, so "Parent" and "parent" both count as matches
        # even though the distribution above reports them separately.
        if filter_matches(target_filter, metadata):
            matching_chunks += 1

    return {
        "total_chunks": len(chunks),
        "chunk_type_distribution": dict(chunk_types),
        "source_type_distribution": dict(source_types),
        "missing_chunk_type": missing_chunk_type,
        "missing_source_type": missing_source_type,
        "parent_deep_crawl_matches": matching_chunks,
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(metadata_filter: dict[str, Any], audit: dict[str, Any]) -> None:
    """Save results as JSON."""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "role": "Metadata Filtering",
        "filter": metadata_filter,
        "filter_description": describe_filter(metadata_filter),
        "audit": audit,
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 65)
    print(" METADATA FILTERING ENGINE")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Build main filter
    # --------------------------------------------------------

    metadata_filter = MetadataFilter.parent_deep_crawl()

    print("\n[1] ACTIVE FILTER")
    print("-" * 65)
    print(metadata_filter)

    print("\nDescription:")
    print(describe_filter(metadata_filter))

    # --------------------------------------------------------
    # 2. Load chunks
    # --------------------------------------------------------

    print("\n[2] LOADING CHUNKS")
    print("-" * 65)
    print(f"Input: {INPUT_FILE}")

    try:
        chunks = load_chunks()
    except Exception as error:
        print(f"\n[ERROR] {error}")
        raise SystemExit(1)  # fail loudly: a swallowed error here must not look like success

    print(f"[OK] Loaded {len(chunks)} chunks.")

    # --------------------------------------------------------
    # 3. Audit metadata (reuses the SAME filter built in step 1)
    # --------------------------------------------------------

    print("\n[3] METADATA AUDIT")
    print("-" * 65)

    audit = audit_metadata(chunks, target_filter=metadata_filter)

    print(f"Total chunks          : {audit['total_chunks']}")
    print(f"Parent chunks          : {audit['chunk_type_distribution'].get('parent', 0)}")
    print(f"Child chunks           : {audit['chunk_type_distribution'].get('child', 0)}")
    print(f"Deep-crawl sources     : {audit['source_type_distribution'].get('deep_crawl', 0)}")
    print(f"Missing chunk_type     : {audit['missing_chunk_type']}")
    print(f"Missing source_type    : {audit['missing_source_type']}")

    if len(audit["chunk_type_distribution"]) > len({k.lower() for k in audit["chunk_type_distribution"]}):
        print("[WARN] chunk_type has inconsistent casing across chunks (see distribution above).")
    if len(audit["source_type_distribution"]) > len({k.lower() for k in audit["source_type_distribution"]}):
        print("[WARN] source_type has inconsistent casing across chunks (see distribution above).")

    # --------------------------------------------------------
    # 4. Show actual filter result
    # --------------------------------------------------------

    print("\n[4] FILTER RESULT")
    print("-" * 65)

    matches = audit["parent_deep_crawl_matches"]

    print("Filter: chunk_type='parent' AND source_type='deep_crawl'")
    print(f"Matching chunks: {matches}")

    # --------------------------------------------------------
    # 5. Save report
    # --------------------------------------------------------

    save_report(metadata_filter, audit)

    print("\n[5] OUTPUT")
    print("-" * 65)
    print(f"[CREATED] {REPORT_FILE}")

    print("\n" + "=" * 65)
    print(" COMPLETED SUCCESSFULLY")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()