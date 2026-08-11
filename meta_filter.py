"""
meta_filter.py
==============

Week 2 - Member 4: Metadata Filtering Engine

Responsibilities:
    1. Build validated ChromaDB-compatible metadata filters.
    2. Support project metadata:
           - chunk_type: parent | child
           - source_type: deep_crawl
    3. Evaluate filters against stored chunk metadata.
    4. Inspect parent/child chunk metadata.
    5. Generate a JSON audit report.

Does NOT:
    - generate embeddings
    - create ChromaDB
    - perform vector similarity search

The actual similarity search is handled by retriever.py.
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
    """
    Valid chunk types in the project.
    """

    PARENT = "parent"
    CHILD = "child"


class SourceType(str, Enum):
    """
    Valid source types currently defined by the project.
    """

    DEEP_CRAWL = "deep_crawl"


# ============================================================
# ERROR TYPE
# ============================================================

class MetadataFilterError(ValueError):
    """
    Raised when an invalid metadata filter is requested.
    """

    pass


# ============================================================
# VALID METADATA FIELDS
# ============================================================

_FIELD_ENUMS = {
    "chunk_type": ChunkType,
    "source_type": SourceType,
}


# ============================================================
# VALUE NORMALIZATION
# ============================================================

def _normalize(
    field: str,
    value: str,
) -> str:
    """
    Normalize and validate a metadata value.

    Example:

        " Parent " -> "parent"
        "DEEP_CRAWL" -> "deep_crawl"
    """

    if field not in _FIELD_ENUMS:

        raise MetadataFilterError(
            f"Unsupported metadata field: '{field}'."
        )

    if not isinstance(value, str):

        raise MetadataFilterError(
            f"{field} must be a string."
        )

    value = value.strip().lower()

    allowed_values = {
        member.value
        for member in _FIELD_ENUMS[field]
    }

    if value not in allowed_values:

        raise MetadataFilterError(
            f"Invalid {field}: '{value}'. "
            f"Allowed values: {sorted(allowed_values)}"
        )

    return value


# ============================================================
# METADATA FILTER BUILDER
# ============================================================

class MetadataFilter:
    """
    Build validated ChromaDB-compatible metadata filters.

    Examples:

        MetadataFilter.build(
            chunk_type="parent"
        )

        MetadataFilter.build(
            source_type="deep_crawl"
        )

        MetadataFilter.build(
            chunk_type="parent",
            source_type="deep_crawl"
        )
    """

    @staticmethod
    def build(
        *,
        chunk_type: str | None = None,
        source_type: str | None = None,
        source_type_in: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build a metadata filter.

        Returns:

            {}

        or:

            {
                "chunk_type": "parent"
            }

        or:

            {
                "$and": [
                    {"chunk_type": "parent"},
                    {"source_type": "deep_crawl"}
                ]
            }
        """

        conditions: list[dict[str, Any]] = []

        # ----------------------------------------------------
        # Chunk type
        # ----------------------------------------------------

        if chunk_type is not None:

            conditions.append(
                {
                    "chunk_type": _normalize(
                        "chunk_type",
                        chunk_type,
                    )
                }
            )

        # ----------------------------------------------------
        # Single source type
        # ----------------------------------------------------

        if source_type is not None:

            conditions.append(
                {
                    "source_type": _normalize(
                        "source_type",
                        source_type,
                    )
                }
            )

        # ----------------------------------------------------
        # Multiple source types
        # ----------------------------------------------------

        if source_type_in is not None:

            values = [
                _normalize(
                    "source_type",
                    value,
                )
                for value in source_type_in
            ]

            if not values:

                raise MetadataFilterError(
                    "source_type_in requires at least "
                    "one value."
                )

            # Remove duplicates while preserving order
            values = list(
                dict.fromkeys(values)
            )

            conditions.append(
                {
                    "source_type": {
                        "$in": values
                    }
                }
            )

        # ----------------------------------------------------
        # No filter
        # ----------------------------------------------------

        if not conditions:

            return {}

        # ----------------------------------------------------
        # One condition
        # ----------------------------------------------------

        if len(conditions) == 1:

            return conditions[0]

        # ----------------------------------------------------
        # Multiple conditions
        # ----------------------------------------------------

        return {
            "$and": conditions
        }

    # ========================================================
    # PROJECT-SPECIFIC FILTERS
    # ========================================================

    @classmethod
    def parent_deep_crawl(
        cls,
    ) -> dict[str, Any]:
        """
        Main project filter:

            chunk_type = parent
            AND
            source_type = deep_crawl
        """

        return cls.build(
            chunk_type="parent",
            source_type="deep_crawl",
        )

    @classmethod
    def parent_chunks(
        cls,
    ) -> dict[str, Any]:
        """
        Filter only parent chunks.
        """

        return cls.build(
            chunk_type="parent"
        )

    @classmethod
    def child_chunks(
        cls,
    ) -> dict[str, Any]:
        """
        Filter only child chunks.
        """

        return cls.build(
            chunk_type="child"
        )

    @classmethod
    def deep_crawl(
        cls,
    ) -> dict[str, Any]:
        """
        Filter only deep-crawled sources.
        """

        return cls.build(
            source_type="deep_crawl"
        )

    @staticmethod
    def none() -> dict[str, Any]:
        """
        Return an empty filter.

        An empty filter means no metadata restriction.
        """

        return {}


# ============================================================
# HUMAN-READABLE FILTER DESCRIPTION
# ============================================================

def describe_filter(
    metadata_filter: dict[str, Any],
) -> str:
    """
    Convert a metadata filter into readable text.

    Example:

        {
            "$and": [
                {"chunk_type": "parent"},
                {"source_type": "deep_crawl"}
            ]
        }

    becomes:

        chunk_type='parent' AND source_type='deep_crawl'
    """

    if not metadata_filter:

        return "No metadata filter applied."

    conditions = metadata_filter.get(
        "$and",
        [metadata_filter],
    )

    parts: list[str] = []

    for condition in conditions:

        if not isinstance(condition, dict):
            continue

        for field, value in condition.items():

            if isinstance(value, dict):

                parts.append(
                    f"{field} {value}"
                )

            else:

                parts.append(
                    f"{field}='{value}'"
                )

    if not parts:

        return "No metadata filter applied."

    return " AND ".join(parts)


# ============================================================
# CHUNK METADATA EXTRACTION
# ============================================================

def _get_chunk_metadata(
    chunk: Any,
) -> dict[str, Any]:
    """
    Extract metadata regardless of chunk format.

    Supports:

        1. LangChain Document
        2. Dictionary containing:
               {"metadata": {...}}
        3. Objects with a metadata attribute
    """

    # --------------------------------------------------------
    # Dictionary-based chunk
    # --------------------------------------------------------

    if isinstance(chunk, dict):

        metadata = chunk.get(
            "metadata",
            {},
        )

    # --------------------------------------------------------
    # LangChain Document / object
    # --------------------------------------------------------

    else:

        metadata = getattr(
            chunk,
            "metadata",
            {},
        )

    # --------------------------------------------------------
    # Guarantee dictionary
    # --------------------------------------------------------

    if not isinstance(
        metadata,
        dict,
    ):

        return {}

    return metadata


# ============================================================
# FILTER CONDITION MATCHING
# ============================================================

def _matches_condition(
    condition: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    """
    Check whether one metadata condition matches.

    Supports:

        {"chunk_type": "parent"}

    and:

        {
            "source_type": {
                "$in": ["deep_crawl"]
            }
        }

    Matching is case-insensitive and ignores
    leading/trailing whitespace.
    """

    if not condition:

        return True

    field, expected = next(
        iter(condition.items())
    )

    actual = metadata.get(
        field
    )

    # --------------------------------------------------------
    # Metadata field doesn't exist
    # --------------------------------------------------------

    if actual is None:

        return False

    actual_normalized = (
        str(actual)
        .strip()
        .lower()
    )

    # --------------------------------------------------------
    # $in operator
    # --------------------------------------------------------

    if isinstance(
        expected,
        dict,
    ) and "$in" in expected:

        allowed = {
            str(value)
            .strip()
            .lower()
            for value in expected["$in"]
        }

        return (
            actual_normalized
            in allowed
        )

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    expected_normalized = (
        str(expected)
        .strip()
        .lower()
    )

    return (
        actual_normalized
        == expected_normalized
    )


# ============================================================
# COMPLETE FILTER MATCH
# ============================================================

def filter_matches(
    metadata_filter: dict[str, Any],
    metadata: dict[str, Any],
) -> bool:
    """
    Evaluate whether metadata satisfies a filter.

    Supports the exact filter structures generated by
    MetadataFilter.build():

        {}
        {"chunk_type": "parent"}

        {
            "$and": [
                {"chunk_type": "parent"},
                {"source_type": "deep_crawl"}
            ]
        }

    """

    # --------------------------------------------------------
    # Empty filter
    # --------------------------------------------------------

    if not metadata_filter:

        return True

    # --------------------------------------------------------
    # Extract conditions
    # --------------------------------------------------------

    conditions = metadata_filter.get(
        "$and",
        [metadata_filter],
    )

    # --------------------------------------------------------
    # All conditions must match
    # --------------------------------------------------------

    return all(
        _matches_condition(
            condition,
            metadata,
        )
        for condition in conditions
    )


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[Any]:
    """
    Load parent/child chunks from the previous pipeline stage.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run advanced_chunker.py first."
        )

    try:

        with open(
            INPUT_FILE,
            "rb",
        ) as file:

            chunks = pickle.load(
                file
            )

    except Exception as error:

        raise RuntimeError(
            f"Failed to load chunks from "
            f"'{INPUT_FILE}': {error}"
        ) from error

    if not isinstance(
        chunks,
        list,
    ):

        raise ValueError(
            "Input file does not contain "
            "a list of chunks."
        )

    return chunks


# ============================================================
# METADATA AUDIT
# ============================================================

def audit_metadata(
    chunks: list[Any],
    target_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Inspect metadata stored inside every chunk.

    Returns statistics including:

        - total chunks
        - chunk type distribution
        - source type distribution
        - missing metadata
        - chunks matching the requested filter
    """

    # --------------------------------------------------------
    # Default filter
    # --------------------------------------------------------

    if target_filter is None:

        target_filter = (
            MetadataFilter.parent_deep_crawl()
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    chunk_types = Counter()

    source_types = Counter()

    missing_chunk_type = 0

    missing_source_type = 0

    matching_chunks = 0

    # --------------------------------------------------------
    # Inspect every chunk
    # --------------------------------------------------------

    for chunk in chunks:

        metadata = _get_chunk_metadata(
            chunk
        )

        # ----------------------------------------------------
        # Chunk type
        # ----------------------------------------------------

        chunk_type = metadata.get(
            "chunk_type"
        )

        if chunk_type:

            chunk_types[
                str(chunk_type)
            ] += 1

        else:

            missing_chunk_type += 1

        # ----------------------------------------------------
        # Source type
        # ----------------------------------------------------

        source_type = metadata.get(
            "source_type"
        )

        if source_type:

            source_types[
                str(source_type)
            ] += 1

        else:

            missing_source_type += 1

        # ----------------------------------------------------
        # Filter matching
        # ----------------------------------------------------

        if filter_matches(
            target_filter,
            metadata,
        ):

            matching_chunks += 1

    # --------------------------------------------------------
    # Return audit report
    # --------------------------------------------------------

    return {
        "total_chunks": len(chunks),

        "chunk_type_distribution": dict(
            chunk_types
        ),

        "source_type_distribution": dict(
            source_types
        ),

        "missing_chunk_type": (
            missing_chunk_type
        ),

        "missing_source_type": (
            missing_source_type
        ),

        "matching_chunks": (
            matching_chunks
        ),

        "filter_description": (
            describe_filter(
                target_filter
            )
        ),
    }


# ============================================================
# PRINT AUDIT
# ============================================================

def print_audit(
    audit: dict[str, Any],
) -> None:
    """
    Print metadata audit information.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "METADATA AUDIT"
    )

    print(
        "=" * 70
    )

    print(
        f"Total chunks          : "
        f"{audit['total_chunks']}"
    )

    print(
        "\nChunk type distribution:"
    )

    for key, value in (
        audit[
            "chunk_type_distribution"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        "\nSource type distribution:"
    )

    for key, value in (
        audit[
            "source_type_distribution"
        ].items()
    ):

        print(
            f"  {key}: {value}"
        )

    print(
        f"\nMissing chunk_type     : "
        f"{audit['missing_chunk_type']}"
    )

    print(
        f"Missing source_type    : "
        f"{audit['missing_source_type']}"
    )

    print(
        f"Matching chunks        : "
        f"{audit['matching_chunks']}"
    )

    print(
        f"\nActive filter          : "
        f"{audit['filter_description']}"
    )

    print(
        "=" * 70
    )


# ============================================================
# SAVE JSON REPORT
# ============================================================

def save_report(
    metadata_filter: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """
    Save the metadata filtering audit as JSON.
    """

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "role": "Metadata Filtering",

        "filter": metadata_filter,

        "filter_description": (
            describe_filter(
                metadata_filter
            )
        ),

        "audit": audit,
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )


# ============================================================
# FILTER TESTS
# ============================================================

def run_filter_tests() -> None:
    """
    Run basic tests to verify that the filter builder
    and evaluator work correctly.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "RUNNING METADATA FILTER TESTS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Test 1: Parent
    # --------------------------------------------------------

    parent_filter = (
        MetadataFilter.parent_chunks()
    )

    parent_metadata = {
        "chunk_type": "parent"
    }

    assert filter_matches(
        parent_filter,
        parent_metadata,
    )

    # --------------------------------------------------------
    # Test 2: Child should not match parent filter
    # --------------------------------------------------------

    child_metadata = {
        "chunk_type": "child"
    }

    assert not filter_matches(
        parent_filter,
        child_metadata,
    )

    # --------------------------------------------------------
    # Test 3: Deep crawl
    # --------------------------------------------------------

    crawl_filter = (
        MetadataFilter.deep_crawl()
    )

    crawl_metadata = {
        "source_type": "deep_crawl"
    }

    assert filter_matches(
        crawl_filter,
        crawl_metadata,
    )

    # --------------------------------------------------------
    # Test 4: Combined filter
    # --------------------------------------------------------

    combined_filter = (
        MetadataFilter.parent_deep_crawl()
    )

    combined_metadata = {
        "chunk_type": "parent",
        "source_type": "deep_crawl",
    }

    assert filter_matches(
        combined_filter,
        combined_metadata,
    )

    # --------------------------------------------------------
    # Test 5: Combined filter should reject child
    # --------------------------------------------------------

    combined_child_metadata = {
        "chunk_type": "child",
        "source_type": "deep_crawl",
    }

    assert not filter_matches(
        combined_filter,
        combined_child_metadata,
    )

    # --------------------------------------------------------
    # Test 6: Case-insensitive matching
    # --------------------------------------------------------

    case_metadata = {
        "chunk_type": " PARENT ",
        "source_type": " DEEP_CRAWL ",
    }

    assert filter_matches(
        combined_filter,
        case_metadata,
    )

    # --------------------------------------------------------
    # Test 7: Empty filter
    # --------------------------------------------------------

    assert filter_matches(
        MetadataFilter.none(),
        {},
    )

    print(
        "\nAll metadata filter tests passed."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Main metadata-filtering audit.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "METADATA FILTERING ENGINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # 1. Run internal tests
    # --------------------------------------------------------

    print(
        "\n[1] RUNNING FILTER TESTS"
    )

    print(
        "-" * 70
    )

    try:

        run_filter_tests()

    except AssertionError as error:

        print(
            "\n[ERROR] Metadata filter tests failed."
        )

        raise SystemExit(1) from error

    # --------------------------------------------------------
    # 2. Build project filter
    # --------------------------------------------------------

    metadata_filter = (
        MetadataFilter.parent_deep_crawl()
    )

    print(
        "\n[2] ACTIVE FILTER"
    )

    print(
        "-" * 70
    )

    print(
        metadata_filter
    )

    print(
        "\nDescription:"
    )

    print(
        describe_filter(
            metadata_filter
        )
    )

    # --------------------------------------------------------
    # 3. Load chunks
    # --------------------------------------------------------

    print(
        "\n[3] LOADING CHUNKS"
    )

    print(
        "-" * 70
    )

    print(
        f"Input: {INPUT_FILE}"
    )

    try:

        chunks = load_chunks()

    except Exception as error:

        print(
            f"\n[ERROR] {error}"
        )

        raise SystemExit(1) from error

    print(
        f"[OK] Loaded {len(chunks)} chunks."
    )

    # --------------------------------------------------------
    # 4. Audit metadata
    # --------------------------------------------------------

    print(
        "\n[4] METADATA AUDIT"
    )

    print(
        "-" * 70
    )

    audit = audit_metadata(
        chunks,
        target_filter=metadata_filter,
    )

    print_audit(
        audit
    )

    # --------------------------------------------------------
    # 5. Warn about missing metadata
    # --------------------------------------------------------

    if audit[
        "missing_chunk_type"
    ] > 0:

        print(
            "\n[WARNING] "
            f"{audit['missing_chunk_type']} chunks "
            "are missing 'chunk_type'."
        )

    if audit[
        "missing_source_type"
    ] > 0:

        print(
            "[WARNING] "
            f"{audit['missing_source_type']} chunks "
            "are missing 'source_type'."
        )

    # --------------------------------------------------------
    # 6. Warn about inconsistent casing
    # --------------------------------------------------------

    chunk_distribution = audit[
        "chunk_type_distribution"
    ]

    source_distribution = audit[
        "source_type_distribution"
    ]

    chunk_type_keys_lower = {
        str(key).lower()
        for key in chunk_distribution
    }

    source_type_keys_lower = {
        str(key).lower()
        for key in source_distribution
    }

    if (
        len(chunk_distribution)
        > len(chunk_type_keys_lower)
    ):

        print(
            "\n[WARNING] "
            "chunk_type contains inconsistent casing."
        )

    if (
        len(source_distribution)
        > len(source_type_keys_lower)
    ):

        print(
            "[WARNING] "
            "source_type contains inconsistent casing."
        )

    # --------------------------------------------------------
    # 7. Save report
    # --------------------------------------------------------

    print(
        "\n[5] SAVING REPORT"
    )

    print(
        "-" * 70
    )

    try:

        save_report(
            metadata_filter,
            audit,
        )

    except Exception as error:

        print(
            f"[ERROR] Failed to save report: {error}"
        )

        raise SystemExit(1) from error

    print(
        f"[CREATED] {REPORT_FILE}"
    )

    # --------------------------------------------------------
    # 8. Final status
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "METADATA FILTERING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()