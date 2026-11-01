"""
ConstraintPreserver
Step 32: Verify cleaned_documents.json

This script checks:
1. The cleaned JSON file exists.
2. The JSON can be loaded.
3. The top-level structure is valid.
4. The dataset contains records.
5. Each record is inspectable.
6. Required information is reported without modifying the dataset.

IMPORTANT:
This script does NOT change cleaned_documents.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_documents.json"
)


# ============================================================
# 2. HELPERS
# ============================================================

def print_separator() -> None:
    """Print a visual separator."""
    print("=" * 70)


def describe_value(value, indent: int = 0) -> None:
    """
    Print a compact description of a Python value.
    """

    prefix = " " * indent

    if isinstance(value, dict):
        print(f"{prefix}type: dict")
        print(f"{prefix}keys: {list(value.keys())}")

    elif isinstance(value, list):
        print(f"{prefix}type: list")
        print(f"{prefix}length: {len(value)}")

        if value:
            print(f"{prefix}first item type: {type(value[0]).__name__}")

    else:
        print(f"{prefix}type: {type(value).__name__}")
        print(f"{prefix}value: {repr(value)[:200]}")


# ============================================================
# 3. LOAD JSON
# ============================================================

def load_json() -> object:
    """Load cleaned_documents.json."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nCleaned dataset was not found.\n\n"
            f"Expected location:\n{INPUT_FILE}\n"
        )

    print(f"Input file:")
    print(INPUT_FILE)
    print()

    file_size_mb = INPUT_FILE.stat().st_size / (1024 * 1024)

    print(f"File size: {file_size_mb:.2f} MB")
    print()

    with INPUT_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    return data


# ============================================================
# 4. BASIC VALIDATION
# ============================================================

def validate_top_level(data: object) -> None:
    """Validate and describe the top-level JSON structure."""

    print_separator()
    print("TOP-LEVEL STRUCTURE")
    print_separator()

    describe_value(data)

    print()

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError(
                "The JSON file contains an empty list."
            )

        print(f"Number of records: {len(data):,}")

    elif isinstance(data, dict):
        if not data:
            raise ValueError(
                "The JSON file contains an empty object."
            )

        print(
            "The JSON top level is a dictionary/object."
        )

        print(
            "Top-level keys:"
        )

        for key in data.keys():
            print(f"  - {key}")

    else:
        raise ValueError(
            "Unexpected JSON top-level type. "
            f"Found: {type(data).__name__}"
        )


# ============================================================
# 5. RECORD INSPECTION
# ============================================================

def get_records(data: object) -> list:
    """
    Extract records without assuming a specific schema.
    """

    if isinstance(data, list):
        return data

    if isinstance(data, dict):

        # Common possibilities.
        possible_keys = [
            "documents",
            "data",
            "records",
            "items",
        ]

        for key in possible_keys:
            value = data.get(key)

            if isinstance(value, list):
                return value

        # If no obvious list exists, return the dictionary
        # as one inspectable object.
        return [data]

    return [data]


def inspect_records(records: list) -> None:
    """Inspect the first few records."""

    print_separator()
    print("RECORD INSPECTION")
    print_separator()

    print(f"Records available: {len(records):,}")
    print()

    if not records:
        raise ValueError("No records were found.")

    sample_count = min(3, len(records))

    print(
        f"Showing the first {sample_count} record(s):"
    )
    print()

    for index in range(sample_count):

        record = records[index]

        print(f"--- Record {index + 1} ---")

        describe_value(record)

        if isinstance(record, dict):

            print()
            print("Fields:")

            for key, value in record.items():

                value_type = type(value).__name__

                if isinstance(value, str):
                    preview = value.replace(
                        "\n",
                        " "
                    )[:120]

                    print(
                        f"  {key}: "
                        f"{value_type} "
                        f"({len(value):,} chars) "
                        f"-> {preview!r}"
                    )

                elif isinstance(value, list):

                    print(
                        f"  {key}: "
                        f"{value_type} "
                        f"({len(value):,} items)"
                    )

                elif isinstance(value, dict):

                    print(
                        f"  {key}: "
                        f"{value_type} "
                        f"({len(value):,} keys)"
                    )

                else:

                    print(
                        f"  {key}: "
                        f"{value_type} "
                        f"-> {repr(value)[:100]}"
                    )

        elif isinstance(record, str):

            print(
                f"Text length: {len(record):,} characters"
            )

            print(
                f"Preview: "
                f"{record[:200]!r}"
            )

        print()


# ============================================================
# 6. RECORD TYPE ANALYSIS
# ============================================================

def analyze_record_types(records: list) -> None:
    """Count the types of records."""

    print_separator()
    print("RECORD TYPE ANALYSIS")
    print_separator()

    counter = Counter(
        type(record).__name__
        for record in records
    )

    for record_type, count in counter.items():

        percentage = (
            count / len(records) * 100
            if records
            else 0
        )

        print(
            f"{record_type:15} "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    print()


# ============================================================
# 7. DICTIONARY FIELD ANALYSIS
# ============================================================

def analyze_dictionary_fields(records: list) -> None:
    """Analyze fields used by dictionary records."""

    print_separator()
    print("FIELD ANALYSIS")
    print_separator()

    dictionary_records = [
        record
        for record in records
        if isinstance(record, dict)
    ]

    if not dictionary_records:

        print(
            "No dictionary records were found."
        )

        return

    field_counter = Counter()

    for record in dictionary_records:

        field_counter.update(
            record.keys()
        )

    print(
        f"Dictionary records: "
        f"{len(dictionary_records):,}"
    )

    print()

    print("Fields found:")

    for field, count in field_counter.most_common():

        percentage = (
            count / len(dictionary_records) * 100
        )

        print(
            f"  {field}: "
            f"{count:,} records "
            f"({percentage:.2f}%)"
        )

    print()


# ============================================================
# 8. TEXT ANALYSIS
# ============================================================

def analyze_text_fields(records: list) -> None:
    """Find text fields and calculate basic statistics."""

    print_separator()
    print("TEXT FIELD ANALYSIS")
    print_separator()

    text_values = []

    for record in records:

        if isinstance(record, str):

            text_values.append(record)

        elif isinstance(record, dict):

            for value in record.values():

                if isinstance(value, str):

                    text_values.append(value)

    if not text_values:

        print(
            "No text values were found."
        )

        return

    lengths = [
        len(text)
        for text in text_values
    ]

    total_characters = sum(lengths)

    average_length = (
        total_characters / len(lengths)
    )

    print(
        f"Text values found: "
        f"{len(text_values):,}"
    )

    print(
        f"Total characters: "
        f"{total_characters:,}"
    )

    print(
        f"Average characters: "
        f"{average_length:,.1f}"
    )

    print(
        f"Shortest text: "
        f"{min(lengths):,} characters"
    )

    print(
        f"Longest text: "
        f"{max(lengths):,} characters"
    )

    print()


# ============================================================
# 9. EMPTY VALUE CHECK
# ============================================================

def check_empty_values(records: list) -> None:
    """Look for empty values in dictionary records."""

    print_separator()
    print("EMPTY VALUE CHECK")
    print_separator()

    dictionary_records = [
        record
        for record in records
        if isinstance(record, dict)
    ]

    if not dictionary_records:

        print(
            "No dictionary records available "
            "for this check."
        )

        return

    empty_counter = Counter()

    for record in dictionary_records:

        for key, value in record.items():

            is_empty = (
                value is None
                or value == ""
                or value == []
                or value == {}
            )

            if is_empty:
                empty_counter[key] += 1

    if not empty_counter:

        print(
            "No empty values detected."
        )

    else:

        print(
            "Empty values found:"
        )

        for key, count in empty_counter.most_common():

            print(
                f"  {key}: "
                f"{count:,}"
            )

    print()


# ============================================================
# 10. FINAL RESULT
# ============================================================

def print_final_result(
    data: object,
    records: list
) -> None:
    """Print final verification result."""

    print_separator()
    print("VERIFICATION RESULT")
    print_separator()

    print("PASS: JSON file can be loaded.")
    print("PASS: JSON structure is readable.")
    print(
        f"PASS: {len(records):,} record(s) detected."
    )

    print()

    print(
        "The cleaned dataset was NOT modified."
    )

    print()

    print(
        "Next step: use the observed schema "
        "to build the next pipeline stage."
    )

    print()

    print_separator()
    print("VERIFICATION COMPLETE")
    print_separator()


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main() -> None:

    print_separator()
    print("CONSTRAINTPRESERVER")
    print("STEP 32 — VERIFY CLEANED DOCUMENTS")
    print_separator()
    print()

    try:

        data = load_json()

        print(
            "JSON loaded successfully."
        )

        print()

        validate_top_level(data)

        records = get_records(data)

        inspect_records(records)

        analyze_record_types(records)

        analyze_dictionary_fields(records)

        analyze_text_fields(records)

        check_empty_values(records)

        print_final_result(
            data,
            records
        )

    except json.JSONDecodeError as error:

        print()
        print_separator()
        print("ERROR")
        print_separator()

        print(
            "The cleaned_documents.json file "
            "contains invalid JSON."
        )

        print()
        print(
            f"JSON error: {error}"
        )

        raise SystemExit(1)

    except Exception as error:

        print()
        print_separator()
        print("ERROR")
        print_separator()

        print(
            str(error)
        )

        raise SystemExit(1)


# ============================================================
# 12. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()