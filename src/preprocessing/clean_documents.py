"""
Clean the raw policy documents for downstream processing.

This script:

1. Reads the original .txt files from data/raw/corpus/
2. Normalizes whitespace and line endings
3. Preserves the original wording
4. Removes obvious formatting noise
5. Keeps paragraph boundaries
6. Creates cleaned_documents.json
7. Does NOT modify the original .txt files

The cleaned text will later be used for:
- sentence segmentation
- policy constraint extraction
- retrieval
- classification
- evaluation
"""

from pathlib import Path
import json
import re


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CORPUS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "corpus"
)

DOCUMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cleaned_documents.json"
)


# ============================================================
# 2. FILE READING
# ============================================================

def read_text_file(file_path: Path) -> str:
    """
    Read a policy text file.

    UTF-8 with BOM is attempted first.
    Windows-1252 is used as a fallback.
    """

    try:

        return file_path.read_text(
            encoding="utf-8-sig"
        )

    except UnicodeDecodeError:

        return file_path.read_text(
            encoding="cp1252",
            errors="replace"
        )


# ============================================================
# 3. TEXT NORMALIZATION
# ============================================================

def normalize_line_endings(text: str) -> str:
    """
    Convert different line-ending formats to \\n.
    """

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    return text


def normalize_unicode_spacing(text: str) -> str:
    """
    Normalize common whitespace characters.

    This does not alter normal punctuation or wording.
    """

    # Non-breaking space
    text = text.replace(
        "\u00a0",
        " "
    )

    # Zero-width space
    text = text.replace(
        "\u200b",
        ""
    )

    # Zero-width non-joiner
    text = text.replace(
        "\u200c",
        ""
    )

    # Zero-width joiner
    text = text.replace(
        "\u200d",
        ""
    )

    return text


def normalize_spaces(text: str) -> str:
    """
    Normalize repeated spaces and tabs.

    Newlines are preserved.
    """

    lines = text.split("\n")

    cleaned_lines = []

    for line in lines:

        line = re.sub(
            r"[ \t]+",
            " ",
            line
        )

        cleaned_lines.append(
            line.strip()
        )

    return "\n".join(
        cleaned_lines
    )


# ============================================================
# 4. REMOVE OBVIOUS FORMATTING NOISE
# ============================================================

def remove_page_break_markers(text: str) -> str:
    """
    Remove common page-break characters.

    This targets formatting artifacts rather than policy
    language.
    """

    text = text.replace(
        "\f",
        "\n"
    )

    return text


def remove_excessive_blank_lines(text: str) -> str:
    """
    Reduce excessive blank lines while preserving
    paragraph boundaries.
    """

    text = re.sub(
        r"\n[ \t]*\n[ \t]*\n+",
        "\n\n",
        text
    )

    return text


# ============================================================
# 5. JOIN BROKEN LINES
# ============================================================

def join_broken_lines(text: str) -> str:
    """
    Join lines that appear to be ordinary wrapped text.

    Paragraph boundaries are preserved.

    Example:

        This is a sentence that was
        broken across two lines.

    becomes:

        This is a sentence that was broken across two lines.

    Lines that appear to represent headings, bullets, or
    numbered items are preserved.
    """

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    cleaned_paragraphs = []

    for paragraph in paragraphs:

        lines = [
            line.strip()
            for line in paragraph.split("\n")
            if line.strip()
        ]

        if not lines:
            continue

        joined_lines = []

        for line in lines:

            # Preserve bullet-style lines.
            if re.match(
                r"^[-*•]\s+",
                line
            ):
                joined_lines.append(
                    line
                )
                continue

            # Preserve numbered lists.
            if re.match(
                r"^\(?\d+[.)]\s+",
                line
            ):
                joined_lines.append(
                    line
                )
                continue

            # Preserve lettered lists.
            if re.match(
                r"^\(?[A-Za-z][.)]\s+",
                line
            ):
                joined_lines.append(
                    line
                )
                continue

            # Preserve heading-like lines.
            if (
                len(line) < 100
                and not re.search(
                    r"[.!?:;,]$",
                    line
                )
            ):
                joined_lines.append(
                    line
                )
                continue

            if not joined_lines:

                joined_lines.append(
                    line
                )

            else:

                previous = joined_lines[-1]

                # If the previous line looks like a
                # heading/list item, start a new line.
                if (
                    re.match(
                        r"^[-*•]\s+",
                        previous
                    )
                    or re.match(
                        r"^\(?\d+[.)]\s+",
                        previous
                    )
                    or re.match(
                        r"^\(?[A-Za-z][.)]\s+",
                        previous
                    )
                ):

                    joined_lines.append(
                        line
                    )

                else:

                    joined_lines[-1] = (
                        previous
                        + " "
                        + line
                    )

        cleaned_paragraphs.append(
            "\n".join(joined_lines)
        )

    return "\n\n".join(
        cleaned_paragraphs
    )


# ============================================================
# 6. CLEAN COMPLETE DOCUMENT
# ============================================================

def clean_document_text(text: str) -> str:
    """
    Apply all cleaning operations in a controlled order.
    """

    text = normalize_line_endings(
        text
    )

    text = normalize_unicode_spacing(
        text
    )

    text = remove_page_break_markers(
        text
    )

    text = normalize_spaces(
        text
    )

    text = remove_excessive_blank_lines(
        text
    )

    text = join_broken_lines(
        text
    )

    text = remove_excessive_blank_lines(
        text
    )

    return text.strip()


# ============================================================
# 7. BASIC TEXT STATISTICS
# ============================================================

def count_words(text: str) -> int:
    """
    Count word-like tokens.
    """

    words = re.findall(
        r"\b[\w'-]+\b",
        text,
        flags=re.UNICODE
    )

    return len(words)


def count_characters(text: str) -> int:
    """
    Count characters including spaces.
    """

    return len(text)


def count_lines(text: str) -> int:
    """
    Count non-empty lines.
    """

    if not text.strip():
        return 0

    return sum(
        1
        for line in text.splitlines()
        if line.strip()
    )


def count_paragraphs(text: str) -> int:
    """
    Count paragraphs separated by blank lines.
    """

    if not text.strip():
        return 0

    paragraphs = re.split(
        r"\n\s*\n",
        text.strip()
    )

    return len(
        [
            paragraph
            for paragraph in paragraphs
            if paragraph.strip()
        ]
    )


# ============================================================
# 8. LOAD DOCUMENT METADATA
# ============================================================

def load_documents_metadata():
    """
    Load documents.json created in the previous step.
    """

    if not DOCUMENTS_FILE.exists():

        raise FileNotFoundError(
            "\nCould not find documents.json.\n\n"
            f"Expected:\n{DOCUMENTS_FILE}\n\n"
            "Run build_documents_json.py first."
        )

    with DOCUMENTS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(
            file
        )

    if not isinstance(
        documents,
        list
    ):

        raise ValueError(
            "documents.json must contain a JSON list."
        )

    return documents


# ============================================================
# 9. BUILD CLEANED DOCUMENT
# ============================================================

def build_cleaned_document(
    metadata: dict
):
    """
    Combine the existing metadata with cleaned text.
    """

    filename = metadata[
        "filename"
    ]

    file_path = (
        CORPUS_DIR
        / filename
    )

    if not file_path.exists():

        raise FileNotFoundError(
            "\nPolicy file referenced by documents.json "
            "was not found:\n"
            f"{file_path}"
        )

    raw_text = read_text_file(
        file_path
    )

    cleaned_text = clean_document_text(
        raw_text
    )

    cleaned_document = {
        "document_id": metadata[
            "document_id"
        ],

        "institution": metadata[
            "institution"
        ],

        "year": metadata[
            "year"
        ],

        "filename": metadata[
            "filename"
        ],

        "relative_path": metadata[
            "relative_path"
        ],

        "domain": metadata[
            "domain"
        ],

        "document_type": metadata[
            "document_type"
        ],

        "text": cleaned_text,

        "word_count": count_words(
            cleaned_text
        ),

        "character_count": count_characters(
            cleaned_text
        ),

        "line_count": count_lines(
            cleaned_text
        ),

        "paragraph_count": count_paragraphs(
            cleaned_text
        )
    }

    return cleaned_document


# ============================================================
# 10. VALIDATION
# ============================================================

def validate_documents(
    documents
):
    """
    Validate the cleaned document collection.
    """

    required_fields = {
        "document_id",
        "institution",
        "year",
        "filename",
        "relative_path",
        "domain",
        "document_type",
        "text",
        "word_count",
        "character_count",
        "line_count",
        "paragraph_count"
    }

    document_ids = set()

    filenames = set()

    for document in documents:

        # ----------------------------------------------------
        # Required fields
        # ----------------------------------------------------

        missing_fields = (
            required_fields
            - set(document.keys())
        )

        if missing_fields:

            raise ValueError(
                f"{document.get('document_id', 'UNKNOWN')} "
                f"is missing fields: "
                f"{sorted(missing_fields)}"
            )

        # ----------------------------------------------------
        # Unique document ID
        # ----------------------------------------------------

        document_id = document[
            "document_id"
        ]

        if document_id in document_ids:

            raise ValueError(
                f"Duplicate document ID: "
                f"{document_id}"
            )

        document_ids.add(
            document_id
        )

        # ----------------------------------------------------
        # Unique filename
        # ----------------------------------------------------

        filename = document[
            "filename"
        ]

        if filename in filenames:

            raise ValueError(
                f"Duplicate filename: "
                f"{filename}"
            )

        filenames.add(
            filename
        )

        # ----------------------------------------------------
        # Text validation
        # ----------------------------------------------------

        text = document[
            "text"
        ]

        if not isinstance(
            text,
            str
        ):

            raise ValueError(
                f"Text is not a string for "
                f"{document_id}"
            )

        if not text.strip():

            raise ValueError(
                f"Empty cleaned text for "
                f"{document_id}"
            )

        # ----------------------------------------------------
        # Statistic validation
        # ----------------------------------------------------

        if document[
            "word_count"
        ] <= 0:

            raise ValueError(
                f"Invalid word count for "
                f"{document_id}"
            )

        if document[
            "character_count"
        ] <= 0:

            raise ValueError(
                f"Invalid character count for "
                f"{document_id}"
            )

        if document[
            "line_count"
        ] <= 0:

            raise ValueError(
                f"Invalid line count for "
                f"{document_id}"
            )

        if document[
            "paragraph_count"
        ] <= 0:

            raise ValueError(
                f"Invalid paragraph count for "
                f"{document_id}"
            )


# ============================================================
# 11. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CONSTRAINTPRESERVER")
    print("CLEANING POLICY DOCUMENTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Check corpus
    # --------------------------------------------------------

    if not CORPUS_DIR.exists():

        raise FileNotFoundError(
            "\nCorpus directory not found:\n"
            f"{CORPUS_DIR}"
        )

    # --------------------------------------------------------
    # Load metadata
    # --------------------------------------------------------

    documents_metadata = (
        load_documents_metadata()
    )

    print()
    print(
        f"Loaded {len(documents_metadata)} "
        f"document records from documents.json."
    )

    # --------------------------------------------------------
    # Process documents
    # --------------------------------------------------------

    cleaned_documents = []

    print()
    print("Cleaning documents...")
    print()

    for index, metadata in enumerate(
        documents_metadata,
        start=1
    ):

        cleaned_document = (
            build_cleaned_document(
                metadata
            )
        )

        cleaned_documents.append(
            cleaned_document
        )

        print(
            f"[{index:03d}/"
            f"{len(documents_metadata):03d}] "
            f"{metadata['filename']}"
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print()
    print("Validating cleaned documents...")

    validate_documents(
        cleaned_documents
    )

    print(
        "Validation passed."
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            cleaned_documents,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_words = sum(
        document["word_count"]
        for document in cleaned_documents
    )

    total_characters = sum(
        document["character_count"]
        for document in cleaned_documents
    )

    total_paragraphs = sum(
        document["paragraph_count"]
        for document in cleaned_documents
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLEANING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Documents processed : "
        f"{len(cleaned_documents)}"
    )

    print(
        f"Total words        : "
        f"{total_words:,}"
    )

    print(
        f"Total characters   : "
        f"{total_characters:,}"
    )

    print(
        f"Total paragraphs   : "
        f"{total_paragraphs:,}"
    )

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# 12. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()