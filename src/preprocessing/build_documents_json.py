"""
Build documents.json from the ConstraintPreserver policy corpus.

This script:
1. Finds every .txt policy document in data/raw/corpus/
2. Extracts basic metadata from each filename
3. Calculates basic document statistics
4. Creates data/processed/documents.json

The original policy files are never modified.
"""

from pathlib import Path
import json
import re
import statistics


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root:
# ConstraintPreserver/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of the policy documents
CORPUS_DIR = PROJECT_ROOT / "data" / "raw" / "corpus"

# Location where documents.json will be created
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "documents.json"


# ============================================================
# 2. TEXT FUNCTIONS
# ============================================================

def read_text_file(file_path: Path) -> str:
    """
    Read a text file.

    UTF-8 is attempted first.
    If that fails, Windows-1252 is used as a fallback.
    """

    try:
        return file_path.read_text(encoding="utf-8-sig")

    except UnicodeDecodeError:
        return file_path.read_text(
            encoding="cp1252",
            errors="replace"
        )


def clean_text(text: str) -> str:
    """
    Perform basic text normalization.

    This does NOT modify the original file.
    It only prepares text for counting.
    """

    # Normalize Windows/Mac line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


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


def count_sentences(text: str) -> int:
    """
    Approximate the number of sentences.

    This is only for corpus statistics.
    It is NOT our final NLP sentence segmentation.
    """

    if not text.strip():
        return 0

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return len(sentences)


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

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    return len(paragraphs)


# ============================================================
# 3. FILENAME METADATA
# ============================================================

def parse_filename(filename: str):
    """
    Extract institution and year from the filename.

    Example:

        Cornell_University_2023.txt

    becomes:

        institution = Cornell University
        year = 2023
    """

    # Remove .txt
    stem = Path(filename).stem

    # Look for a year at the end.
    year_match = re.search(
        r"_(20\d{2})$",
        stem
    )

    if year_match:

        year = int(
            year_match.group(1)
        )

        institution_raw = stem[
            :year_match.start()
        ]

    else:

        year = None
        institution_raw = stem

    # Convert underscores to spaces
    institution = institution_raw.replace(
        "_",
        " "
    )

    # Remove repeated whitespace
    institution = re.sub(
        r"\s+",
        " ",
        institution
    ).strip()

    return institution, year


# ============================================================
# 4. DOCUMENT ID
# ============================================================

def create_document_id(index: int) -> str:
    """
    Create a deterministic document ID.

    Examples:

        POL_001
        POL_002
        POL_003
    """

    return f"POL_{index:03d}"


# ============================================================
# 5. FIND POLICY FILES
# ============================================================

def find_policy_files():
    """
    Find all .txt files under the corpus directory.

    rglob() searches recursively, so this also works if the
    extracted dataset contains an extra nested folder.
    """

    files = [
        path
        for path in CORPUS_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".txt"
    ]

    # Sort so document IDs are reproducible.
    files.sort(
        key=lambda path: path.relative_to(
            CORPUS_DIR
        ).as_posix().lower()
    )

    return files


# ============================================================
# 6. BUILD DOCUMENT RECORD
# ============================================================

def build_document_record(
    index: int,
    file_path: Path
):
    """
    Create one metadata record for one policy document.
    """

    # Read original document
    raw_text = read_text_file(file_path)

    # Clean only for analysis/statistics
    text = clean_text(raw_text)

    # Extract institution/year
    institution, year = parse_filename(
        file_path.name
    )

    # Create path relative to project root
    relative_path = file_path.relative_to(
        PROJECT_ROOT
    ).as_posix()

    document = {
        "document_id": create_document_id(index),

        "institution": institution,

        "year": year,

        "filename": file_path.name,

        "relative_path": relative_path,

        "domain": "Institutional Generative AI Policy",

        "document_type": "Policy/Guideline",

        "word_count": count_words(text),

        "sentence_count": count_sentences(text),

        "paragraph_count": count_paragraphs(text)
    }

    return document


# ============================================================
# 7. VALIDATION
# ============================================================

def validate_documents(documents):
    """
    Validate the generated metadata before saving it.
    """

    required_fields = {
        "document_id",
        "institution",
        "year",
        "filename",
        "relative_path",
        "domain",
        "document_type",
        "word_count",
        "sentence_count",
        "paragraph_count"
    }

    # --------------------------------------------------------
    # Check document IDs
    # --------------------------------------------------------

    document_ids = [
        document["document_id"]
        for document in documents
    ]

    if len(document_ids) != len(
        set(document_ids)
    ):
        raise ValueError(
            "Duplicate document IDs detected."
        )

    # --------------------------------------------------------
    # Check filenames
    # --------------------------------------------------------

    filenames = [
        document["filename"]
        for document in documents
    ]

    if len(filenames) != len(
        set(filenames)
    ):
        raise ValueError(
            "Duplicate filenames detected."
        )

    # --------------------------------------------------------
    # Check required fields
    # --------------------------------------------------------

    for document in documents:

        missing_fields = (
            required_fields
            - set(document.keys())
        )

        if missing_fields:

            raise ValueError(
                f"{document['document_id']} "
                f"is missing fields: "
                f"{missing_fields}"
            )

        # Basic sanity checks
        if document["word_count"] < 0:
            raise ValueError(
                f"Invalid word count for "
                f"{document['document_id']}"
            )

        if document["sentence_count"] < 0:
            raise ValueError(
                f"Invalid sentence count for "
                f"{document['document_id']}"
            )

        if document["paragraph_count"] < 0:
            raise ValueError(
                f"Invalid paragraph count for "
                f"{document['document_id']}"
            )


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CONSTRAINTPRESERVER")
    print("BUILDING documents.json")
    print("=" * 70)

    # --------------------------------------------------------
    # Check corpus directory
    # --------------------------------------------------------

    if not CORPUS_DIR.exists():

        raise FileNotFoundError(
            "\nThe corpus directory was not found.\n\n"
            f"Expected location:\n{CORPUS_DIR}\n\n"
            "Make sure your .txt policy files are inside:\n"
            "data/raw/corpus/"
        )

    print()
    print("Corpus directory:")
    print(CORPUS_DIR)

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    policy_files = find_policy_files()

    if not policy_files:

        raise FileNotFoundError(
            "\nNo .txt files were found.\n\n"
            f"Checked:\n{CORPUS_DIR}"
        )

    print()
    print(
        f"Found {len(policy_files)} "
        f".txt policy files."
    )

    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    documents = []

    print()
    print("Processing documents...")
    print()

    for index, file_path in enumerate(
        policy_files,
        start=1
    ):

        document = build_document_record(
            index,
            file_path
        )

        documents.append(document)

        print(
            f"[{index:03d}/{len(policy_files):03d}] "
            f"{file_path.name}"
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    print()
    print("Validating metadata...")

    validate_documents(documents)

    print("Validation passed.")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        json.dump(
            documents,
            output_file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Calculate corpus statistics
    # --------------------------------------------------------

    word_counts = [
        document["word_count"]
        for document in documents
    ]

    sentence_counts = [
        document["sentence_count"]
        for document in documents
    ]

    paragraph_counts = [
        document["paragraph_count"]
        for document in documents
    ]

    total_words = sum(word_counts)
    total_sentences = sum(sentence_counts)
    total_paragraphs = sum(paragraph_counts)

    average_words = (
        statistics.mean(word_counts)
        if word_counts
        else 0
    )

    median_words = (
        statistics.median(word_counts)
        if word_counts
        else 0
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOCUMENT METADATA CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(f"Documents        : {len(documents)}")
    print(f"Total words      : {total_words:,}")
    print(f"Average words    : {average_words:,.1f}")
    print(f"Median words     : {median_words:,.1f}")
    print(f"Total sentences  : {total_sentences:,}")
    print(f"Total paragraphs : {total_paragraphs:,}")

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()