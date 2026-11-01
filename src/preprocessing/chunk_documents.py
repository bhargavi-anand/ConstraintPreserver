"""
ConstraintPreserver
===================

PHASE 4: DOCUMENT CHUNKING

Reads:
    data/processed/cleaned_documents.json

Creates:
    data/processed/chunks.json

Purpose:
    Split cleaned policy documents into retrieval-friendly chunks
    while preserving document metadata and paragraph boundaries.

Important design goals:
    1. Never modify the cleaned source dataset.
    2. Preserve document-level metadata.
    3. Prefer paragraph boundaries.
    4. Avoid splitting words.
    5. Avoid splitting sentences when possible.
    6. Give every chunk a stable chunk_id.
    7. Preserve the original document order.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "cleaned_documents.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"


# ============================================================
# 2. CHUNK SETTINGS
# ============================================================

# Target size for each chunk.
#
# This is a TARGET, not a hard requirement.
# The algorithm prefers to finish at a paragraph or sentence
# boundary instead of cutting text exactly at this number.
TARGET_WORDS = 450

# Maximum preferred chunk size.
#
# A chunk may exceed TARGET_WORDS slightly when finishing the
# current paragraph/sentence is more appropriate.
MAX_WORDS = 600

# Minimum chunk size.
#
# Small paragraphs are combined with nearby paragraphs whenever
# possible so that we don't create thousands of tiny chunks.
MIN_WORDS = 80


# ============================================================
# 3. BASIC TEXT UTILITIES
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace without changing the actual wording.

    Multiple spaces become one space.
    Multiple blank lines are preserved separately elsewhere.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        lines.append(line)

    return "\n".join(lines)


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split a document into paragraphs.

    Paragraphs are separated by one or more blank lines.

    If the source document has weak paragraph formatting,
    non-empty lines are still retained rather than discarded.
    """

    text = normalize_whitespace(text)

    raw_paragraphs = re.split(r"\n\s*\n+", text)

    paragraphs = []

    for paragraph in raw_paragraphs:
        paragraph = paragraph.strip()

        if paragraph:
            paragraphs.append(paragraph)

    return paragraphs


def word_count(text: str) -> int:
    """
    Count words using a simple whitespace/token rule.
    """

    return len(re.findall(r"\S+", text))


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    This is intentionally conservative.

    It handles common sentence-ending punctuation:
        .
        !
        ?

    It does not attempt advanced NLP sentence detection.
    """

    text = text.strip()

    if not text:
        return []

    # Split after sentence-ending punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", text)

    return [part.strip() for part in parts if part.strip()]


# ============================================================
# 4. CHUNK A SINGLE PARAGRAPH
# ============================================================

def split_large_paragraph(paragraph: str) -> list[str]:
    """
    Split a paragraph that is too large.

    First try sentence boundaries.

    If a sentence itself is too large, split it by words as a
    final fallback.

    This prevents a single extremely large paragraph from
    producing an oversized chunk.
    """

    paragraph_words = word_count(paragraph)

    if paragraph_words <= MAX_WORDS:
        return [paragraph]

    sentences = split_sentences(paragraph)

    if len(sentences) <= 1:
        words = paragraph.split()

        pieces = []

        for start in range(0, len(words), TARGET_WORDS):
            piece = " ".join(words[start:start + TARGET_WORDS])

            if piece.strip():
                pieces.append(piece.strip())

        return pieces

    pieces = []
    current = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)

        # If one sentence itself is too large, break it by words.
        if sentence_words > MAX_WORDS:
            if current:
                pieces.append(" ".join(current).strip())
                current = []
                current_words = 0

            words = sentence.split()

            for start in range(0, len(words), TARGET_WORDS):
                piece = " ".join(words[start:start + TARGET_WORDS])

                if piece.strip():
                    pieces.append(piece.strip())

            continue

        # Normal sentence handling.
        if current and current_words + sentence_words > MAX_WORDS:
            pieces.append(" ".join(current).strip())
            current = []
            current_words = 0

        current.append(sentence)
        current_words += sentence_words

    if current:
        pieces.append(" ".join(current).strip())

    return pieces


# ============================================================
# 5. BUILD CHUNKS FOR ONE DOCUMENT
# ============================================================

def build_document_chunks(document: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one cleaned document into multiple chunks.
    """

    document_id = str(document.get("document_id", "")).strip()

    if not document_id:
        raise ValueError("Document is missing document_id.")

    text = document.get("text", "")

    if not isinstance(text, str):
        raise ValueError(
            f"Document {document_id} has a non-string text field."
        )

    text = text.strip()

    if not text:
        return []

    paragraphs = split_into_paragraphs(text)

    # --------------------------------------------------------
    # Expand very large paragraphs into sentence-level pieces.
    # --------------------------------------------------------

    paragraph_pieces = []

    for paragraph_index, paragraph in enumerate(paragraphs):

        pieces = split_large_paragraph(paragraph)

        for piece_index, piece in enumerate(pieces):

            paragraph_pieces.append(
                {
                    "text": piece,
                    "source_paragraph_index": paragraph_index,
                    "source_piece_index": piece_index,
                }
            )

    # --------------------------------------------------------
    # Group paragraph pieces into retrieval chunks.
    # --------------------------------------------------------

    chunks = []

    current_parts = []
    current_word_count = 0
    current_start_paragraph = None
    current_end_paragraph = None

    for piece in paragraph_pieces:

        piece_text = piece["text"]
        piece_words = word_count(piece_text)

        # Start a new chunk.
        if not current_parts:
            current_parts = [piece_text]
            current_word_count = piece_words
            current_start_paragraph = piece["source_paragraph_index"]
            current_end_paragraph = piece["source_paragraph_index"]
            continue

        proposed_count = current_word_count + piece_words

        # ----------------------------------------------------
        # If adding the piece would exceed MAX_WORDS,
        # finalize the current chunk.
        # ----------------------------------------------------

        if proposed_count > MAX_WORDS:

            chunk_text = "\n\n".join(current_parts).strip()

            chunks.append(
                {
                    "document_id": document_id,
                    "institution": document.get("institution"),
                    "year": document.get("year"),
                    "filename": document.get("filename"),
                    "relative_path": document.get("relative_path"),
                    "domain": document.get("domain"),
                    "document_type": document.get("document_type"),
                    "chunk_index": len(chunks),
                    "source_paragraph_start": current_start_paragraph,
                    "source_paragraph_end": current_end_paragraph,
                    "text": chunk_text,
                    "word_count": word_count(chunk_text),
                    "character_count": len(chunk_text),
                }
            )

            # Start new chunk.
            current_parts = [piece_text]
            current_word_count = piece_words
            current_start_paragraph = piece["source_paragraph_index"]
            current_end_paragraph = piece["source_paragraph_index"]

            continue

        # ----------------------------------------------------
        # If current chunk is already large enough and adding
        # another piece would take it beyond the target,
        # finalize it.
        # ----------------------------------------------------

        if (
            current_word_count >= MIN_WORDS
            and proposed_count > TARGET_WORDS
        ):

            chunk_text = "\n\n".join(current_parts).strip()

            chunks.append(
                {
                    "document_id": document_id,
                    "institution": document.get("institution"),
                    "year": document.get("year"),
                    "filename": document.get("filename"),
                    "relative_path": document.get("relative_path"),
                    "domain": document.get("domain"),
                    "document_type": document.get("document_type"),
                    "chunk_index": len(chunks),
                    "source_paragraph_start": current_start_paragraph,
                    "source_paragraph_end": current_end_paragraph,
                    "text": chunk_text,
                    "word_count": word_count(chunk_text),
                    "character_count": len(chunk_text),
                }
            )

            current_parts = [piece_text]
            current_word_count = piece_words
            current_start_paragraph = piece["source_paragraph_index"]
            current_end_paragraph = piece["source_paragraph_index"]

            continue

        # ----------------------------------------------------
        # Otherwise append the piece to the current chunk.
        # ----------------------------------------------------

        current_parts.append(piece_text)
        current_word_count = proposed_count
        current_end_paragraph = piece["source_paragraph_index"]

    # --------------------------------------------------------
    # Add final chunk.
    # --------------------------------------------------------

    if current_parts:

        chunk_text = "\n\n".join(current_parts).strip()

        chunks.append(
            {
                "document_id": document_id,
                "institution": document.get("institution"),
                "year": document.get("year"),
                "filename": document.get("filename"),
                "relative_path": document.get("relative_path"),
                "domain": document.get("domain"),
                "document_type": document.get("document_type"),
                "chunk_index": len(chunks),
                "source_paragraph_start": current_start_paragraph,
                "source_paragraph_end": current_end_paragraph,
                "text": chunk_text,
                "word_count": word_count(chunk_text),
                "character_count": len(chunk_text),
            }
        )

    # --------------------------------------------------------
    # Give every chunk a globally unique ID.
    # --------------------------------------------------------

    for chunk in chunks:
        chunk["chunk_id"] = (
            f"{document_id}_chunk_{chunk['chunk_index']:04d}"
        )

    return chunks


# ============================================================
# 6. LOAD CLEANED DATASET
# ============================================================

def load_documents() -> list[dict[str, Any]]:
    """
    Load cleaned_documents.json and safely determine its structure.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nCleaned dataset was not found.\n\n"
            f"Expected:\n{INPUT_FILE}\n\n"
            "Run the cleaning stage first."
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Dataset is directly a list.
    if isinstance(data, list):
        return data

    # Dataset wrapped in "documents".
    if isinstance(data, dict) and isinstance(
        data.get("documents"), list
    ):
        return data["documents"]

    # Dataset wrapped in "data".
    if isinstance(data, dict) and isinstance(
        data.get("data"), list
    ):
        return data["data"]

    raise ValueError(
        "Could not determine the document list structure "
        "inside cleaned_documents.json."
    )


# ============================================================
# 7. BUILD ALL CHUNKS
# ============================================================

def build_all_chunks(
    documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    all_chunks = []

    for document_number, document in enumerate(documents, start=1):

        document_chunks = build_document_chunks(document)

        all_chunks.extend(document_chunks)

        if document_number % 10 == 0:
            print(
                f"Processed {document_number:,} documents "
                f"| chunks created: {len(all_chunks):,}"
            )

    return all_chunks


# ============================================================
# 8. VALIDATE CHUNKS
# ============================================================

def validate_chunks(
    chunks: list[dict[str, Any]]
) -> None:

    if not chunks:
        raise ValueError("No chunks were created.")

    required_fields = {
        "chunk_id",
        "document_id",
        "institution",
        "year",
        "filename",
        "relative_path",
        "domain",
        "document_type",
        "chunk_index",
        "source_paragraph_start",
        "source_paragraph_end",
        "text",
        "word_count",
        "character_count",
    }

    chunk_ids = set()

    for index, chunk in enumerate(chunks):

        missing = required_fields - set(chunk.keys())

        if missing:
            raise ValueError(
                f"Chunk {index} is missing fields: {sorted(missing)}"
            )

        chunk_id = chunk["chunk_id"]

        if chunk_id in chunk_ids:
            raise ValueError(
                f"Duplicate chunk_id detected: {chunk_id}"
            )

        chunk_ids.add(chunk_id)

        if not isinstance(chunk["text"], str):
            raise ValueError(
                f"Chunk {chunk_id} has invalid text."
            )

        if not chunk["text"].strip():
            raise ValueError(
                f"Chunk {chunk_id} has empty text."
            )

        actual_words = word_count(chunk["text"])

        if actual_words != chunk["word_count"]:
            raise ValueError(
                f"Word count mismatch for {chunk_id}: "
                f"stored={chunk['word_count']} "
                f"actual={actual_words}"
            )

        actual_characters = len(chunk["text"])

        if actual_characters != chunk["character_count"]:
            raise ValueError(
                f"Character count mismatch for {chunk_id}: "
                f"stored={chunk['character_count']} "
                f"actual={actual_characters}"
            )


# ============================================================
# 9. SAVE DATASET
# ============================================================

def save_chunks(chunks: list[dict[str, Any]]) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# 10. PRINT STATISTICS
# ============================================================

def print_statistics(
    documents: list[dict[str, Any]],
    chunks: list[dict[str, Any]]
) -> None:

    chunk_word_counts = [
        chunk["word_count"]
        for chunk in chunks
    ]

    chunks_per_document = {}

    for chunk in chunks:
        document_id = chunk["document_id"]

        chunks_per_document.setdefault(
            document_id,
            0
        )

        chunks_per_document[document_id] += 1

    print()
    print("=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    print()
    print(f"Documents processed : {len(documents):,}")
    print(f"Total chunks        : {len(chunks):,}")

    print()
    print("Chunk word statistics:")

    print(
        f"  Minimum words     : "
        f"{min(chunk_word_counts):,}"
    )

    print(
        f"  Maximum words     : "
        f"{max(chunk_word_counts):,}"
    )

    print(
        f"  Average words     : "
        f"{statistics.mean(chunk_word_counts):,.1f}"
    )

    print(
        f"  Median words      : "
        f"{statistics.median(chunk_word_counts):,.1f}"
    )

    print()
    print("Chunks per document:")

    document_counts = list(
        chunks_per_document.values()
    )

    print(
        f"  Minimum           : "
        f"{min(document_counts):,}"
    )

    print(
        f"  Maximum           : "
        f"{max(document_counts):,}"
    )

    print(
        f"  Average           : "
        f"{statistics.mean(document_counts):,.1f}"
    )

    print()
    print("Output file:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

def main() -> None:

    print("=" * 70)
    print("ConstraintPreserver - Document Chunking")
    print("=" * 70)

    print()
    print("Input:")
    print(INPUT_FILE)

    print()
    print("Loading cleaned documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents):,} cleaned documents."
    )

    print()
    print("Building chunks...")

    chunks = build_all_chunks(documents)

    print()
    print("Validating chunks...")

    validate_chunks(chunks)

    print("Chunk validation passed.")

    print()
    print("Saving chunks...")

    save_chunks(chunks)

    print_statistics(
        documents,
        chunks
    )


# ============================================================
# 12. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()