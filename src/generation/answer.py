import re
import sys

from src.retrieval.search import load_index, load_model, search


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5

# Minimum FAISS similarity required before evidence
# is considered strong enough to use.
MIN_SCORE = 0.45

# Maximum number of sentences included in the answer.
MAX_SENTENCES = 6


# ============================================================
# RETRIEVE EVIDENCE
# ============================================================

def retrieve_evidence(query, model, index, metadata):
    """
    Retrieve the most relevant policy chunks.

    Only chunks meeting the minimum similarity threshold
    are retained as evidence.
    """

    results = search(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
        top_k=DEFAULT_TOP_K,
    )

    evidence = [
        result
        for result in results
        if result.get("score", 0.0) >= MIN_SCORE
    ]

    return evidence


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def split_sentences(text):
    """
    Split policy text into readable sentences.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 30
    ]


# ============================================================
# SELECT RELEVANT SENTENCES
# ============================================================

def select_relevant_sentences(query, evidence):
    """
    Select the strongest policy sentences using both:

    1. Semantic similarity from FAISS
    2. Keyword overlap with the query

    Semantic similarity remains the primary signal.

    The answer is extractive, meaning we only return
    sentences that already exist in the retrieved policy text.
    """

    query_words = {
        word.lower()
        for word in re.findall(
            r"[a-zA-Z]{4,}",
            query,
        )
    }

    candidates = []

    for result in evidence:

        text = result.get("text", "")

        for sentence in split_sentences(text):

            sentence_words = {
                word.lower()
                for word in re.findall(
                    r"[a-zA-Z]{4,}",
                    sentence,
                )
            }

            overlap = len(
                query_words & sentence_words
            )

            # Normalize keyword overlap.
            overlap_score = min(
                overlap / max(len(query_words), 1),
                1.0,
            )

            # FAISS semantic similarity.
            semantic_score = result.get(
                "score",
                0.0,
            )

            # Semantic similarity is the primary signal.
            combined_score = (
                0.75 * semantic_score
                + 0.25 * overlap_score
            )

            candidates.append(
                {
                    "sentence": sentence,
                    "overlap": overlap,
                    "overlap_score": overlap_score,
                    "semantic_score": semantic_score,
                    "combined_score": combined_score,

                    # Source information.
                    "institution": result.get(
                        "institution"
                    ),
                    "year": result.get(
                        "year"
                    ),
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "document_id": result.get(
                        "document_id"
                    ),
                    "filename": result.get(
                        "filename"
                    ),
                }
            )

    # Highest combined relevance first.
    candidates.sort(
        key=lambda item: item["combined_score"],
        reverse=True,
    )

    selected = []

    # Used to avoid repeating the same sentence.
    seen = set()

    for candidate in candidates:

        sentence = candidate["sentence"]

        normalized = sentence.lower()

        if normalized in seen:
            continue

        seen.add(normalized)

        selected.append(candidate)

        if len(selected) >= MAX_SENTENCES:
            break

    return selected


# ============================================================
# BUILD ANSWER
# ============================================================

def build_answer(query, evidence):
    """
    Build a conservative answer directly from
    retrieved policy evidence.

    No unsupported policy statements are generated.
    """

    # --------------------------------------------------------
    # No evidence
    # --------------------------------------------------------

    if not evidence:
        return (
            "INSUFFICIENT EVIDENCE\n\n"
            "I could not find enough relevant policy evidence "
            "to answer this question reliably."
        ), []

    # --------------------------------------------------------
    # Select relevant sentences
    # --------------------------------------------------------

    selected = select_relevant_sentences(
        query=query,
        evidence=evidence,
    )

    # --------------------------------------------------------
    # No sufficiently relevant sentences
    # --------------------------------------------------------

    if not selected:
        return (
            "INSUFFICIENT EVIDENCE\n\n"
            "The retrieved policy passages did not contain "
            "enough directly relevant information."
        ), []

    # --------------------------------------------------------
    # Build grounded answer
    # --------------------------------------------------------

    answer_parts = []

    for item in selected:

        sentence = item["sentence"]

        institution = item.get(
            "institution",
            "Unknown institution",
        )

        year = item.get(
            "year",
            "Unknown year",
        )

        chunk_id = item.get(
            "chunk_id",
            "Unknown chunk",
        )

        answer_parts.append(
            f"{sentence} "
            f"[{institution}, {year}, {chunk_id}]"
        )

    answer = " ".join(answer_parts)

    return answer, selected


# ============================================================
# DISPLAY ANSWER
# ============================================================

def display_answer(
    query,
    answer,
    evidence,
    selected,
):
    """
    Display the grounded answer and the supporting
    policy sources.
    """

    print()
    print("=" * 70)
    print("CONSTRAINT PRESERVER - ANSWER")
    print("=" * 70)

    print()
    print(f"Question: {query}")

    print()
    print("Answer:")
    print(answer)

    # --------------------------------------------------------
    # Sources
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUPPORTING POLICY SOURCES")
    print("=" * 70)

    shown_sources = set()

    for item in selected:

        source_key = (
            item.get("institution"),
            item.get("year"),
            item.get("chunk_id"),
        )

        if source_key in shown_sources:
            continue

        shown_sources.add(source_key)

        print()

        print(
            f"Institution : "
            f"{item.get('institution')}"
        )

        print(
            f"Year        : "
            f"{item.get('year')}"
        )

        print(
            f"Document ID : "
            f"{item.get('document_id')}"
        )

        print(
            f"Document    : "
            f"{item.get('filename')}"
        )

        print(
            f"Chunk       : "
            f"{item.get('chunk_id')}"
        )

        print(
            f"Similarity  : "
            f"{item.get('semantic_score', 0.0):.4f}"
        )

        print(
            f"Relevance   : "
            f"{item.get('combined_score', 0.0):.4f}"
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate command-line input
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()
        print("Usage:")

        print(
            'python -m src.generation.answer '
            '"Can AI be used for administrative work?"'
        )

        return

    # Combine all command-line arguments into one question.
    query = " ".join(
        sys.argv[1:]
    ).strip()

    if not query:
        print(
            "Question cannot be empty."
        )
        return

    # --------------------------------------------------------
    # Load retrieval system
    # --------------------------------------------------------

    print(
        "Loading retrieval system..."
    )

    index, metadata = load_index()

    model = load_model()

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    print()
    print(
        "Searching policy corpus..."
    )

    evidence = retrieve_evidence(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
    )

    # --------------------------------------------------------
    # Build answer
    # --------------------------------------------------------

    answer, selected = build_answer(
        query=query,
        evidence=evidence,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_answer(
        query=query,
        answer=answer,
        evidence=evidence,
        selected=selected,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()