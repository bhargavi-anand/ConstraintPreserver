import json
import re
import sys
from pathlib import Path

from src.retrieval.search import load_index, load_model, search


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5
MIN_SCORE = 0.45

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "data" / "constraints"
OUTPUT_FILE = OUTPUT_DIR / "extracted_constraints.json"


# ============================================================
# RETRIEVE POLICY EVIDENCE
# ============================================================

def retrieve_evidence(query, model, index, metadata):
    """
    Retrieve policy chunks relevant to the user's question.
    """

    results = search(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
        top_k=DEFAULT_TOP_K,
    )

    return [
        result
        for result in results
        if result.get("score", 0.0) >= MIN_SCORE
    ]


# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_sentences(text):
    """
    Split policy text into individual sentences.
    """

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) >= 25
    ]


# ============================================================
# CONSTRAINT TYPE DETECTION
# ============================================================

def detect_constraint_type(sentence):
    """
    Detect the most explicit policy constraint type.

    The detector is intentionally conservative.
    """

    text = sentence.lower()

    prohibition_patterns = [
        r"\bmust not\b",
        r"\bmay not\b",
        r"\bdo not\b",
        r"\bshould not\b",
        r"\bnot permitted\b",
        r"\bprohibited\b",
        r"\bnot appropriate\b",
        r"\bnot allowed\b",
        r"\bforbidden\b",
    ]

    requirement_patterns = [
        r"\bmust\b",
        r"\brequired\b",
        r"\brequires\b",
        r"\bneed to\b",
        r"\bshall\b",
        r"\bensure that\b",
    ]

    recommendation_patterns = [
        r"\brecommended\b",
        r"\brecommend\b",
        r"\bshould\b",
        r"\bencouraged\b",
        r"\badvised\b",
    ]

    permission_patterns = [
        r"\bmay be used\b",
        r"\bcan be used\b",
        r"\bmay use\b",
        r"\bcan use\b",
        r"\bacceptable use\b",
        r"\bappropriate use\b",
    ]

    for pattern in prohibition_patterns:
        if re.search(pattern, text):
            return "PROHIBITION"

    for pattern in requirement_patterns:
        if re.search(pattern, text):
            return "REQUIREMENT"

    for pattern in recommendation_patterns:
        if re.search(pattern, text):
            return "RECOMMENDATION"

    for pattern in permission_patterns:
        if re.search(pattern, text):
            return "PERMISSION"

    return "INFORMATION"


# ============================================================
# RISK DETECTION
# ============================================================

def detect_risk(sentence):
    """
    Detect an explicitly stated risk level.
    """

    text = sentence.lower()

    if re.search(r"\bhigh risk\b", text):
        return "HIGH"

    if re.search(r"\bmedium risk\b", text):
        return "MEDIUM"

    if re.search(r"\blow risk\b", text):
        return "LOW"

    return None


# ============================================================
# CONSTRAINT EXTRACTION
# ============================================================

def extract_constraints(query, evidence):
    """
    Extract policy constraints from retrieved evidence.

    The original policy sentence is preserved so that
    every constraint remains traceable to its source.
    """

    constraints = []

    for result in evidence:

        text = result.get(
            "text",
            "",
        )

        sentences = split_sentences(text)

        for sentence in sentences:

            constraint_type = detect_constraint_type(
                sentence
            )

            risk = detect_risk(
                sentence
            )

            # Ignore ordinary informational sentences.
            if constraint_type == "INFORMATION":
                continue

            constraint = {
                "constraint_type": constraint_type,

                "risk": risk,

                "statement": sentence,

                "source": {
                    "institution": result.get(
                        "institution"
                    ),
                    "year": result.get(
                        "year"
                    ),
                    "document_id": result.get(
                        "document_id"
                    ),
                    "filename": result.get(
                        "filename"
                    ),
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "similarity": result.get(
                        "score",
                        0.0,
                    ),
                },
            }

            constraints.append(
                constraint
            )

    return constraints


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def deduplicate_constraints(constraints):
    """
    Remove identical constraints while preserving
    source information.
    """

    unique = []
    seen = set()

    for constraint in constraints:

        source = constraint.get(
            "source",
            {},
        )

        key = (
            constraint.get(
                "constraint_type"
            ),
            constraint.get(
                "statement",
                "",
            ).lower(),
            source.get(
                "institution"
            ),
            source.get(
                "year"
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        unique.append(
            constraint
        )

    return unique


# ============================================================
# SAVE JSON
# ============================================================

def save_constraints(
    query,
    constraints,
):
    """
    Save extracted constraints to JSON.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "query": query,
        "constraint_count": len(
            constraints
        ),
        "constraints": constraints,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return OUTPUT_FILE


# ============================================================
# DISPLAY CONSTRAINTS
# ============================================================

def display_constraints(
    query,
    constraints,
):
    """
    Display structured constraints.
    """

    print()
    print("=" * 70)
    print(
        "CONSTRAINT PRESERVER - "
        "STRUCTURED CONSTRAINTS"
    )
    print("=" * 70)

    print()
    print(
        f"Question: {query}"
    )

    print()
    print(
        f"Constraints found: "
        f"{len(constraints)}"
    )

    if not constraints:

        print()
        print(
            "NO CONSTRAINTS FOUND"
        )

        print()
        print(
            "No explicit policy constraints "
            "were detected in the retrieved evidence."
        )

        print()
        print("=" * 70)

        return

    for number, constraint in enumerate(
        constraints,
        start=1,
    ):

        source = constraint.get(
            "source",
            {},
        )

        print()
        print("-" * 70)

        print(
            f"Constraint #{number}"
        )

        print(
            f"Type       : "
            f"{constraint.get('constraint_type')}"
        )

        print(
            f"Risk       : "
            f"{constraint.get('risk') or 'Not specified'}"
        )

        print(
            f"Institution: "
            f"{source.get('institution')}"
        )

        print(
            f"Year       : "
            f"{source.get('year')}"
        )

        print(
            f"Document   : "
            f"{source.get('filename')}"
        )

        print(
            f"Chunk      : "
            f"{source.get('chunk_id')}"
        )

        print(
            f"Similarity : "
            f"{source.get('similarity', 0.0):.4f}"
        )

        print()
        print(
            "Policy statement:"
        )

        print(
            constraint.get(
                "statement"
            )
        )

    print()
    print("=" * 70)

    print()
    print(
        f"Saved JSON:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print()
        print("Usage:")

        print(
            'python -m src.constraints.extractor '
            '"What restrictions apply to AI tools?"'
        )

        return

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
    # Retrieve evidence
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
    # Extract constraints
    # --------------------------------------------------------

    print()
    print(
        "Extracting policy constraints..."
    )

    constraints = extract_constraints(
        query=query,
        evidence=evidence,
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    constraints = deduplicate_constraints(
        constraints
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_constraints(
        query=query,
        constraints=constraints,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_constraints(
        query=query,
        constraints=constraints,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()