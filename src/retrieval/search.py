import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_FILE = PROJECT_ROOT / "data" / "index" / "chunks.faiss"
METADATA_FILE = PROJECT_ROOT / "data" / "index" / "chunks_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 5


# ============================================================
# LOAD INDEX + METADATA
# ============================================================

def load_index():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found:\n{INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n{METADATA_FILE}"
        )

    print("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_FILE))

    print("Loading metadata...")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: "
            f"{index.ntotal} vectors vs {len(metadata)} metadata records."
        )

    return index, metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    return model


# ============================================================
# SEARCH
# ============================================================

def search(query, model, index, metadata, top_k=DEFAULT_TOP_K):
    query = query.strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    # Must match the index-building configuration:
    # normalized embeddings + inner product.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for rank, (score, index_position) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        if index_position < 0:
            continue

        result = metadata[index_position].copy()

        result["rank"] = rank
        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(query, results):
    print()
    print("=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"Query: {query}")
    print()

    for result in results:
        print("-" * 70)

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"Score      : {result['score']:.4f}"
        )

        print(
            f"Chunk ID    : {result.get('chunk_id')}"
        )

        print(
            f"Document ID : {result.get('document_id')}"
        )

        print(
            f"Institution : {result.get('institution')}"
        )

        print(
            f"Year        : {result.get('year')}"
        )

        print(
            f"Document    : {result.get('filename')}"
        )

        print(
            f"Domain      : {result.get('domain')}"
        )

        print()
        print("Text:")
        print(result.get("text", "").strip())

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
            'python -m src.retrieval.search "your search question"'
        )
        print()
        print("Example:")
        print(
            'python -m src.retrieval.search "What are the admission requirements?"'
        )
        return

    query = " ".join(sys.argv[1:])

    index, metadata = load_index()
    model = load_model()

    results = search(
        query=query,
        model=model,
        index=index,
        metadata=metadata,
        top_k=DEFAULT_TOP_K,
    )

    display_results(query, results)


if __name__ == "__main__":
    main()