import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "index"

INDEX_FILE = OUTPUT_DIR / "chunks.faiss"
METADATA_FILE = OUTPUT_DIR / "chunks_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():
    print("Loading chunks...")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict):
        chunks = data.get("chunks", data.get("data", []))
    else:
        raise ValueError("Unsupported chunks.json format.")

    if not chunks:
        raise ValueError("No chunks found.")

    print(f"Loaded {len(chunks)} chunks.")

    return chunks


# ============================================================
# BUILD EMBEDDINGS
# ============================================================

def build_embeddings(chunks):
    print()
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()

        if not text:
            raise ValueError(
                f"Chunk {chunk.get('chunk_id', 'UNKNOWN')} has empty text."
            )

        texts.append(text)

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    print()
    print(f"Embedding dimension: {dimension}")

    # Because embeddings are normalized, inner product
    # is equivalent to cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS index contains {index.ntotal} vectors.")

    return index


# ============================================================
# SAVE INDEX + METADATA
# ============================================================

def save_outputs(index, chunks):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    metadata = []

    for chunk in chunks:
        metadata.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("document_id"),
                "institution": chunk.get("institution"),
                "year": chunk.get("year"),
                "filename": chunk.get("filename"),
                "relative_path": chunk.get("relative_path"),
                "domain": chunk.get("domain"),
                "document_type": chunk.get("document_type"),
                "text": chunk.get("text"),
                "word_count": chunk.get("word_count"),
                "character_count": chunk.get("character_count"),
            }
        )

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print()
    print("Output files:")
    print(INDEX_FILE)
    print(METADATA_FILE)


# ============================================================
# VERIFY
# ============================================================

def verify(index, chunks):
    print()
    print("=" * 60)
    print("INDEX VERIFICATION")
    print("=" * 60)

    print(f"Chunks              : {len(chunks)}")
    print(f"FAISS vectors       : {index.ntotal}")
    print(f"Embedding dimension : {index.d}")

    if index.ntotal != len(chunks):
        raise ValueError("Mismatch between chunks and FAISS vectors.")

    if len(chunks) != len(set(c["chunk_id"] for c in chunks)):
        raise ValueError("Duplicate chunk IDs detected.")

    print("Vector count check  : PASS")
    print("Chunk ID check      : PASS")
    print("INDEX BUILD         : PASS")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CONSTRAINT PRESERVER - BUILD VECTOR INDEX")
    print("=" * 60)

    chunks = load_chunks()

    embeddings = build_embeddings(chunks)

    index = build_faiss_index(embeddings)

    save_outputs(index, chunks)

    verify(index, chunks)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()