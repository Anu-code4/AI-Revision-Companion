

"""
Generate embeddings for document chunks.

This module:
1. Loads chunks from chunks.json.
2. Creates embeddings using SentenceTransformer.
3. Preserves all metadata.
4. Saves embedded_chunks.json.
"""

import json
from sentence_transformers import SentenceTransformer


# ==========================================================
# Generate Embeddings
# ==========================================================

def run_embeddings():
    """
    Generate embeddings for document chunks and save them.
    """

    # ======================================================
    # Load Embedding Model
    # ======================================================

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ======================================================
    # Load Chunks
    # ======================================================

    print("Loading chunks...")

    with open("chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # ======================================================
    # Generate Embeddings
    # ======================================================

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(texts)

    # ======================================================
    # Combine Metadata + Embeddings
    # ======================================================

    embedded_chunks = []

    for chunk, embedding in zip(chunks, embeddings):

        embedded_chunks.append(
            {
                "id": chunk["id"],
                "source": chunk["source"],
                "file_type": chunk["file_type"],
                "text": chunk["text"],
                "embedding": embedding.tolist()
            }
        )

    # ======================================================
    # Save
    # ======================================================

    with open("embedded_chunks.json", "w", encoding="utf-8") as file:

        json.dump(
            embedded_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\n✅ Created embeddings for {len(embedded_chunks)} chunks.")
    print("Embeddings saved to embedded_chunks.json")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    run_embeddings()
