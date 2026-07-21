

import json
import numpy as np
import faiss


# ==========================================================
# Build Vector Store
# ==========================================================

def build_vector_store():
    """
    Build FAISS vector store from embedded_chunks.json.
    """

    # ======================================================
    # Load embedded chunks
    # ======================================================

    with open("embedded_chunks.json", "r", encoding="utf-8") as file:
        embedded_chunks = json.load(file)

    print(f"Loaded {len(embedded_chunks)} embedded chunks.")

    # ======================================================
    # Extract embeddings
    # ======================================================

    embeddings = []

    for chunk in embedded_chunks:
        embeddings.append(chunk["embedding"])

    # ======================================================
    # Convert to NumPy array
    # ======================================================

    embeddings = np.array(embeddings).astype("float32")

    print(f"Embedding Matrix Shape: {embeddings.shape}")

    # ======================================================
    # Get embedding dimension
    # ======================================================

    dimension = embeddings.shape[1]

    print(f"Embedding Dimension: {dimension}")

    # ======================================================
    # Create FAISS Index
    # ======================================================

    index = faiss.IndexFlatL2(dimension)

    print("FAISS Index Created Successfully.")

    # ======================================================
    # Add embeddings
    # ======================================================

    index.add(embeddings)

    print(f"Vectors Stored in Index: {index.ntotal}")

    # ======================================================
    # Save FAISS Index
    # ======================================================

    faiss.write_index(index, "faiss_index.bin")

    print("FAISS index saved as faiss_index.bin")

    print("\n✅ Vector Database Created Successfully!")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    build_vector_store()