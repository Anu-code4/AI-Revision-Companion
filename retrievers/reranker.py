

"""
Cross-Encoder Reranker.
"""

import logging

from sentence_transformers import CrossEncoder

from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

logger.info("Loading CrossEncoder model...")

model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

def rerank(
    query: str,
    retrieved_chunks: list,
    top_k: int = 5,
) -> list:
    """
    Rerank retrieved chunks using a CrossEncoder.
    """

    if not retrieved_chunks:
        return []

    # ------------------------------------------------------
    # Create Query-Chunk Pairs
    # ------------------------------------------------------

    pairs = [
        (query, chunk["text"])
        for chunk in retrieved_chunks
    ]

    # ------------------------------------------------------
    # Predict Relevance Scores
    # ------------------------------------------------------

    scores = model.predict(pairs)

    # ------------------------------------------------------
    # Attach Scores
    # ------------------------------------------------------

    for chunk, score in zip(retrieved_chunks, scores):
        chunk["rerank_score"] = float(score)

    # ------------------------------------------------------
    # Sort by CrossEncoder Score
    # ------------------------------------------------------

    retrieved_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    logger.info(
        f"Reranked {len(retrieved_chunks)} chunks."
    )

    # ------------------------------------------------------
    # Keep Top-K
    # ------------------------------------------------------

    results = retrieved_chunks[:top_k]

    # ------------------------------------------------------
    # Confidence Score
    # ------------------------------------------------------

    confidence = results[0]["rerank_score"]

    logger.info(
        f"Retrieval confidence: {confidence:.4f}"
    )

    # Attach confidence to every returned chunk
    for chunk in results:
        chunk["confidence"] = confidence

    return results