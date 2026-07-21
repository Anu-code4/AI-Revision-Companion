

"""
LLM response generation.

This module builds prompts using retrieved context,
sends them to Ollama, and returns the generated answer.
"""

import logging

from ollama import chat

from config import MODEL_NAME, TEMPERATURE
from utils.logger import setup_logger
from .prompts import SYSTEM_PROMPT
from query_router import QueryType


# ==========================================================
# Logger
# ==========================================================

setup_logger()
logger = logging.getLogger(__name__)


# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(
    question: str,
    retrieved_chunks: list,
    conversation_history,
    query_type: QueryType,
) -> str:
    """
    Build the final prompt sent to the LLM.
    """

    context = ""

    if query_type == QueryType.DOCUMENT:

        context = "\n\n".join(
            chunk["text"] if isinstance(chunk, dict) else chunk
            for chunk in retrieved_chunks
        )

    history = ""

    if conversation_history:

        history = "\n".join(
            f"{message['role']}: {message['content']}"
            for message in conversation_history
        )

    prompt = f"""
{SYSTEM_PROMPT}

==================================================
Conversation History
==================================================

{history}

==================================================
Retrieved Context
==================================================

{context}

==================================================
Query Type
==================================================

{query_type.value}

==================================================
User Question
==================================================

{question}

==================================================
Answer
==================================================

Instructions:

If Query Type is DOCUMENT:
- Answer ONLY using Retrieved Context.
- If the answer is unavailable, reply exactly:
  "I don't know based on the provided documents."

If Query Type is MEMORY:
- Answer ONLY using Conversation History.

If Query Type is CHAT:
- Respond naturally as a helpful AI assistant.
""".strip()

    return prompt


# ==========================================================
# Generate Answer (Non-Streaming)
# ==========================================================

def generate_answer(
    question: str,
    retrieved_chunks: list,
    conversation_history=None,
    query_type: QueryType = QueryType.DOCUMENT,
):

    logger.info("Entered generate_answer().")

    # ------------------------------------------------------
    # Handle different query types
    # ------------------------------------------------------

    if query_type == QueryType.DOCUMENT:

        logger.info("Generating response for document query.")

        if not retrieved_chunks:
            logger.warning("No retrieved chunks received.")
            return {
                "answer": "I don't know based on the provided documents.",
                "confidence": None,
                "sources": [],
            }

    elif query_type == QueryType.CHAT:

        logger.info("Generating response for chat query.")

    elif query_type == QueryType.MEMORY:

        logger.info("Generating response for memory query.")

    # ------------------------------------------------------
    # Build Prompt
    # ------------------------------------------------------

    prompt = build_prompt(
        question,
        retrieved_chunks,
        conversation_history,
        query_type,
    )

    logger.info("Prompt constructed successfully.")

    # ------------------------------------------------------
    # Call Ollama
    # ------------------------------------------------------

    try:

        logger.info("Sending prompt to Ollama.")

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": TEMPERATURE,
            },
            stream=False,
        )

        logger.info("Response received from Ollama.")

    except Exception:

        logger.exception("Failed to generate answer.")

        return {
            "answer": "Sorry, something went wrong while generating the answer.",
            "confidence": None,
            "sources": [],
        }

    answer = response["message"]["content"]

    logger.info("Answer generated successfully.")

# ------------------------------------------------------
# Confidence Score
# ------------------------------------------------------

    confidence = None

    if query_type == QueryType.DOCUMENT and retrieved_chunks:

        confidence = float(retrieved_chunks[0].get("confidence", 0.0))

    # Clamp value between 0 and 1
    confidence = max(0.0, min(confidence, 1.0))
    # ------------------------------------------------------
    # Sources
    # ------------------------------------------------------

    sources = []

    if query_type == QueryType.DOCUMENT and retrieved_chunks:

        citations = {}

        for chunk in retrieved_chunks:

            if not isinstance(chunk, dict):
                continue

            source = chunk.get("source", "Unknown Source")
            chunk_id = chunk.get("id")

            citations.setdefault(source, []).append(chunk_id)

        for source, ids in citations.items():

            sources.append(
                {
                    "document": source,
                    "chunks": sorted(set(ids)),
                }
            )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": sources,
    }


# ==========================================================
# Stream Answer
# ==========================================================

def stream_answer(
    question: str,
    retrieved_chunks: list,
    conversation_history=None,
    query_type: QueryType = QueryType.DOCUMENT,
):
    """
    Stream response tokens from Ollama.
    """

    logger.info("Entered stream_answer().")

    if query_type == QueryType.DOCUMENT and not retrieved_chunks:

        yield "I don't know based on the provided documents."
        return

    prompt = build_prompt(
        question,
        retrieved_chunks,
        conversation_history,
        query_type,
    )

    try:

        logger.info("Streaming response from Ollama.")

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": TEMPERATURE,
            },
            stream=True,
        )

        for chunk in response:

            content = chunk["message"]["content"]

            if content:
                yield content

        logger.info("Streaming completed.")

    except Exception:

        logger.exception("Streaming failed.")

        yield "Sorry, something went wrong while generating the answer."