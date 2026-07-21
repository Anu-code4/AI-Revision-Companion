

"""
Main application entry point.
"""

import logging

from utils.logger import setup_logger

from retrievers.hybrid_retriever import hybrid_retriever
from llm.generator import (
    generate_answer,
    stream_answer,
)
from llm.query_rewriter import rewrite_query

from config import TOP_K
from memory.memory import ConversationMemory
from query_router import route_query, QueryType

# ==========================================================
# Logger
# ==========================================================

setup_logger()
logger = logging.getLogger(__name__)

# ==========================================================
# Conversation Memory
# ==========================================================

memory = ConversationMemory(max_history=5)


# ==========================================================
# Normal Response
# ==========================================================

def get_answer(query: str) -> dict:

    logger.info(f"Received query: {query}")

    # ======================================================
    # Step 1: Route Query
    # ======================================================

    query_type = route_query(query)

    logger.info(f"Query Type: {query_type}")

    rewritten_query = query
    retrieved_chunks = []

    # ======================================================
    # Step 2: Document Query
    # ======================================================

    if query_type == QueryType.DOCUMENT:

        rewritten_query = rewrite_query(
            question=query,
            conversation_history=memory.get_history(),
        )

        logger.info(f"Rewritten Query: {rewritten_query}")

        print("\n" + "=" * 60)
        print(f"Original Query : {query}")
        print(f"Rewritten Query: {rewritten_query}")
        print("=" * 60)

        retrieved_chunks = hybrid_retriever(
            query=rewritten_query,
            top_k=TOP_K,
        )

        print(f"\nRetrieved {len(retrieved_chunks)} chunks\n")

        for chunk in retrieved_chunks:

            print("-" * 60)
            print(f"Document : {chunk['source']}")
            print(f"Retriever: {chunk['retriever']}")
            print(f"RRF Score: {chunk['rrf_score']:.6f}")
            print(chunk["text"][:200])

    # ======================================================
    # Step 3: Memory Query
    # ======================================================

    elif query_type == QueryType.MEMORY:

        logger.info("Memory query detected. Skipping retrieval.")

    # ======================================================
    # Step 4: Chat Query
    # ======================================================

    elif query_type == QueryType.CHAT:

        logger.info("Chat query detected. Skipping retrieval.")

    else:

        logger.warning("Unknown query type.")

    # ======================================================
    # Step 5: Generate Answer
    # ======================================================

    result = generate_answer(
        question=rewritten_query,
        retrieved_chunks=retrieved_chunks,
        conversation_history=memory.get_history(),
        query_type=query_type,
    )

    # ======================================================
    # Step 6: Update Memory
    # ======================================================

    memory.add_user_message(query)
    memory.add_ai_message(result["answer"])

    logger.info("Answer generated successfully.")

    return result


# ==========================================================
# Streaming Response
# ==========================================================

def stream_response(query: str):
    """
    Stream response from the LLM.
    """

    logger.info(f"Received streaming query: {query}")

    query_type = route_query(query)

    logger.info(f"Query Type: {query_type}")

    rewritten_query = query
    retrieved_chunks = []

    # ======================================================
    # Document Query
    # ======================================================

    if query_type == QueryType.DOCUMENT:

        rewritten_query = rewrite_query(
            question=query,
            conversation_history=memory.get_history(),
        )

        logger.info(f"Rewritten Query: {rewritten_query}")

        retrieved_chunks = hybrid_retriever(
            query=rewritten_query,
            top_k=TOP_K,
        )

    # ======================================================
    # Memory Query
    # ======================================================

    elif query_type == QueryType.MEMORY:

        logger.info("Memory query detected. Skipping retrieval.")

    # ======================================================
    # Chat Query
    # ======================================================

    elif query_type == QueryType.CHAT:

        logger.info("Chat query detected. Skipping retrieval.")

    else:

        logger.warning("Unknown query type.")

    # ======================================================
    # Stream Answer
    # ======================================================

    full_answer = ""

    for token in stream_answer(
        question=rewritten_query,
        retrieved_chunks=retrieved_chunks,
        conversation_history=memory.get_history(),
        query_type=query_type,
    ):
        full_answer += token
        yield token

    # ======================================================
    # Update Memory
    # ======================================================

    memory.add_user_message(query)
    memory.add_ai_message(full_answer)

    logger.info("Streaming completed.")


# ==========================================================
# CLI
# ==========================================================

def main():

    print("\n🚀 AI-Powered Document Retrieval System Ready\n")

    while True:

        query = input("Ask: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("\nGoodbye!\n")
            break

        if not query:
            print("Please enter a question.")
            continue

        result = get_answer(query)

        print("\nAnswer:\n")
        print(result["answer"])

        if result.get("confidence"):
            print(f"\nConfidence: {result['confidence']}")

        if result.get("sources"):

            print("\nSources:")

            for source in result["sources"]:

                chunks = ", ".join(
                    map(str, source["chunks"])
                )

                print(
                    f"- {source['document']} "
                    f"(Chunks: {chunks})"
                )

        print("-" * 80)


if __name__ == "__main__":
    main()