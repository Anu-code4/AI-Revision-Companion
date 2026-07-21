

"""
FastAPI entry point for AI-Powered Document Retrieval System
"""

import logging
import shutil
from pathlib import Path
from typing import Annotated, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from utils.logger import setup_logger

from preprocessing.chunking import run_chunking
from preprocessing.embeddings import run_embeddings
from preprocessing.vector_store import build_vector_store

from ingestion.file_tracker import update_metadata

from app import get_answer, stream_response


# ==========================================================
# Logger
# ==========================================================

setup_logger()
logger = logging.getLogger(__name__)


# ==========================================================
# FastAPI App
# ==========================================================

app = FastAPI(
    title="AI-Powered Document Retrieval System",
    version="1.0.0",
    description="Production RAG API",
)

DOCUMENT_FOLDER = Path("Document")


# ==========================================================
# Request Model
# ==========================================================

class ChatRequest(BaseModel):
    question: str


# ==========================================================
# Response Models
# ==========================================================

class Source(BaseModel):
    document: str
    chunks: List[int]


class ChatResponse(BaseModel):
    answer: str
    confidence: float | None = None
    sources: List[Source] = []


# ==========================================================
# Helper
# ==========================================================

def rebuild_index():

    logger.info("Rebuilding search index...")

    run_chunking()
    run_embeddings()
    build_vector_store()
    update_metadata()

    logger.info("Index rebuilt successfully.")


# ==========================================================
# Home Route
# ==========================================================

@app.get("/")
def home():

    logger.info("Home endpoint accessed.")

    return {
        "message": "AI-Powered Document Retrieval System API is running."
    }


# ==========================================================
# Chat Route
# ==========================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    logger.info("=" * 70)
    logger.info("Received chat request")
    logger.info("Question: %s", request.question)

    result = get_answer(request.question)

    logger.info("Chat request completed successfully.")

    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["sources"],
    )


# ==========================================================
# Upload Documents
# ==========================================================

@app.post("/upload")
async def upload_documents(
    files: Annotated[list[UploadFile], File(...)],
):

    DOCUMENT_FOLDER.mkdir(exist_ok=True)

    uploaded_files = []

    logger.info("Uploading %d document(s)...", len(files))

    for file in files:

        destination = DOCUMENT_FOLDER / file.filename

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Uploaded: %s", file.filename)

        uploaded_files.append(file.filename)

    try:

        logger.info("Starting document indexing...")

        rebuild_index()

        logger.info("Document indexing completed successfully.")

    except Exception as e:

        logger.exception("Document indexing failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {str(e)}",
        )

    return {
        "message": f"{len(uploaded_files)} file(s) uploaded and indexed successfully.",
        "uploaded_files": uploaded_files,
    }


# ==========================================================
# List Documents
# ==========================================================

@app.get("/documents")
def list_documents():

    DOCUMENT_FOLDER.mkdir(exist_ok=True)

    documents = []

    for file in sorted(DOCUMENT_FOLDER.iterdir()):

        if file.is_file():

            documents.append(
                {
                    "filename": file.name,
                    "type": file.suffix.replace(".", "").upper(),
                    "size_kb": round(file.stat().st_size / 1024, 2),
                }
            )

    logger.info("Returned %d indexed document(s).", len(documents))

    return documents


# ==========================================================
# Delete Document
# ==========================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):

    file_path = DOCUMENT_FOLDER / filename

    if not file_path.exists():

        logger.warning("Delete failed. File not found: %s", filename)

        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    try:

        file_path.unlink()

        logger.info("Deleted document: %s", filename)

        rebuild_index()

        return {
            "message": f"{filename} deleted successfully."
        }

    except Exception as e:

        logger.exception("Failed to delete document.")

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ==========================================================
# Streaming Route
# ==========================================================

@app.post("/stream")
def stream(request: ChatRequest):

    logger.info("=" * 70)
    logger.info("Streaming request received")
    logger.info("Question: %s", request.question)

    return StreamingResponse(
        stream_response(request.question),
        media_type="text/plain",
    )