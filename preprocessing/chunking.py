

"""
Document Loader and Chunking Module

This module:
1. Reads all supported documents (.docx, .pdf) from the Document folder.
2. Extracts text.
3. Splits text into chunks.
4. Saves chunks into chunks.json.
"""

import json
import os

import fitz
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.file_tracker import detect_changes


# ==========================================================
# Constants
# ==========================================================

SUPPORTED_EXTENSIONS = {
    ".docx",
    ".pdf",
}


# ==========================================================
# Extract text from DOCX
# ==========================================================

def extract_text_from_docx(file_path):
    """
    Read all text from a Word document.
    """

    doc = Document(file_path)

    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


# ==========================================================
# Extract text from PDF
# ==========================================================

def extract_text_from_pdf(file_path):
    """
    Read all text from a PDF document.
    """

    pdf = fitz.open(file_path)

    text = ""

    for page in pdf:
        text += page.get_text() + "\n"

    pdf.close()

    return text


# ==========================================================
# Detect file type
# ==========================================================

def extract_text(file_path):
    """
    Detect the file type and extract text.
    """

    _, extension = os.path.splitext(file_path)

    extension = extension.lower()

    if extension == ".docx":
        return extract_text_from_docx(file_path)

    elif extension == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(f"Unsupported file type: {extension}")


# ==========================================================
# Chunking
# ==========================================================

def create_chunks(text):
    """
    Split text into overlapping chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    return splitter.split_text(text)


# ==========================================================
# Load Documents
# ==========================================================

def load_documents(folder_path, files=None):
    """
    Read DOCX and PDF files.

    Args:
        folder_path: Path to Document folder.
        files: Optional list of files.
               If None, all supported files are loaded.

    Returns:
        List of document dictionaries.
    """

    documents = []

    if files is None:
        files = [
            os.path.join(folder_path, filename)
            for filename in os.listdir(folder_path)
        ]

    for file_path in files:

        if not os.path.isfile(file_path):
            continue

        filename = os.path.basename(file_path)

        _, extension = os.path.splitext(filename)

        extension = extension.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            continue

        print(f"Reading: {filename}")

        text = extract_text(file_path)

        documents.append(
            {
                "source": filename,
                "file_type": extension,
                "text": text,
            }
        )

    return documents


# ==========================================================
# Run Chunking
# ==========================================================

def run_chunking():
    """
    Create chunks.json from all new or modified documents.
    """

    folder_path = os.path.join(os.getcwd(), "Document")

    new_files, modified_files, _ = detect_changes()

    files_to_process = new_files + modified_files

    if not files_to_process:
        print("✅ No new or modified documents found.")
        return

    documents = load_documents(
        folder_path,
        files_to_process,
    )

    chunk_data = []

    chunk_id = 0

    for document in documents:

        chunks = create_chunks(document["text"])

        for chunk in chunks:

            chunk_data.append(
                {
                    "id": chunk_id,
                    "source": document["source"],
                    "file_type": document["file_type"],
                    "text": chunk,
                }
            )

            chunk_id += 1

    with open(
        "chunks.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunk_data,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\n✅ {len(chunk_data)} chunks created "
        f"from {len(documents)} document(s)."
    )

    print("Chunks saved to chunks.json")


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":
    run_chunking()