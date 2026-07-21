# AI-Powered Document Retrieval System

> A production-ready Retrieval-Augmented Generation (RAG) system for intelligent document retrieval, semantic search, reranking, and question answering using Hybrid Search, Cross-Encoder Reranking, FastAPI, Streamlit, and Ollama.

---

## Overview

AI-Powered Document Retrieval System is a production-oriented Generative AI application that enables users to upload documents and ask natural language questions. The system retrieves the most relevant document chunks using Hybrid Retrieval (FAISS + BM25), improves ranking with a Cross-Encoder Reranker, and generates context-aware answers using a Large Language Model (LLM).

The project is designed with a modular architecture and includes document ingestion, semantic search, query rewriting, conversation memory, REST APIs, a Streamlit interface, Docker support, logging, and automated testing.

---

## Features

- Hybrid Retrieval (FAISS + BM25)
- Cross-Encoder Reranking
- Retrieval-Augmented Generation (RAG)
- Semantic Search using Sentence Transformers
- Query Rewriting for follow-up questions
- Conversation Memory
- FastAPI REST API
- Streamlit Web Interface
- PDF and DOCX document support
- Automatic document indexing
- Source citation
- Confidence scoring
- Docker support
- GitHub Actions CI
- Logging and testing


---

## System Architecture

```text
                   User
                     │
                     ▼
          Streamlit Web Interface
                     │
                     ▼
               FastAPI Backend
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
  Query Router            Conversation Memory
        │
        ▼
  Query Rewriter
        │
        ▼
 Hybrid Retriever (FAISS + BM25)
        │
        ▼
 Cross-Encoder Reranker
        │
        ▼
     Context Builder
        │
        ▼
   Ollama (LLM)
        │
        ▼
 Generated Answer + Sources
```




---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python 3.11 |
| Backend | FastAPI |
| Frontend | Streamlit |
| LLM | Ollama (Llama 3.2) |
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Keyword Search | BM25 |
| Reranker | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| AI Frameworks | Sentence Transformers, LangChain |
| Document Processing | PyMuPDF, python-docx |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Version Control | Git & GitHub |




---

## Project Structure

```text
AI-Powered-Document-Retrieval-System/
│
├── api.py                      # FastAPI application
├── app.py                      # CLI entry point
├── config.py                   # Project configuration
├── query_router.py             # Query routing logic
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── Document/                   # Sample documents
│   ├── *.pdf
│   └── *.docx
│
├── preprocessing/
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── retrievers/
│   ├── retriever.py
│   ├── bm25_retriever.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   └── metadata_filter.py
│
├── llm/
│   ├── generator.py
│   ├── prompts.py
│   ├── query_rewriter.py
│   └── multi_query.py
│
├── memory/
│   └── memory.py
│
├── frontend/
│   ├── streamlit_app.py
│   └── styles.css
│
├── evaluation/
│   ├── evaluation.py
│   ├── evaluation_dataset.py
│   ├── production_evaluator.py
│   └── results/
│
├── tests/
│   ├── test_api.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_retriever.py
│   ├── test_bm25_retriever.py
│   ├── test_hybrid_retriever.py
│   ├── test_reranker.py
│   ├── test_generator.py
│   ├── test_memory.py
│   ├── test_query_router.py
│   ├── test_query_rewriter.py
│   └── test_metadata_filter.py
│
├── utils/
│   └── logger.py
│
├── data/
├── metadata/
├── logs/
├── .github/
│   └── workflows/
│
└── README.md
```







---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ai-powered-document-retrieval-system.git
cd ai-powered-document-retrieval-system
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Start Ollama

Install Ollama from:

https://ollama.com

Pull the required model:

```bash
ollama pull llama3.2
```

Start Ollama:

```bash
ollama serve
```

### 5. Build the Vector Database

Process the documents and generate embeddings:

```bash
python preprocessing/chunking.py
python preprocessing/embeddings.py
python preprocessing/vector_store.py
```

### 6. Run the FastAPI Server

```bash
uvicorn api:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

### 7. Launch the Streamlit Interface

```bash
streamlit run frontend/streamlit_app.py
```

The application will open automatically in your browser.
