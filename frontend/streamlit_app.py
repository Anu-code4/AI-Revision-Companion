

from pathlib import Path

import requests
import streamlit as st

# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="AI-Powered Document Retrieval System",
    page_icon="🤖",
    layout="wide",
)

# ==========================================================
# Load CSS
# ==========================================================

css_path = Path(__file__).parent / "styles.css"

if css_path.exists():
    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ==========================================================
# Session State
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("🤖 AI-Powered Document Retrieval System")
    st.caption("Production RAG Assistant")

    st.divider()

    files = st.file_uploader(
        "📂 Upload PDF / DOCX",
        type=["pdf", "docx"],
        accept_multiple_files=True,
    )

    if st.button("📤 Upload & Index"):

        if files:

            payload = [
                (
                    "files",
                    (
                        f.name,
                        f.getvalue(),
                        f.type,
                    ),
                )
                for f in files
            ]

            with st.spinner("📤 Uploading & Indexing Documents..."):

                response = requests.post(
                    "http://127.0.0.1:8000/upload",
                    files=payload,
                    timeout=300,
                )

            if response.ok:
                st.toast("Documents indexed successfully! 🎉")
                st.success("Documents indexed successfully.")
                st.balloons()
            else:
                st.error(response.text)

        else:
            st.warning("Please select at least one document.")

    st.divider()

    # ==========================================================
# Indexed Documents
# ==========================================================

st.divider()

st.subheader("📂 Indexed Documents")

try:

    response = requests.get(
        "http://127.0.0.1:8000/documents",
        timeout=30,
    )

    response.raise_for_status()

    documents = response.json()

    if documents:

        for doc in documents:

            c1, c2 = st.columns([5, 1])

            with c1:

                icon = "📄"

                if doc["type"] == "DOCX":
                    icon = "📝"

                st.write(
                    f"{icon} **{doc['filename']}**"
                )

                st.caption(
                    f"{doc['size_kb']} KB"
                )

            with c2:

                if st.button(
                    "🗑",
                    key=f"delete_{doc['filename']}",
                    help="Delete document",
                ):

                    delete = requests.delete(
                        f"http://127.0.0.1:8000/documents/{doc['filename']}",
                        timeout=300,
                    )

                    if delete.ok:

                        st.toast(
                            "Document deleted."
                        )

                        st.rerun()

                    else:

                        st.error(
                            delete.text
                        )

    else:

        st.caption("No indexed documents.")

except Exception:

    st.caption("Unable to load documents.")

st.divider()

if st.button("🗑 Clear Chat"):

    st.session_state.messages = []

    st.rerun()

    st.divider()

    st.subheader("📊 Statistics")

    st.metric(
        "Retriever",
        "Hybrid",
    )

    st.metric(
        "Model",
        "Llama 3.2",
    )

    st.metric(
        "Conversation",
        len(st.session_state.messages),
    )

# ==========================================================
# Header
# ==========================================================

col1, col2 = st.columns([8, 2])

with col1:
    st.title("🤖 AI-Powered Document Retrieval System")
    st.caption("Intelligent Document Assistant")

with col2:
    st.success("🟢 Ready")

st.divider()

# ==========================================================
# Welcome Screen
# ==========================================================

if not st.session_state.messages:

    st.info(
        """
### 👋 Welcome

1. Upload one or more PDF/DOCX documents.

2. Click **Upload & Index**.

3. Ask questions about your documents.

The assistant answers using **Hybrid RAG** and displays the source
documents used to generate the response.
"""
    )

# ==========================================================
# Display Chat History
# ==========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            confidence = message.get("confidence")
            sources = message.get("sources", [])

            if confidence is not None:

                if confidence >= 0.80:
                    st.success(f"🟢 High Confidence ({confidence:.2f})")

                elif confidence >= 0.50:
                    st.warning(f"🟡 Medium Confidence ({confidence:.2f})")

                else:
                    st.error(f"🔴 Low Confidence ({confidence:.2f})")

            if sources:

                st.markdown("##### 📚 Sources")

                for source in sources:
                    st.markdown(f"- `{source}`")

# ==========================================================
# Chat Input
# ==========================================================

prompt = st.chat_input(
    "Ask anything about your documents..."
)

if prompt:

    # User message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response

    with st.chat_message("assistant"):

        with st.spinner("🧠 Generating answer..."):

            try:

                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={"question": prompt},
                    timeout=120,
                )

                response.raise_for_status()

                data = response.json()

                answer = data.get(
                    "answer",
                    "No answer returned.",
                )

                confidence = data.get("confidence")

                sources = data.get("sources", [])

            except Exception as e:

                answer = f"❌ {e}"
                confidence = None
                sources = []

        st.markdown(answer)

        if confidence is not None:

            if confidence >= 0.80:
                st.success(f"🟢 High Confidence ({confidence:.2f})")

            elif confidence >= 0.50:
                st.warning(f"🟡 Medium Confidence ({confidence:.2f})")

            else:
                st.error(f"🔴 Low Confidence ({confidence:.2f})")

        if sources:

            st.markdown("##### 📚 Sources")

            for source in sources:
                st.markdown(f"- `{source}`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "sources": sources,
        }
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.caption(
    "🚀 AI-Powered Document Retrieval System | FastAPI • Streamlit • Hybrid RAG • CrossEncoder • BM25 • FAISS"
)