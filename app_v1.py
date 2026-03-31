import streamlit as st
from pathlib import Path
from rag_pipeline import (
    create_index_if_not_exists,
    ingest_pdf,
    get_rag_chain,
    check_existing_namespace  # <-- add this in rag_pipeline
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Modern RAG Chatbot", layout="wide")
st.title("📄🔍 Modern RAG PDF Chatbot")

# ===============================
# Session State
# ===============================
if "namespace" not in st.session_state:
    st.session_state.namespace = None

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ===============================
# AUTO LOAD EXISTING DATA
# ===============================
if not st.session_state.rag_chain:
    existing_namespace = check_existing_namespace()
    if existing_namespace:
        st.session_state.namespace = existing_namespace
        st.session_state.rag_chain = get_rag_chain(existing_namespace)


# ===============================
# Upload Section (Optional)
# ===============================
st.subheader("📂 Upload PDF (Optional)")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    file_path = UPLOAD_DIR / uploaded_file.name

    if not file_path.exists():
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully.")
    else:
        st.info("File already exists locally.")

    if st.button("Process & Index PDF"):

        with st.spinner("Creating index if needed..."):
            create_index_if_not_exists()

        with st.spinner("Checking existing namespace..."):
            existing_namespace = check_existing_namespace(
                uploaded_file.name
            )

        if existing_namespace:
            st.session_state.namespace = existing_namespace
            st.session_state.rag_chain = get_rag_chain(existing_namespace)
            st.success("⚡ PDF already indexed. Loaded existing data.")
        else:
            with st.spinner("Ingesting PDF..."):
                namespace = ingest_pdf(str(file_path))

            st.session_state.namespace = namespace
            st.session_state.rag_chain = get_rag_chain(namespace)

            st.success("✅ PDF indexed successfully.")


# ===============================
# Chat Section (Always Available)
# ===============================
st.divider()
st.subheader("💬 Chat with Knowledge Base")

if st.session_state.rag_chain:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")

    if user_input:

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(user_input)
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

else:
    st.info("ℹ️ No indexed PDFs found. Upload and process a PDF or use existing indexed data.")
