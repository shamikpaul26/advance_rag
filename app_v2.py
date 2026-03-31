import streamlit as st
from pathlib import Path

from rag_hybrid import (
    create_index_if_not_exists,
    ingest_pdf,
    get_rag_chain,
    check_existing_namespace,
    BM25_PARAMS_PATH,          # used to guard auto-load
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("📄🔍 Hybrid RAG PDF Chatbot (BM25 + Vector)")

# ===============================
# Session State
# ===============================
for key, default in [
    ("namespace", None),
    ("rag_chain", None),
    ("messages", []),
    ("alpha", 0.5),            # hybrid blend ratio
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===============================
# AUTO LOAD EXISTING DATA
# ── Only if BM25 params exist,
#    because HybridRetriever needs
#    a fitted encoder at query time.
# ===============================
if not st.session_state.rag_chain:
    if Path(BM25_PARAMS_PATH).exists():
        existing_namespace = check_existing_namespace()
        if existing_namespace:
            try:
                st.session_state.namespace = existing_namespace
                st.session_state.rag_chain = get_rag_chain(
                    existing_namespace,
                    alpha=st.session_state.alpha,
                )
            except Exception as e:
                st.warning(f"⚠️ Auto-load failed: {e}")
    # If BM25 params are missing, silently skip — user must ingest first.


# ===============================
# Sidebar — Hybrid Search Controls
# ===============================
with st.sidebar:
    st.header("⚙️ Hybrid Search Settings")

    alpha = st.slider(
        label="Alpha (0 = Keyword-only BM25 · 1 = Vector-only)",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.alpha,
        step=0.05,
        help=(
            "Controls the blend between BM25 keyword search and dense vector search.\n\n"
            "• 0.0 → pure BM25 keyword\n"
            "• 0.5 → balanced hybrid (recommended)\n"
            "• 1.0 → pure semantic vector"
        ),
    )

    # Re-build chain when alpha changes and a namespace is already loaded
    if alpha != st.session_state.alpha:
        st.session_state.alpha = alpha
        if st.session_state.namespace:
            try:
                st.session_state.rag_chain = get_rag_chain(
                    st.session_state.namespace,
                    alpha=alpha,
                )
                st.success("Chain updated with new alpha.")
            except Exception as e:
                st.error(f"Failed to update chain: {e}")

    st.divider()

    # Show current search mode as a friendly label
    if alpha == 0.0:
        mode_label = "🔤 Pure Keyword (BM25)"
    elif alpha == 1.0:
        mode_label = "🧠 Pure Semantic (Vector)"
    elif alpha < 0.4:
        mode_label = "🔤➕ Keyword-heavy Hybrid"
    elif alpha > 0.6:
        mode_label = "🧠➕ Semantic-heavy Hybrid"
    else:
        mode_label = "⚖️ Balanced Hybrid"

    st.markdown(f"**Current mode:** {mode_label}")

    st.divider()

    # Clear chat history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ===============================
# Upload Section
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

    if st.button("⚙️ Process & Index PDF"):

        with st.spinner("Creating Pinecone index if needed..."):
            create_index_if_not_exists()

        # ── FIX: namespace is MD5(file_path), NOT the filename.
        #         Check by generating the namespace from the saved path
        #         and querying Pinecone directly.
        # ──────────────────────────────────────────────────────────────
        from rag_hybrid import generate_namespace
        expected_namespace = generate_namespace(str(file_path))

        with st.spinner("Checking if PDF is already indexed..."):
            existing_namespace = check_existing_namespace()   # returns first namespace

        # Match against the expected MD5 namespace for this specific file
        already_indexed = (existing_namespace == expected_namespace)

        if already_indexed:
            st.session_state.namespace = expected_namespace
            try:
                st.session_state.rag_chain = get_rag_chain(
                    expected_namespace,
                    alpha=st.session_state.alpha,
                )
                st.success("⚡ PDF already indexed. Loaded existing data.")
            except FileNotFoundError:
                # BM25 params were deleted — re-ingest to refit encoder
                st.warning(
                    "BM25 params not found locally. Re-ingesting to refit encoder..."
                )
                already_indexed = False   # fall through to ingest block

        if not already_indexed:
            with st.spinner("Ingesting PDF and fitting BM25 encoder..."):
                try:
                    namespace = ingest_pdf(str(file_path))
                    st.session_state.namespace = namespace
                    st.session_state.rag_chain = get_rag_chain(
                        namespace,
                        alpha=st.session_state.alpha,
                    )
                    st.success("✅ PDF indexed successfully with Hybrid Search.")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")


# ===============================
# Chat Section
# ===============================
st.divider()
st.subheader("💬 Chat with Your Knowledge Base")

if st.session_state.rag_chain:

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.rag_chain.invoke(user_input)
                except Exception as e:
                    answer = f"⚠️ Error generating answer: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info(
        "ℹ️ No indexed PDF loaded. "
        "Upload and process a PDF above, or ensure a previously indexed PDF "
        "and its BM25 params (`bm25_params.json`) are present."
    )