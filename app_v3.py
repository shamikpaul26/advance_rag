import streamlit as st
from pathlib import Path

from rag_hybrid import (
    create_index_if_not_exists,
    ingest_pdf,
    get_rag_chain,
    check_existing_namespace,
    generate_namespace,
    BM25_PARAMS_PATH,
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("Advanced RAG Chatbot  —  BM25 + Vector + Re-ranking")

# ===============================
# Session State
# ===============================
for key, default in [
    ("namespace",     None),
    ("rag_chain",     None),
    ("messages",      []),
    ("alpha",         0.5),
    ("initial_top_k", 10),
    ("final_top_n",   3),
    ("rerank_model",  "rerank-english-v3.0"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===============================
# Sidebar — Search & Re-rank Controls
# ===============================
with st.sidebar:
    st.header("⚙️ Search Settings")

    # ── Hybrid alpha ──────────────────────────────────────────────────
    alpha = st.slider(
        "Alpha  (0 = BM25-only · 1 = Vector-only)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.alpha, step=0.05,
        help="Blend ratio between keyword (BM25) and semantic (dense) search.",
    )

    if alpha == 0.0:
        st.caption("🔤 Pure Keyword (BM25)")
    elif alpha == 1.0:
        st.caption("🧠 Pure Semantic (Vector)")
    elif alpha < 0.4:
        st.caption("🔤➕ Keyword-heavy Hybrid")
    elif alpha > 0.6:
        st.caption("🧠➕ Semantic-heavy Hybrid")
    else:
        st.caption("⚖️ Balanced Hybrid")

    st.divider()

    # ── Re-ranking controls ───────────────────────────────────────────
    st.subheader(" Re-ranking (Cohere)")

    rerank_model = st.selectbox(
        "Re-rank Model",
        options=[
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            "rerank-english-v2.0",
        ],
        index=0,
        help=(
            "• english-v3.0       → fastest, English only\n"
            "• multilingual-v3.0  → supports 100+ languages\n"
            "• english-v2.0       → legacy fallback"
        ),
    )

    initial_top_k = st.slider(
        "Candidates fetched (initial_top_k)",
        min_value=4, max_value=20,
        value=st.session_state.initial_top_k, step=1,
        help="How many chunks Pinecone returns BEFORE re-ranking. More = better recall, slower.",
    )

    final_top_n = st.slider(
        "Chunks kept after re-ranking (final_top_n)",
        min_value=1, max_value=min(initial_top_k, 8),
        value=min(st.session_state.final_top_n, initial_top_k), step=1,
        help="Top-N chunks sent to the LLM after Cohere re-ranks them.",
    )

    st.info(
        f"Pipeline: Pinecone fetches **{initial_top_k}** candidates  →  "
        f"Cohere re-ranks  →  **{final_top_n}** chunks sent to GPT-4o-mini."
    )

    st.divider()

    # Rebuild chain if any setting changed and namespace is loaded
    settings_changed = (
        alpha         != st.session_state.alpha         or
        initial_top_k != st.session_state.initial_top_k or
        final_top_n   != st.session_state.final_top_n   or
        rerank_model  != st.session_state.rerank_model
    )

    if settings_changed and st.session_state.namespace:
        st.session_state.alpha         = alpha
        st.session_state.initial_top_k = initial_top_k
        st.session_state.final_top_n   = final_top_n
        st.session_state.rerank_model  = rerank_model
        try:
            st.session_state.rag_chain = get_rag_chain(
                st.session_state.namespace,
                alpha=alpha,
                initial_top_k=initial_top_k,
                final_top_n=final_top_n,
                rerank_model=rerank_model,
            )
            st.success("✅ Chain updated.")
        except Exception as e:
            st.error(f"Failed to rebuild chain: {e}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ===============================
# AUTO LOAD EXISTING DATA
# Guard: BM25 params must exist because HybridRetriever needs
# a fitted encoder at query time.
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
                    initial_top_k=st.session_state.initial_top_k,
                    final_top_n=st.session_state.final_top_n,
                    rerank_model=st.session_state.rerank_model,
                )
            except Exception as e:
                st.warning(f"⚠️ Auto-load failed: {e}")


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

        # Namespace is MD5(file_path), not the filename
        expected_namespace = generate_namespace(str(file_path))
        existing_namespace = check_existing_namespace()
        already_indexed    = (existing_namespace == expected_namespace)

        if already_indexed:
            # BM25 params must also exist locally; if not, re-ingest
            if Path(BM25_PARAMS_PATH).exists():
                st.session_state.namespace = expected_namespace
                try:
                    st.session_state.rag_chain = get_rag_chain(
                        expected_namespace,
                        alpha=st.session_state.alpha,
                        initial_top_k=st.session_state.initial_top_k,
                        final_top_n=st.session_state.final_top_n,
                        rerank_model=st.session_state.rerank_model,
                    )
                    st.success("⚡ PDF already indexed. Loaded existing data.")
                except Exception as e:
                    st.error(f"Failed to load chain: {e}")
            else:
                st.warning("BM25 params missing locally. Re-ingesting to refit encoder...")
                already_indexed = False   # fall through

        if not already_indexed:
            with st.spinner("Ingesting PDF + fitting BM25 encoder..."):
                try:
                    namespace = ingest_pdf(str(file_path))
                    st.session_state.namespace = namespace
                    st.session_state.rag_chain = get_rag_chain(
                        namespace,
                        alpha=st.session_state.alpha,
                        initial_top_k=st.session_state.initial_top_k,
                        final_top_n=st.session_state.final_top_n,
                        rerank_model=st.session_state.rerank_model,
                    )
                    st.success("✅ PDF indexed with Hybrid Search + Re-ranking.")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")


# ===============================
# Chat Section
# ===============================
st.divider()
st.subheader("💬 Chat with Your Knowledge Base")

if st.session_state.rag_chain:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving → Re-ranking → Generating..."):
                try:
                    answer = st.session_state.rag_chain.invoke(user_input)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info(
        "ℹ️ No indexed PDF loaded. Upload and process a PDF above, "
        "or ensure a previously indexed PDF and `bm25_params.json` are present."
    )