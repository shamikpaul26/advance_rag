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
st.title("📄🔍 Hybrid RAG  —  BM25 + Vector + MergerRetriever + LongContextReorder")

# ===============================
# Session State
# ===============================
for key, default in [
    ("namespace",     None),
    ("rag_chain",     None),
    ("messages",      []),
    ("alpha",         0.5),
    ("dense_top_k",   5),
    ("hybrid_top_k",  5),
    ("final_top_n",   4),
    ("rerank_model",  "rerank-english-v3.0"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===============================
# Sidebar — Controls
# ===============================
with st.sidebar:
    st.header("⚙️ Retrieval Settings")

    # ── Hybrid alpha ──────────────────────────────────────────────────
    alpha = st.slider(
        "Alpha  (0 = BM25-only · 1 = Vector-only)",
        min_value=0.0, max_value=1.0,
        value=st.session_state.alpha, step=0.05,
        help="Controls the BM25 vs dense blend inside HybridBaseRetriever.",
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

    # ── MergerRetriever knobs ─────────────────────────────────────────
    st.subheader("🔀 MergerRetriever")
    st.caption(
        "Two retrievers run independently and their results are merged + deduplicated "
        "before re-ranking."
    )

    dense_top_k = st.slider(
        "Dense retriever top_k",
        min_value=2, max_value=15,
        value=st.session_state.dense_top_k, step=1,
        help="Candidates fetched by the pure semantic (dense) retriever.",
    )

    hybrid_top_k = st.slider(
        "Hybrid retriever top_k",
        min_value=2, max_value=15,
        value=st.session_state.hybrid_top_k, step=1,
        help="Candidates fetched by the BM25 + dense hybrid retriever.",
    )

    total_candidates = dense_top_k + hybrid_top_k
    st.caption(f"Max merged candidates before dedup: **{total_candidates}**")

    st.divider()

    # ── Re-ranking ────────────────────────────────────────────────────
    st.subheader("🎯 Re-ranking (Cohere)")

    rerank_model = st.selectbox(
        "Re-rank Model",
        options=[
            "rerank-english-v3.0",
            "rerank-multilingual-v3.0",
            "rerank-english-v2.0",
        ],
        index=0,
    )

    final_top_n = st.slider(
        "Chunks kept after re-ranking (final_top_n)",
        min_value=1, max_value=min(total_candidates, 10),
        value=min(st.session_state.final_top_n, total_candidates), step=1,
        help="Top-N chunks passed to LongContextReorder and then to the LLM.",
    )

    st.divider()

    # ── LongContextReorder info ───────────────────────────────────────
    st.subheader("📐 LongContextReorder")
    st.info(
        "After re-ranking, docs are reordered so the **most relevant chunks "
        "appear at the start and end** of the context window, and least relevant "
        "in the middle — reducing LLM 'lost-in-the-middle' attention bias. "
        "This is always applied automatically."
    )

    # Pipeline summary
    st.divider()
    st.markdown("**Pipeline summary**")
    st.markdown(
        f"DenseRetriever `{dense_top_k}` + HybridRetriever `{hybrid_top_k}`  \n"
        f"→ MergerRetriever → Dedup  \n"
        f"→ Cohere re-rank → top `{final_top_n}`  \n"
        f"→ LongContextReorder → GPT-4o-mini"
    )

    st.divider()

    # Rebuild chain when settings change
    settings_changed = (
        alpha         != st.session_state.alpha         or
        dense_top_k   != st.session_state.dense_top_k   or
        hybrid_top_k  != st.session_state.hybrid_top_k  or
        final_top_n   != st.session_state.final_top_n   or
        rerank_model  != st.session_state.rerank_model
    )

    if settings_changed and st.session_state.namespace:
        st.session_state.alpha        = alpha
        st.session_state.dense_top_k  = dense_top_k
        st.session_state.hybrid_top_k = hybrid_top_k
        st.session_state.final_top_n  = final_top_n
        st.session_state.rerank_model = rerank_model
        try:
            st.session_state.rag_chain = get_rag_chain(
                st.session_state.namespace,
                alpha=alpha,
                dense_top_k=dense_top_k,
                hybrid_top_k=hybrid_top_k,
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
                    dense_top_k=st.session_state.dense_top_k,
                    hybrid_top_k=st.session_state.hybrid_top_k,
                    final_top_n=st.session_state.final_top_n,
                    rerank_model=st.session_state.rerank_model,
                )
                st.toast("⚡ Auto-loaded existing indexed data.")
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

        expected_namespace = generate_namespace(str(file_path))
        existing_namespace = check_existing_namespace()
        already_indexed    = (existing_namespace == expected_namespace)

        if already_indexed and Path(BM25_PARAMS_PATH).exists():
            st.session_state.namespace = expected_namespace
            try:
                st.session_state.rag_chain = get_rag_chain(
                    expected_namespace,
                    alpha=st.session_state.alpha,
                    dense_top_k=st.session_state.dense_top_k,
                    hybrid_top_k=st.session_state.hybrid_top_k,
                    final_top_n=st.session_state.final_top_n,
                    rerank_model=st.session_state.rerank_model,
                )
                st.success("⚡ PDF already indexed. Loaded existing data.")
            except Exception as e:
                st.error(f"Failed to load chain: {e}")
        else:
            if already_indexed and not Path(BM25_PARAMS_PATH).exists():
                st.warning("BM25 params missing locally — re-ingesting to refit encoder...")

            with st.spinner("Ingesting PDF + fitting BM25 encoder..."):
                try:
                    namespace = ingest_pdf(str(file_path))
                    st.session_state.namespace = namespace
                    st.session_state.rag_chain = get_rag_chain(
                        namespace,
                        alpha=st.session_state.alpha,
                        dense_top_k=st.session_state.dense_top_k,
                        hybrid_top_k=st.session_state.hybrid_top_k,
                        final_top_n=st.session_state.final_top_n,
                        rerank_model=st.session_state.rerank_model,
                    )
                    st.success("✅ PDF indexed with MergerRetriever + LongContextReorder.")
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
            with st.spinner("Merging → Re-ranking → Reordering → Generating..."):
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