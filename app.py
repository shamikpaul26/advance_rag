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
st.title("RAG Chatbot  —  Hybrid · Fusion · Rerank · LongContextReorder")

# ===============================
# Session State
# ===============================
DEFAULTS = {
    "namespace":    None,
    "rag_chain":    None,
    "messages":     [],
    "mode":         "hybrid",
    "alpha":        0.5,
    "initial_top_k": 10,
    "final_top_n":  4,
    "rerank_model": "rerank-english-v3.0",
    "num_queries":  4,
    "top_k_per_q":  5,
    "rrf_k":        60,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def _build_chain():
    """Helper — rebuild rag_chain from current session state."""
    st.session_state.rag_chain = get_rag_chain(
        namespace=st.session_state.namespace,
        mode=st.session_state.mode,
        alpha=st.session_state.alpha,
        initial_top_k=st.session_state.initial_top_k,
        final_top_n=st.session_state.final_top_n,
        rerank_model=st.session_state.rerank_model,
        num_queries=st.session_state.num_queries,
        top_k_per_q=st.session_state.top_k_per_q,
        rrf_k=st.session_state.rrf_k,
    )


# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.header("⚙️ Retrieval Settings")

    # ── Mode selector ─────────────────────────────────────────────────
    mode = st.radio(
        "Retrieval Mode",
        options=["hybrid", "fusion"],
        format_func=lambda m: {
            "hybrid": "⚡ Hybrid  (BM25 + Vector)",
            "fusion": "🔀 RAG Fusion  (Multi-query + RRF)",
        }[m],
        index=["hybrid", "fusion"].index(st.session_state.mode),
        help=(
            "**Hybrid** — single query, BM25 + dense blend, fast.\n\n"
            "**RAG Fusion** — generates N sub-queries, searches each, "
            "merges results with Reciprocal Rank Fusion. Higher recall, slower."
        ),
    )

    st.divider()

    # ── Shared: alpha ─────────────────────────────────────────────────
    alpha = st.slider(
        "Alpha  (0 = BM25-only · 1 = Vector-only)",
        0.0, 1.0, st.session_state.alpha, 0.05,
        help="BM25 vs dense blend applied to every Pinecone query.",
    )
    if   alpha == 0.0: st.caption("🔤 Pure Keyword (BM25)")
    elif alpha == 1.0: st.caption("🧠 Pure Semantic (Vector)")
    elif alpha <  0.4: st.caption("🔤➕ Keyword-heavy Hybrid")
    elif alpha >  0.6: st.caption("🧠➕ Semantic-heavy Hybrid")
    else:              st.caption("⚖️ Balanced Hybrid")

    st.divider()

    # ── Hybrid-specific controls ──────────────────────────────────────
    if mode == "hybrid":
        st.subheader("⚡ Hybrid Settings")
        initial_top_k = st.slider(
            "Candidates fetched (initial_top_k)",
            4, 20, st.session_state.initial_top_k, 1,
            help="Chunks Pinecone returns before re-ranking.",
        )
        num_queries = st.session_state.num_queries   # unused but kept in state
        top_k_per_q = st.session_state.top_k_per_q
        rrf_k       = st.session_state.rrf_k

    # ── Fusion-specific controls ──────────────────────────────────────
    else:
        st.subheader("🔀 RAG Fusion Settings")

        num_queries = st.slider(
            "Sub-queries to generate",
            2, 8, st.session_state.num_queries, 1,
            help=(
                "GPT-4o-mini rephrases the original query this many times. "
                "More = better recall, more LLM calls."
            ),
        )
        top_k_per_q = st.slider(
            "Top-k per sub-query",
            2, 10, st.session_state.top_k_per_q, 1,
            help="Pinecone results fetched for each sub-query before RRF.",
        )
        rrf_k = st.slider(
            "RRF constant (k)",
            10, 100, st.session_state.rrf_k, 5,
            help=(
                "RRF score = Σ 1/(k + rank). "
                "Higher k reduces the dominance of top-ranked docs."
            ),
        )
        initial_top_k = num_queries * top_k_per_q   # informational
        st.caption(f"Max candidates into re-ranker: ~{initial_top_k} (after RRF dedup)")

    st.divider()

    # ── Re-ranking ────────────────────────────────────────────────────
    st.subheader("Re-ranking (Cohere)")

    rerank_model = st.selectbox(
        "Re-rank Model",
        ["rerank-english-v3.0", "rerank-multilingual-v3.0", "rerank-english-v2.0"],
        index=0,
    )
    final_top_n = st.slider(
        "Chunks kept after re-ranking (final_top_n)",
        1, 8, st.session_state.final_top_n, 1,
        help="Top-N docs passed to LongContextReorder then to the LLM.",
    )

    # ── LongContextReorder ────────────────────────────────────────────
    st.divider()
    st.subheader("📐 LongContextReorder")
    st.info(
        "Best docs placed at **start & end** of context window automatically. "
        "Reduces LLM 'lost-in-the-middle' attention bias."
    )

    # ── Pipeline summary ──────────────────────────────────────────────
    st.divider()
    st.markdown("**Pipeline summary**")
    if mode == "hybrid":
        st.markdown(
            f"Hybrid search → `{st.session_state.initial_top_k}` candidates  \n"
            f"→ Cohere re-rank → top `{final_top_n}`  \n"
            f"→ LongContextReorder → GPT-4o-mini"
        )
    else:
        st.markdown(
            f"GPT-4o-mini generates `{num_queries}` sub-queries  \n"
            f"→ Hybrid search `{top_k_per_q}` each → RRF fusion  \n"
            f"→ Cohere re-rank → top `{final_top_n}`  \n"
            f"→ LongContextReorder → GPT-4o-mini"
        )

    st.divider()

    # Detect any setting change and rebuild chain
    new_state = dict(
        mode=mode, alpha=alpha,
        initial_top_k=initial_top_k if mode == "hybrid" else st.session_state.initial_top_k,
        final_top_n=final_top_n, rerank_model=rerank_model,
        num_queries=num_queries, top_k_per_q=top_k_per_q, rrf_k=rrf_k,
    )
    changed = any(st.session_state[k] != v for k, v in new_state.items())

    if changed and st.session_state.namespace:
        for k, v in new_state.items():
            st.session_state[k] = v
        try:
            _build_chain()
            st.success("✅ Chain updated.")
        except Exception as e:
            st.error(f"Failed to rebuild chain: {e}")
    elif changed:
        for k, v in new_state.items():
            st.session_state[k] = v

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ===============================
# AUTO LOAD EXISTING DATA
# ===============================
if not st.session_state.rag_chain:
    if Path(BM25_PARAMS_PATH).exists():
        existing_ns = check_existing_namespace()
        if existing_ns:
            try:
                st.session_state.namespace = existing_ns
                _build_chain()
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

        expected_ns    = generate_namespace(str(file_path))
        existing_ns    = check_existing_namespace()
        already_indexed = (existing_ns == expected_ns)

        if already_indexed and Path(BM25_PARAMS_PATH).exists():
            st.session_state.namespace = expected_ns
            try:
                _build_chain()
                st.success("⚡ PDF already indexed. Loaded existing data.")
            except Exception as e:
                st.error(f"Failed to load chain: {e}")
        else:
            if already_indexed:
                st.warning("BM25 params missing — re-ingesting to refit encoder...")
            with st.spinner("Ingesting PDF + fitting BM25 encoder..."):
                try:
                    ns = ingest_pdf(str(file_path))
                    st.session_state.namespace = ns
                    _build_chain()
                    st.success("✅ PDF indexed successfully.")
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {e}")


# ===============================
# Chat Section
# ===============================
st.divider()
st.subheader("💬 Chat with Your Knowledge Base")

if st.session_state.rag_chain:

    # Mode badge
    if st.session_state.mode == "fusion":
        st.caption(
            f"🔀 RAG Fusion active — generating {st.session_state.num_queries} sub-queries per question"
        )
    else:
        st.caption("⚡ Hybrid mode active")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your PDF...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        spinner_msg = (
            "Generating sub-queries → Searching → RRF → Re-ranking → Generating..."
            if st.session_state.mode == "fusion"
            else "Searching → Re-ranking → Reordering → Generating..."
        )

        with st.chat_message("assistant"):
            with st.spinner(spinner_msg):
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