import os
import hashlib
from collections import defaultdict
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import cohere

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, COHERE_API_KEY


# ─────────────────────────────────────────
# LLM model registry
# ─────────────────────────────────────────

# Models used at different stages:
#   FUSION_LLM  — generates sub-queries         (cheap, fast)
#   ANSWER_LLM  — synthesises the final answer  (smart)
FUSION_LLM_MODEL = "gpt-4o-mini"       # query expansion
ANSWER_LLM_MODEL = "gpt-4o-mini"       # final answer  ← swap to "gpt-4o" for higher quality


# ─────────────────────────────────────────
# Generate Namespace
# ─────────────────────────────────────────

def generate_namespace(file_path: str) -> str:
    """Generate a unique MD5-based namespace from a file path."""
    return hashlib.md5(file_path.encode()).hexdigest()


# ─────────────────────────────────────────
# Create Pinecone Index  (dotproduct for Hybrid)
# ─────────────────────────────────────────

def create_index_if_not_exists():
    """
    Create a Pinecone index configured for hybrid search.
    Requires metric='dotproduct' (NOT cosine) for sparse+dense hybrid.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[INFO] Created Pinecone index: {PINECONE_INDEX_NAME}")
    else:
        print(f"[INFO] Index '{PINECONE_INDEX_NAME}' already exists.")


# ─────────────────────────────────────────
# Check Existing Namespace
# ─────────────────────────────────────────

def check_existing_namespace(filename: str = None) -> str | None:
    """
    Check if a namespace already exists in Pinecone.
    Returns the first available namespace, or None.
    Safely returns None if the index does not exist yet.
    """
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing_indexes = pc.list_indexes().names()
        if PINECONE_INDEX_NAME not in existing_indexes:
            print(f"[INFO] Index '{PINECONE_INDEX_NAME}' does not exist yet.")
            return None

        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})

        if not namespaces:
            return None

        if filename:
            return filename if filename in namespaces else None

        return list(namespaces.keys())[0]

    except Exception as e:
        print(f"[WARN] check_existing_namespace failed: {e}")
        return None


# ─────────────────────────────────────────
# BM25 Encoder  — fit / save / load
# ─────────────────────────────────────────

BM25_PARAMS_PATH = "bm25_params.json"


def _fit_and_save_bm25(corpus: list[str]) -> BM25Encoder:
    encoder = BM25Encoder()
    encoder.fit(corpus)
    encoder.dump(BM25_PARAMS_PATH)
    print(f"[INFO] BM25 params saved to '{BM25_PARAMS_PATH}'.")
    return encoder


def _load_bm25() -> BM25Encoder:
    if not os.path.exists(BM25_PARAMS_PATH):
        raise FileNotFoundError(
            f"BM25 params not found at '{BM25_PARAMS_PATH}'. "
            "Ingest a PDF first so the encoder can be fitted."
        )
    encoder = BM25Encoder()
    encoder.load(BM25_PARAMS_PATH)
    return encoder


# ─────────────────────────────────────────
# Ingest PDF  — Dense + Sparse upsert
# ─────────────────────────────────────────

def ingest_pdf(file_path: str) -> str:
    """
    Load, chunk, and upsert a PDF into Pinecone with hybrid vectors.
    Each record stores:
      - values        → dense OpenAI embedding
      - sparse_values → BM25 term weights
      - metadata      → source + raw text
    """
    namespace = generate_namespace(file_path)

    loader    = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks   = splitter.split_documents(documents)
    texts    = [c.page_content for c in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks from '{file_path}'.")

    bm25_encoder  = _fit_and_save_bm25(texts)
    dense_embed   = OpenAIEmbeddings(model="text-embedding-3-small")
    dense_vectors = dense_embed.embed_documents(texts)
    sparse_vectors = bm25_encoder.encode_documents(texts)

    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    records = [
        {
            "id": f"{namespace}-chunk-{i}",
            "values": dense,
            "sparse_values": sparse,
            "metadata": {"text": text, "source": file_path, "chunk_index": i},
        }
        for i, (text, dense, sparse) in enumerate(zip(texts, dense_vectors, sparse_vectors))
    ]

    batch_size = 100
    for start in range(0, len(records), batch_size):
        index.upsert(vectors=records[start: start + batch_size], namespace=namespace)
        print(f"[INFO] Upserted chunks {start}–{start + min(batch_size, len(records)-start) - 1}.")

    print(f"[INFO] Ingestion complete. Namespace: '{namespace}'.")
    return namespace


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    # Filter out any docs that slipped through with empty content
    filled = [doc for doc in docs if doc.page_content.strip()]
    if not filled:
        print("[WARN] format_docs received no non-empty documents.")
        return ""
    return "\n\n".join(doc.page_content for doc in filled)


# ─────────────────────────────────────────
# Cohere Re-ranker
# ─────────────────────────────────────────

class CohereReranker:
    """
    Cross-encoder re-ranker via Cohere API.
    Reads query + document TOGETHER for more accurate relevance scores.
    """

    def __init__(self, top_n: int = 4, model: str = "rerank-english-v3.0"):
        self.top_n  = top_n
        self.model  = model
        self.client = cohere.Client(api_key=COHERE_API_KEY)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[d.page_content for d in documents],
            top_n=self.top_n,
            return_documents=False,
        )

        reranked = []
        for r in response.results:
            doc = documents[r.index]
            doc.metadata["rerank_score"] = round(r.relevance_score, 4)
            reranked.append(doc)

        print(
            f"[INFO] Re-ranked {len(documents)} → top {len(reranked)}. "
            f"Scores: {[d.metadata['rerank_score'] for d in reranked]}"
        )
        return reranked


# ─────────────────────────────────────────
# Low-level Hybrid Search helper
# (shared by both HybridRetriever and RAGFusionRetriever)
# ─────────────────────────────────────────

class _HybridSearchEngine:
    """
    Thin wrapper around Pinecone hybrid query.
    Reused by HybridRetriever and RAGFusionRetriever to avoid duplication.
    """

    def __init__(self, namespace: str, alpha: float, bm25_encoder: BM25Encoder):
        self.namespace    = namespace
        self.alpha        = alpha
        self.bm25_encoder = bm25_encoder
        self.pc_index     = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
        self.embeddings   = OpenAIEmbeddings(model="text-embedding-3-small")

    def search(self, query: str, top_k: int) -> list[Document]:
        dense_q  = self.embeddings.embed_query(query)
        sparse_q = self.bm25_encoder.encode_queries(query)

        scaled_dense  = [v * self.alpha for v in dense_q]
        scaled_sparse = {
            "indices": sparse_q["indices"],
            "values":  [v * (1 - self.alpha) for v in sparse_q["values"]],
        }

        results = self.pc_index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        docs = []
        for match in results.get("matches", []):
            # Defensive copy: Pinecone returns metadata as a read-only dict
            # snapshot in newer SDK versions. Calling .pop() on it either
            # silently fails or mutates a shared object, leaving
            # page_content="" which causes the LLM to answer "I don't know."
            # Always copy before extracting text.
            raw_meta = match.get("metadata") or {}
            meta     = dict(raw_meta)                  # safe mutable copy
            text     = meta.pop("text", "")             # extract → page_content
            meta["hybrid_score"] = round(match.get("score", 0), 4)

            if not text:
                # Skip empty chunks so they never reach the LLM prompt
                print(f"[WARN] Empty text for match id={match.get('id')} — skipping.")
                continue

            docs.append(Document(page_content=text, metadata=meta))

        return docs


# ─────────────────────────────────────────
# RAG Fusion  — multi-query + RRF
# ─────────────────────────────────────────

class RAGFusion:
    """
    RAG Fusion pipeline:

    ┌──────────────────────────────────────────────────────────────────────┐
    │  Original Query                                                      │
    │       ↓                                                              │
    │  GPT-4o-mini  →  generates N rephrased sub-queries                  │
    │       ↓                                                              │
    │  For each sub-query: Hybrid Search (BM25 + Dense) → top_k docs      │
    │       ↓                                                              │
    │  Reciprocal Rank Fusion (RRF)                                        │
    │    → merges ranked lists from all sub-queries                        │
    │    → score(doc) = Σ  1 / (rank_i + k)  across all query lists       │
    │    → produces a single unified ranking                               │
    │       ↓                                                              │
    │  Top-N documents by RRF score                                        │
    └──────────────────────────────────────────────────────────────────────┘

    Why RAG Fusion?
    A single query may miss relevant documents due to vocabulary mismatch.
    Generating multiple rephrasings and fusing results with RRF dramatically
    improves recall and reduces sensitivity to how the query was phrased.

    RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
      k=60 is the standard constant (dampens high-rank advantage).
      A document appearing in rank 1 of every sub-query scores highest.

    Parameters
    ----------
    search_engine  : shared _HybridSearchEngine instance
    num_queries    : number of sub-queries to generate (default 4)
    top_k_per_q   : Pinecone results per sub-query
    rrf_k          : RRF constant (default 60)
    rrf_top_n      : docs to return after RRF fusion
    """

    # Sub-query generation prompt
    _QUERY_GEN_PROMPT = ChatPromptTemplate.from_template("""
You are an AI assistant helping improve document retrieval.
Given the user's question below, generate {num_queries} different rephrasings
of it to improve search recall. Each rephrasing should approach the question
from a slightly different angle or use different vocabulary.

Output ONLY the {num_queries} questions, one per line, no numbering, no extra text.

Original question: {question}
""")

    def __init__(
        self,
        search_engine: "_HybridSearchEngine",
        num_queries: int = 4,
        top_k_per_q: int = 5,
        rrf_k: int = 60,
        rrf_top_n: int = 10,
    ):
        self.search_engine = search_engine
        self.num_queries   = num_queries
        self.top_k_per_q   = top_k_per_q
        self.rrf_k         = rrf_k
        self.rrf_top_n     = rrf_top_n

        self.query_gen_chain = (
            self._QUERY_GEN_PROMPT
            | ChatOpenAI(model=FUSION_LLM_MODEL, temperature=0.3)
            | StrOutputParser()
        )

    def _generate_queries(self, question: str) -> list[str]:
        """
        Use GPT-4o-mini to generate N rephrasings of the original question.

        Fixes applied:
        1. Strip leading numbering/bullets GPT commonly adds
           e.g. "1. How does..." becomes "How does..."
        2. Skip empty lines after stripping
        3. Deduplicate while preserving order
        4. Always include the original question first
        """
        import re

        raw = self.query_gen_chain.invoke({
            "question":    question,
            "num_queries": self.num_queries,
        })

        cleaned = []
        for line in raw.strip().splitlines():
            # Strip leading numbering: "1.", "1)", "-", "*", "•"
            line = re.sub(r"^\s*(\d+[\.):]\s*|[-*•]\s*)", "", line).strip()
            if line:
                cleaned.append(line)

        # Deduplicate preserving order, then prepend the original
        seen = {question}
        unique = [question]
        for q in cleaned:
            if q not in seen:
                seen.add(q)
                unique.append(q)

        print(f"[INFO] RAG Fusion generated {len(unique)} queries: {unique}")
        return unique

    def _reciprocal_rank_fusion(
        self,
        ranked_lists: list[list[Document]],
    ) -> list[Document]:
        """
        Merge N ranked document lists using Reciprocal Rank Fusion.

        score(doc) = Σ  1 / (k + rank)   summed over all lists that contain doc.
        Documents are keyed by their page_content to handle duplicates.
        """
        scores: dict[str, float]   = defaultdict(float)
        doc_map: dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content.strip()
                if not key:
                    # Skip empty docs — they collapse into one "" key and
                    # pass empty text to the LLM, causing "I don't know."
                    print("[WARN] RRF: skipping doc with empty page_content.")
                    continue
                scores[key] += 1.0 / (self.rrf_k + rank)
                if key not in doc_map:
                    doc_map[key] = doc

        # Sort by descending RRF score
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)

        fused = []
        for key in sorted_keys[: self.rrf_top_n]:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = round(scores[key], 6)
            fused.append(doc)

        print(
            f"[INFO] RRF fused {sum(len(r) for r in ranked_lists)} docs "
            f"→ {len(fused)} unique, top-{self.rrf_top_n} kept."
        )
        return fused

    def retrieve(self, question: str) -> list[Document]:
        """Generate sub-queries, search each, fuse with RRF."""
        queries      = self._generate_queries(question)
        ranked_lists = []

        for q in queries:
            docs = self.search_engine.search(q, top_k=self.top_k_per_q)
            ranked_lists.append(docs)
            print(f"[INFO] Sub-query '{q[:60]}…' → {len(docs)} docs")

        return self._reciprocal_rank_fusion(ranked_lists)


# ─────────────────────────────────────────
# Unified Retriever
# Hybrid  →  (optional RAG Fusion RRF)
#         →  Cohere Re-rank
#         →  LongContextReorder
# ─────────────────────────────────────────

class UnifiedRetriever:
    """
    Single retriever class that supports two modes, selectable at runtime:

    MODE: "hybrid"
    ──────────────
    ┌────────────────────────────────────────────────────────────────┐
    │  Query                                                         │
    │    ├── OpenAI embed()  →  dense vector                        │
    │    └── BM25 encode()   →  sparse vector                       │
    │  Pinecone hybrid query (initial_top_k candidates)             │
    │  Cohere Re-rank  →  final_top_n                               │
    │  LongContextReorder                                            │
    └────────────────────────────────────────────────────────────────┘

    MODE: "fusion"
    ──────────────
    ┌────────────────────────────────────────────────────────────────┐
    │  Query                                                         │
    │    └── GPT-4o-mini  →  N rephrased sub-queries                │
    │  Each sub-query: Hybrid Search  →  top_k docs                 │
    │  Reciprocal Rank Fusion (RRF)  →  fused ranking               │
    │  Cohere Re-rank  →  final_top_n                               │
    │  LongContextReorder                                            │
    └────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    namespace      : Pinecone namespace from ingest_pdf()
    mode           : "hybrid" | "fusion"
    alpha          : BM25/dense blend (0=BM25-only, 1=vector-only)
    initial_top_k  : candidates per Pinecone query (hybrid mode)
    final_top_n    : docs kept after Cohere re-ranking
    rerank_model   : Cohere re-rank model name
    num_queries    : sub-queries to generate (fusion mode only)
    top_k_per_q    : Pinecone results per sub-query (fusion mode only)
    rrf_k          : RRF damping constant (fusion mode only)
    """

    def __init__(
        self,
        namespace: str,
        mode: str = "hybrid",
        alpha: float = 0.5,
        initial_top_k: int = 10,
        final_top_n: int = 4,
        rerank_model: str = "rerank-english-v3.0",
        num_queries: int = 4,
        top_k_per_q: int = 5,
        rrf_k: int = 60,
    ):
        if mode not in ("hybrid", "fusion"):
            raise ValueError(f"mode must be 'hybrid' or 'fusion', got '{mode}'")

        self.mode          = mode
        self.initial_top_k = initial_top_k

        bm25 = _load_bm25()

        # Shared search engine used by both modes
        self._engine = _HybridSearchEngine(
            namespace=namespace,
            alpha=alpha,
            bm25_encoder=bm25,
        )

        # RAG Fusion component (only active in fusion mode)
        self._fusion = RAGFusion(
            search_engine=self._engine,
            num_queries=num_queries,
            top_k_per_q=top_k_per_q,
            rrf_k=rrf_k,
            rrf_top_n=initial_top_k,   # feed same budget into reranker
        )

        self._reranker = CohereReranker(top_n=final_top_n, model=rerank_model)
        self._reorder  = LongContextReorder()

    def get_relevant_documents(self, query: str) -> list[Document]:

        # ── Step 1: Retrieval (mode-dependent) ───────────────────────
        if self.mode == "fusion":
            candidates = self._fusion.retrieve(query)
        else:
            candidates = self._engine.search(query, top_k=self.initial_top_k)
            print(f"[INFO] Hybrid search returned {len(candidates)} candidates.")

        # ── Step 2: Cohere Re-ranking ─────────────────────────────────
        reranked = self._reranker.rerank(query=query, documents=candidates)

        # ── Step 3: LongContextReorder ────────────────────────────────
        reordered = self._reorder.transform_documents(reranked)
        print(f"[INFO] LongContextReorder done. Final count: {len(reordered)}.")

        return reordered

    def __call__(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


# ─────────────────────────────────────────
# RAG Chain builder
# ─────────────────────────────────────────

def get_rag_chain(
    namespace: str,
    mode: str = "hybrid",
    alpha: float = 0.5,
    initial_top_k: int = 10,
    final_top_n: int = 4,
    rerank_model: str = "rerank-english-v3.0",
    num_queries: int = 4,
    top_k_per_q: int = 5,
    rrf_k: int = 60,
):
    """
    Build the full RAG chain.

    mode="hybrid"  →  BM25+Dense → Rerank → LongContextReorder → GPT-4o-mini
    mode="fusion"  →  Multi-query + RRF → Rerank → LongContextReorder → GPT-4o-mini

    Parameters
    ----------
    namespace      : Pinecone namespace from ingest_pdf()
    mode           : "hybrid" | "fusion"
    alpha          : BM25/dense blend ratio
    initial_top_k  : Pinecone candidates before re-ranking
    final_top_n    : docs kept after re-ranking
    rerank_model   : Cohere re-rank model
    num_queries    : sub-queries for fusion mode
    top_k_per_q    : per-sub-query top_k for fusion mode
    rrf_k          : RRF constant for fusion mode
    """

    retriever = UnifiedRetriever(
        namespace=namespace,
        mode=mode,
        alpha=alpha,
        initial_top_k=initial_top_k,
        final_top_n=final_top_n,
        rerank_model=rerank_model,
        num_queries=num_queries,
        top_k_per_q=top_k_per_q,
        rrf_k=rrf_k,
    )

    llm = ChatOpenAI(model=ANSWER_LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.
Answer ONLY from the provided context.
If the answer is not found, say "I don't know."

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {
            "context":  RunnableLambda(retriever) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain