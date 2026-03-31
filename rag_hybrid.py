import os
import hashlib
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
import cohere

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers.merger_retriever import MergerRetriever         # LOTR — merges N retrievers
from langchain_community.document_transformers import LongContextReorder

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME, COHERE_API_KEY


# -----------------------------------------
# Generate Namespace
# -----------------------------------------

def generate_namespace(file_path: str) -> str:
    """Generate a unique MD5-based namespace from a file path."""
    return hashlib.md5(file_path.encode()).hexdigest()


# -----------------------------------------
# Create Pinecone Index (dotproduct for Hybrid)
# -----------------------------------------

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
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"[INFO] Created Pinecone index: {PINECONE_INDEX_NAME}")
    else:
        print(f"[INFO] Index '{PINECONE_INDEX_NAME}' already exists.")


# -----------------------------------------
# Check Existing Namespace
# -----------------------------------------

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


# -----------------------------------------
# BM25 Encoder — fit & save / load helpers
# -----------------------------------------

BM25_PARAMS_PATH = "bm25_params.json"


def _fit_and_save_bm25(corpus: list[str]) -> BM25Encoder:
    """Fit BM25 on the ingested corpus and persist params to disk."""
    encoder = BM25Encoder()
    encoder.fit(corpus)
    encoder.dump(BM25_PARAMS_PATH)
    print(f"[INFO] BM25 params saved to '{BM25_PARAMS_PATH}'.")
    return encoder


def _load_bm25() -> BM25Encoder:
    """Load a previously fitted BM25 encoder from disk."""
    if not os.path.exists(BM25_PARAMS_PATH):
        raise FileNotFoundError(
            f"BM25 params not found at '{BM25_PARAMS_PATH}'. "
            "Ingest a PDF first so the encoder can be fitted."
        )
    encoder = BM25Encoder()
    encoder.load(BM25_PARAMS_PATH)
    return encoder


# -----------------------------------------
# Ingest PDF — Dense + Sparse upsert
# -----------------------------------------

def ingest_pdf(file_path: str) -> str:
    """
    Load, chunk, and upsert a PDF into Pinecone with hybrid vectors.
    Each record stores:
      - values        → dense OpenAI embedding
      - sparse_values → BM25 term weights
      - metadata      → source + raw text
    """
    namespace = generate_namespace(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks from '{file_path}'.")

    bm25_encoder = _fit_and_save_bm25(texts)

    dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dense_vectors = dense_embeddings.embed_documents(texts)
    sparse_vectors = bm25_encoder.encode_documents(texts)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 100
    records = []

    for i, (text, dense, sparse) in enumerate(zip(texts, dense_vectors, sparse_vectors)):
        records.append({
            "id": f"{namespace}-chunk-{i}",
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "text": text,
                "source": file_path,
                "chunk_index": i,
            }
        })

    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"[INFO] Upserted chunks {start}–{start + len(batch) - 1}.")

    print(f"[INFO] Ingestion complete. Namespace: '{namespace}'.")
    return namespace


# -----------------------------------------
# Format Documents for Prompt
# -----------------------------------------

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------------------
# Cohere Re-ranker
# -----------------------------------------

class CohereReranker:
    """
    Re-ranks a list of LangChain Documents using Cohere's cross-encoder.

    Unlike bi-encoders (used in vector search), the cross-encoder reads
    the query AND document TOGETHER, giving much more accurate scores.
    """

    def __init__(
        self,
        top_n: int = 3,
        model: str = "rerank-english-v3.0",
    ):
        self.top_n = top_n
        self.model = model
        self.client = cohere.Client(api_key=COHERE_API_KEY)

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=self.top_n,
            return_documents=False,
        )

        reranked_docs = []
        for result in response.results:
            doc = documents[result.index]
            doc.metadata["rerank_score"] = round(result.relevance_score, 4)
            reranked_docs.append(doc)

        print(
            f"[INFO] Re-ranked {len(documents)} → kept top {len(reranked_docs)} docs. "
            f"Scores: {[d.metadata['rerank_score'] for d in reranked_docs]}"
        )

        return reranked_docs


# -----------------------------------------
# Dense-only LangChain BaseRetriever
# (needed so MergerRetriever can wrap it)
# -----------------------------------------

class DenseRetriever(BaseRetriever):
    """
    A standard LangChain BaseRetriever backed by Pinecone dense vectors only.

    This is ONE of the two sources fed into MergerRetriever.
    It uses cosine-style semantic search via PineconeVectorStore.

    Why separate from HybridRetriever?
    MergerRetriever requires LangChain BaseRetriever instances.
    We expose dense-only and hybrid-only as separate retrievers so
    MergerRetriever can independently pull candidates from each
    strategy before merging and deduplicating.
    """

    namespace: str
    top_k: int = 5

    # Pydantic fields — not init args
    _vectorstore: PineconeVectorStore = None

    def __init__(self, namespace: str, top_k: int = 5, **kwargs):
        super().__init__(namespace=namespace, top_k=top_k, **kwargs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=namespace,
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:
        docs = self._vectorstore.similarity_search(query, k=self.top_k)
        for doc in docs:
            doc.metadata["retriever_source"] = "dense"
        print(f"[INFO] DenseRetriever returned {len(docs)} docs.")
        return docs


# -----------------------------------------
# Hybrid-only LangChain BaseRetriever
# (needed so MergerRetriever can wrap it)
# -----------------------------------------

class HybridBaseRetriever(BaseRetriever):
    """
    A LangChain BaseRetriever that performs BM25 + dense hybrid search
    against Pinecone. This is the SECOND source fed into MergerRetriever.

    Exposes the same interface as DenseRetriever so MergerRetriever
    treats both uniformly.
    """

    namespace: str
    alpha: float = 0.5
    top_k: int = 5

    _pc_index: object = None
    _embeddings: OpenAIEmbeddings = None
    _bm25_encoder: BM25Encoder = None

    def __init__(self, namespace: str, alpha: float = 0.5, top_k: int = 5, **kwargs):
        super().__init__(namespace=namespace, alpha=alpha, top_k=top_k, **kwargs)
        self._pc_index  = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
        self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._bm25_encoder = _load_bm25()

    def _scale_scores(self, dense, sparse, alpha):
        return (
            [v * alpha for v in dense],
            {
                "indices": sparse["indices"],
                "values":  [v * (1 - alpha) for v in sparse["values"]],
            }
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:
        dense_q  = self._embeddings.embed_query(query)
        sparse_q = self._bm25_encoder.encode_queries(query)
        scaled_dense, scaled_sparse = self._scale_scores(dense_q, sparse_q, self.alpha)

        results = self._pc_index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        docs = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            metadata["hybrid_score"]      = round(match.get("score", 0), 4)
            metadata["retriever_source"]  = "hybrid"
            docs.append(Document(page_content=text, metadata=metadata))

        print(f"[INFO] HybridBaseRetriever returned {len(docs)} docs.")
        return docs


# -----------------------------------------
# MergedHybridRetriever
# Orchestrates: MergerRetriever → Dedup →
#               CohereReranker → LongContextReorder
# -----------------------------------------

class MergedHybridRetriever:
    """
    Full retrieval pipeline combining two strategies via MergerRetriever,
    then applying Cohere re-ranking and LongContextReorder.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Query                                                              │
    │    ├── DenseRetriever       (pure semantic, top_k=5)               │
    │    └── HybridBaseRetriever  (BM25 + dense blend, top_k=5)          │
    │                ↓                                                    │
    │         MergerRetriever  (concatenates both result lists)           │
    │                ↓                                                    │
    │         Deduplication    (remove identical page_content)            │
    │                ↓                                                    │
    │         CohereReranker   (cross-encoder scores, keep top_n)         │
    │                ↓                                                    │
    │         LongContextReorder                                          │
    │           → most relevant docs placed at START and END              │
    │           → least relevant docs placed in the MIDDLE                │
    │           → combats "lost in the middle" LLM attention bias         │
    └─────────────────────────────────────────────────────────────────────┘

    Why LongContextReorder?
    Research shows LLMs perform best on content at the beginning and end
    of their context window, and tend to ignore content in the middle.
    LongContextReorder interleaves docs so the highest-scored chunks
    appear at positions the LLM pays most attention to.

    Parameters
    ----------
    namespace      : Pinecone namespace from ingest_pdf()
    alpha          : BM25/dense blend for HybridBaseRetriever
    dense_top_k    : candidates from DenseRetriever
    hybrid_top_k   : candidates from HybridBaseRetriever
    final_top_n    : docs kept after Cohere re-ranking
    rerank_model   : Cohere re-rank model name
    """

    def __init__(
        self,
        namespace: str,
        alpha: float = 0.5,
        dense_top_k: int = 5,
        hybrid_top_k: int = 5,
        final_top_n: int = 4,
        rerank_model: str = "rerank-english-v3.0",
    ):
        # ── Two independent retrievers ────────────────────────────────
        dense_retriever = DenseRetriever(
            namespace=namespace,
            top_k=dense_top_k,
        )
        hybrid_retriever = HybridBaseRetriever(
            namespace=namespace,
            alpha=alpha,
            top_k=hybrid_top_k,
        )

        # ── MergerRetriever (LOTR) ────────────────────────────────────
        # Calls both retrievers in parallel and concatenates their results.
        # Automatically deduplicates by document ID when available.
        self.merger = MergerRetriever(
            retrievers=[dense_retriever, hybrid_retriever]
        )

        # ── Cohere Re-ranker ──────────────────────────────────────────
        self.reranker = CohereReranker(top_n=final_top_n, model=rerank_model)

        # ── LongContextReorder ────────────────────────────────────────
        # Reorders so best docs are at start/end, worst in the middle.
        self.reorder = LongContextReorder()

    def _deduplicate(self, docs: list[Document]) -> list[Document]:
        """Remove duplicate chunks by page_content."""
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        print(f"[INFO] After dedup: {len(unique)} unique docs.")
        return unique

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Step 1 — Merge results from both retrievers
        merged = self.merger.invoke(query)
        print(f"[INFO] MergerRetriever returned {len(merged)} total docs.")

        # Step 2 — Deduplicate
        unique_docs = self._deduplicate(merged)

        # Step 3 — Cohere re-ranking
        reranked = self.reranker.rerank(query=query, documents=unique_docs)

        # Step 4 — LongContextReorder
        # Reorders so highest-relevance docs appear at start & end of list.
        reordered = self.reorder.transform_documents(reranked)
        print(f"[INFO] LongContextReorder applied. Final doc count: {len(reordered)}.")

        return reordered

    def __call__(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


# -----------------------------------------
# RAG Chain
# -----------------------------------------

def get_rag_chain(
    namespace: str,
    alpha: float = 0.5,
    dense_top_k: int = 5,
    hybrid_top_k: int = 5,
    final_top_n: int = 4,
    rerank_model: str = "rerank-english-v3.0",
):
    """
    Build the full RAG chain:

      DenseRetriever ──┐
                       ├─► MergerRetriever → Dedup → CohereRerank
      HybridRetriever ─┘                           → LongContextReorder
                                                   → GPT-4o-mini → Answer

    Parameters
    ----------
    namespace      : Pinecone namespace from ingest_pdf()
    alpha          : BM25/dense blend (0=keyword, 1=vector, 0.5=balanced)
    dense_top_k    : candidates from pure dense retriever
    hybrid_top_k   : candidates from hybrid BM25+dense retriever
    final_top_n    : docs kept after Cohere re-ranking
    rerank_model   : Cohere re-rank model name
    """

    retriever = MergedHybridRetriever(
        namespace=namespace,
        alpha=alpha,
        dense_top_k=dense_top_k,
        hybrid_top_k=hybrid_top_k,
        final_top_n=final_top_n,
        rerank_model=rerank_model,
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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