import os
import hashlib
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

        # Guard: check index exists before connecting to it
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

    How it works:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Query + Chunk-1  →  Cross-Encoder  →  Relevance Score         │
    │  Query + Chunk-2  →  Cross-Encoder  →  Relevance Score         │
    │  ...                                                            │
    │  Sort by score → keep top_n                                     │
    └─────────────────────────────────────────────────────────────────┘

    Unlike bi-encoders (used in vector search), the cross-encoder reads
    the query AND document TOGETHER, giving much more accurate scores.

    Parameters
    ----------
    top_n : int
        Number of top documents to keep after re-ranking.
    model : str
        Cohere re-rank model. Options:
          - "rerank-english-v3.0"        (English only, fastest)
          - "rerank-multilingual-v3.0"   (multilingual)
          - "rerank-english-v2.0"        (legacy)
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
        """
        Send query + document texts to Cohere re-rank API.
        Returns top_n documents sorted by cross-encoder relevance score.
        """
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]

        # Cohere re-rank API call
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=self.top_n,
            return_documents=False,   # we already have them; avoid redundant transfer
        )

        # response.results → list of RerankResult(index, relevance_score)
        reranked_docs = []
        for result in response.results:
            doc = documents[result.index]
            # Attach re-rank score to metadata for transparency / debugging
            doc.metadata["rerank_score"] = round(result.relevance_score, 4)
            reranked_docs.append(doc)

        print(
            f"[INFO] Re-ranked {len(documents)} → kept top {len(reranked_docs)} docs. "
            f"Scores: {[d.metadata['rerank_score'] for d in reranked_docs]}"
        )

        return reranked_docs


# -----------------------------------------
# Hybrid Retriever (BM25 + Dense + Rerank)
# -----------------------------------------

class HybridRetriever:
    """
    Retrieves candidates using Pinecone hybrid search (BM25 + dense vectors),
    then passes them to CohereReranker for cross-encoder re-scoring.

    Full pipeline per query:
    ┌──────────────────────────────────────────────────────────────────┐
    │  Query                                                           │
    │    ├── OpenAI embed()  →  dense vector                          │
    │    └── BM25 encode()   →  sparse vector                         │
    │                                                                  │
    │  Pinecone hybrid query (top_k = initial_top_k)                  │
    │    → alpha × dense_score + (1-alpha) × sparse_score             │
    │    → returns candidate Documents                                 │
    │                                                                  │
    │  Cohere Re-ranker (cross-encoder)                               │
    │    → reads query + each candidate together                       │
    │    → returns final_top_n most relevant Documents                 │
    └──────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    namespace      : Pinecone namespace to search.
    alpha          : Hybrid blend — 0=pure BM25, 1=pure vector, 0.5=balanced.
    initial_top_k  : Candidates to fetch from Pinecone BEFORE re-ranking.
                     Should be 3–5× larger than final_top_n.
    final_top_n    : Documents to keep AFTER re-ranking (fed to LLM).
    rerank_model   : Cohere re-rank model name.
    """

    def __init__(
        self,
        namespace: str,
        alpha: float = 0.5,
        initial_top_k: int = 10,
        final_top_n: int = 3,
        rerank_model: str = "rerank-english-v3.0",
    ):
        self.namespace = namespace
        self.alpha = alpha
        self.initial_top_k = initial_top_k

        self.pc_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.bm25_encoder = _load_bm25()
        self.reranker = CohereReranker(top_n=final_top_n, model=rerank_model)

    def _scale_scores(
        self,
        dense: list[float],
        sparse: dict,
        alpha: float,
    ) -> tuple[list[float], dict]:
        """Scale dense and sparse vectors by alpha for hybrid blending."""
        scaled_dense = [v * alpha for v in dense]
        scaled_sparse = {
            "indices": sparse["indices"],
            "values":  [v * (1 - alpha) for v in sparse["values"]],
        }
        return scaled_dense, scaled_sparse

    def get_relevant_documents(self, query: str) -> list[Document]:
        """
        Step 1 — Hybrid retrieval from Pinecone (initial_top_k candidates).
        Step 2 — Cohere re-ranking to final_top_n documents.
        """

        # ── Step 1: Hybrid Retrieval ──────────────────────────────────
        dense_query  = self.embeddings.embed_query(query)
        sparse_query = self.bm25_encoder.encode_queries(query)

        scaled_dense, scaled_sparse = self._scale_scores(
            dense_query, sparse_query, self.alpha
        )

        results = self.pc_index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=self.initial_top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        candidates = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            metadata["hybrid_score"] = round(match.get("score", 0), 4)
            candidates.append(Document(page_content=text, metadata=metadata))

        print(f"[INFO] Hybrid search returned {len(candidates)} candidates.")

        # ── Step 2: Cohere Re-ranking ─────────────────────────────────
        reranked = self.reranker.rerank(query=query, documents=candidates)

        return reranked

    def __call__(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


# -----------------------------------------
# Hybrid RAG Chain (with Re-ranking)
# -----------------------------------------

def get_rag_chain(
    namespace: str,
    alpha: float = 0.5,
    initial_top_k: int = 10,
    final_top_n: int = 3,
    rerank_model: str = "rerank-english-v3.0",
):
    """
    Build a full hybrid RAG chain:
      BM25 + Vector retrieval  →  Cohere re-ranking  →  GPT-4o-mini answer

    Parameters
    ----------
    namespace      : Pinecone namespace from ingest_pdf()
    alpha          : 0=keyword-only, 1=vector-only, 0.5=balanced hybrid
    initial_top_k  : candidates fetched from Pinecone before re-ranking
    final_top_n    : documents kept after re-ranking fed into LLM prompt
    rerank_model   : Cohere re-rank model name
    """

    retriever = HybridRetriever(
        namespace=namespace,
        alpha=alpha,
        initial_top_k=initial_top_k,
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