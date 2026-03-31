import os
import hashlib
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


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
    
    Hybrid search requires:
      - metric = "dotproduct"  (NOT cosine)
      - dimension = 1536       (matches text-embedding-3-small)
    
    The index stores both dense (vector) and sparse (BM25) values,
    enabling Pinecone's built-in hybrid re-ranking via alpha blending.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="dotproduct",          # Required for hybrid search
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
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
    - If filename is provided, return its namespace if indexed.
    - Otherwise, return the first available namespace.
    """
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})

    if not namespaces:
        return None

    if filename:
        return filename if filename in namespaces else None

    return list(namespaces.keys())[0]


# -----------------------------------------
# BM25 Encoder — fit & save / load helpers
# -----------------------------------------

BM25_PARAMS_PATH = "bm25_params.json"


def _fit_and_save_bm25(corpus: list[str]) -> BM25Encoder:
    """Fit BM25 on the ingested corpus and persist params for future use."""
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
    Load, chunk, embed, and upsert a PDF into Pinecone with hybrid vectors.

    Each Pinecone record contains:
      - 'values'        → dense OpenAI embedding  (vector search)
      - 'sparse_values' → BM25 term weights        (keyword search)
      - 'metadata'      → source path + raw text   (for retrieval)

    Returns the namespace string for later querying.
    """
    namespace = generate_namespace(file_path)

    # ── 1. Load & split ──────────────────────────────────────────────
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)

    texts = [chunk.page_content for chunk in chunks]
    print(f"[INFO] Loaded {len(texts)} chunks from '{file_path}'.")

    # ── 2. Fit BM25 on this document's corpus ────────────────────────
    bm25_encoder = _fit_and_save_bm25(texts)

    # ── 3. Compute dense + sparse encodings ──────────────────────────
    dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    dense_vectors = dense_embeddings.embed_documents(texts)
    sparse_vectors = bm25_encoder.encode_documents(texts)  # list of {indices, values}

    # ── 4. Upsert into Pinecone ───────────────────────────────────────
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    batch_size = 100
    records = []

    for i, (text, dense, sparse) in enumerate(zip(texts, dense_vectors, sparse_vectors)):
        records.append({
            "id": f"{namespace}-chunk-{i}",
            "values": dense,                        # dense vector
            "sparse_values": sparse,                # BM25 sparse vector
            "metadata": {
                "text": text,
                "source": file_path,
                "chunk_index": i,
            }
        })

    # Upsert in batches to respect Pinecone's request size limits
    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"[INFO] Upserted chunks {start}–{start + len(batch) - 1}.")

    print(f"[INFO] Ingestion complete. Namespace: '{namespace}'.")
    return namespace


# -----------------------------------------
# Hybrid Retriever
# -----------------------------------------

class HybridRetriever:
    """
    Custom retriever that blends BM25 (sparse) and OpenAI (dense) vectors
    using Pinecone's native hybrid query with alpha blending.

    alpha controls the trade-off:
      alpha = 1.0  → pure dense  (semantic / vector)
      alpha = 0.0  → pure sparse (keyword / BM25)
      alpha = 0.5  → balanced hybrid  (recommended default)
    """

    def __init__(
        self,
        namespace: str,
        alpha: float = 0.5,
        top_k: int = 4,
    ):
        self.namespace = namespace
        self.alpha = alpha
        self.top_k = top_k

        self.pc_index = Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.bm25_encoder = _load_bm25()

    def _scale_scores(
        self,
        dense: list[float],
        sparse: dict,
        alpha: float
    ) -> tuple[list[float], dict]:
        """
        Scale dense and sparse vectors by alpha so their combined dot-product
        in Pinecone reflects the desired blend ratio.
        """
        scaled_dense = [v * alpha for v in dense]
        scaled_sparse = {
            "indices": sparse["indices"],
            "values":  [v * (1 - alpha) for v in sparse["values"]],
        }
        return scaled_dense, scaled_sparse

    def get_relevant_documents(self, query: str) -> list[Document]:
        """Encode the query and run a hybrid search against Pinecone."""

        # Encode query: dense + sparse
        dense_query = self.embeddings.embed_query(query)
        sparse_query = self.bm25_encoder.encode_queries(query)

        # Alpha-scale before sending to Pinecone
        scaled_dense, scaled_sparse = self._scale_scores(
            dense_query, sparse_query, self.alpha
        )

        # Pinecone hybrid query
        results = self.pc_index.query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=self.top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        # Convert to LangChain Documents
        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.pop("text", "")
            documents.append(Document(page_content=text, metadata=metadata))

        return documents

    # Make the retriever callable so RunnableLambda can wrap it
    def __call__(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)


# -----------------------------------------
# Format Documents for Prompt
# -----------------------------------------

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------------------
# Hybrid RAG Chain
# -----------------------------------------

def get_rag_chain(namespace: str, alpha: float = 0.5):
    """
    Build a hybrid RAG chain.

    Parameters
    ----------
    namespace : str
        Pinecone namespace to query (returned by ingest_pdf).
    alpha : float
        Blend ratio for hybrid search (0 = keyword-only, 1 = vector-only).
        Default 0.5 gives equal weight to BM25 and semantic search.

    Returns
    -------
    A LangChain LCEL chain that accepts a question string and returns an answer.
    """

    retriever = HybridRetriever(namespace=namespace, alpha=alpha, top_k=4)

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
            # RunnableLambda wraps the HybridRetriever so it fits the LCEL pipeline
            "context": RunnableLambda(retriever) | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain