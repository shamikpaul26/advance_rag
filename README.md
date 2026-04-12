# Advance RAG Chatbot

> **Technical Documentation** — Version 1.0
>
> Retrieval Modes: Hybrid BM25+Vector · RAG Fusion  
> Components: `rag_hybrid.py` · `app.py`  
> Features: Re-ranking · LongContextReorder · Streamlit UI

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Retrieval Modes](#3-retrieval-modes)
4. [Module Reference — rag_hybrid.py](#4-module-reference--rag_hybridpy)
5. [Module Reference — app.py](#5-module-reference--apppy)
6. [Configuration & Dependencies](#6-configuration--dependencies)
7. [Data Flow Details](#7-data-flow-details)
8. [Parameter Tuning Guide](#8-parameter-tuning-guide)
9. [Known Gotchas & Implementation Notes](#9-known-gotchas--implementation-notes)
10. [File Structure](#10-file-structure)

---

## 1. Overview

The Hybrid RAG Chatbot is a production-quality Retrieval-Augmented Generation system designed to answer questions over PDF documents. It integrates **Pinecone** for hybrid vector search, **Cohere** for cross-encoder re-ranking, and **OpenAI GPT-4o-mini** as the answer model. The system is served through an interactive Streamlit web interface with real-time parameter tuning.


The project is structured around two core files:

- `rag_hybrid.py` — all retrieval logic: ingestion, encoding, search, re-ranking, and LangChain chain assembly
- `app.py` — Streamlit UI: file upload, session management, sidebar controls, and chat interface

### System Highlights

| Capability | Description |
|---|---|
| **Dual retrieval modes** | Hybrid (BM25 + dense vector) and RAG Fusion (multi-query + RRF) |
| **Cohere re-ranking** | Cross-encoder relevance scoring for precision |
| **LongContextReorder** | Mitigates LLM lost-in-the-middle bias |
| **Live parameter tuning** | Adjust alpha, top-k, re-rank model, etc. via sidebar |
| **Auto-detection** | Detects and reuses previously indexed PDFs |

---

## 2. Architecture


                ┌────────────────────┐
                │    User Query      │
                └─────────┬──────────┘
                          │
          ┌───────────────▼────────────────┐
          │        Query Encoding          │
          │  Dense (Embeddings) + BM25     │
          └───────────────┬────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │        Retrieval Layer            │
        │  ┌────────────┬───────────────┐   │
        │  │  Hybrid    │  RAG Fusion   │   │
        │  │ (1 Query)  │ (Multi Query) │   │
        │  └─────┬──────┴───────┬───────┘   │
        └────────┼──────────────┼───────────┘
                 │              │
                 │        ┌─────▼──────┐
                 │        │   RRF      │
                 │        │  Fusion    │
                 │        └─────┬──────┘
                 │              │
                 └──────┬───────┘
                        ▼
              ┌───────────────────┐
              │   Pinecone Search │
              └─────────┬─────────┘
                        ▼
              ┌───────────────────┐
              │  Re-ranking Layer │
              │ (Cohere Cross-Enc)│
              └─────────┬─────────┘
                        ▼
          ┌────────────────────────────┐
          │ LongContextReorder         │
          │ (Fix attention bias)       │
          └─────────┬──────────────────┘
                    ▼
          ┌────────────────────────────┐
          │     GPT-4o-mini LLM        │
          │   (Final Answer Gen)       │
          └─────────┬──────────────────┘
                    ▼
              ┌──────────────┐
              │   Response   │
              └──────────────┘


### 2.1 High-Level Architecture Diagram

Both retrieval modes share the same encoding, re-ranking, and answer-generation layers — only the retrieval step differs.

| Pipeline Stage | Details |
|---|---|
| **① User Query** | Plain-text question entered in Streamlit chat UI |
| **② Query Encoding** | Dense: OpenAI `text-embedding-3-small` (1536-dim) / Sparse: BM25Encoder (fitted on PDF corpus) |
| **③ Retrieval Mode** | **HYBRID:** Single Pinecone query, alpha-blended / **FUSION:** N sub-queries via GPT-4o-mini → RRF merge |
| **④ Pinecone Search** | Serverless index (AWS us-east-1), dotproduct metric — required for sparse+dense hybrid |
| **⑤ RRF Fusion *(Fusion only)*** | `score(d) = Σ 1/(k+rank)` across all sub-query lists. Produces single unified ranked document list |
| **⑥ Cohere Re-ranking** | Cross-encoder relevance scoring. Models: `rerank-english-v3.0` / `multilingual-v3.0` |
| **⑦ LongContextReorder** | Best docs placed at start & end of context. Reduces LLM lost-in-the-middle attention bias |
| **⑧ GPT-4o-mini** | Synthesises final answer from reordered context. Temperature = 0, grounded strictly to provided docs |

### 2.2 Component Interaction

| Component | Technology | Role |
|---|---|---|
| **UI Layer** | Streamlit | File upload, parameter sidebar, chat interface, session state |
| **Dense Embedding** | OpenAI `text-embedding-3-small` | 1536-dim vectors for semantic similarity |
| **Sparse Encoding** | BM25Encoder (`pinecone-text`) | TF-IDF-style term-frequency sparse vectors |
| **Vector Store** | Pinecone (Serverless, dotproduct) | Stores hybrid records, handles weighted retrieval |
| **Query Expansion** | GPT-4o-mini (Fusion mode) | Generates N rephrasings of the original query |
| **RRF Fusion** | Custom Python (`_reciprocal_rank_fusion`) | Merges ranked document lists across sub-queries |
| **Re-ranker** | Cohere API (cross-encoder) | Rescores top candidates with full query+doc context |
| **Context Reorder** | LangChain LongContextReorder | Places best docs at context start and end |
| **Answer LLM** | GPT-4o-mini (temperature=0) | Grounded answer generation from reordered context |

---

## 3. Retrieval Modes

### 3.1 Hybrid Mode (BM25 + Dense Vector)

In Hybrid mode, a single query is encoded in parallel by both the BM25 sparse encoder and the OpenAI dense embedder. The two vectors are alpha-blended and submitted as a single Pinecone hybrid query.

**Pipeline:**
```
Query → BM25 sparse encode + OpenAI dense embed
      → Scale vectors by alpha / (1 - alpha)
      → Pinecone hybrid query (initial_top_k candidates)
      → Cohere re-rank → top final_top_n docs
      → LongContextReorder → GPT-4o-mini
```

**Alpha blending** controls the BM25-vs-dense trade-off:

| Alpha | Behaviour |
|---|---|
| `0.0` | Pure keyword (BM25 only) |
| `0.5` | Balanced hybrid (default) |
| `1.0` | Pure semantic (dense only) |

### 3.2 RAG Fusion Mode (Multi-query + RRF)

RAG Fusion addresses vocabulary mismatch by rephrasing the original query N times using GPT-4o-mini. Each rephrasing is independently searched via hybrid retrieval. All result lists are merged using Reciprocal Rank Fusion (RRF) before re-ranking.

**Pipeline:**
```
Query → GPT-4o-mini → N sub-queries
      → For each sub-query: Hybrid Search (top_k_per_q)
      → Reciprocal Rank Fusion (RRF constant k=60)
      → Cohere re-rank → top final_top_n docs
      → LongContextReorder → GPT-4o-mini
```

**RRF scoring formula:**

```
score(d) = Σ 1 / (k + rank_i(d))   for all sub-query lists containing doc d
```

Where `k=60` is the standard dampening constant.

### 3.3 Mode Comparison

| Property | ⚡ Hybrid | 🔀 RAG Fusion |
|---|---|---|
| **LLM calls** | 0 (for retrieval) | 1 per query (sub-query gen) |
| **Pinecone calls** | 1 | N (one per sub-query) |
| **Recall** | Good | Higher (multi-angle search) |
| **Speed** | Fast | Slower (N×LLM + N×Pinecone) |
| **Best for** | Single, well-formed queries | Vague or ambiguous questions |

---

## 4. Module Reference — `rag_hybrid.py`

### 4.1 `generate_namespace(file_path)`

Generates a deterministic MD5 hash from the file path string. Used as the Pinecone namespace to isolate each unique file.

```python
generate_namespace(file_path: str) -> str
```

| Parameter | Type | Description |
|---|---|---|
| `file_path` | `str` | Absolute or relative path to the PDF file |

**Returns:** 32-character hexadecimal MD5 string (e.g., `'a3f1d9c2e4b7...'`).

### 4.2 `create_index_if_not_exists()`

Checks whether the configured Pinecone index exists and creates it if not. Uses `dotproduct` metric and AWS `us-east-1` serverless spec.

> ⚠️ **Critical:** Hybrid search requires `metric='dotproduct'`. Using `'cosine'` will silently produce incorrect results because the alpha-scaled sparse values rely on dot product.

### 4.3 `ingest_pdf(file_path)`

End-to-end PDF ingestion pipeline. Loads the PDF, splits into overlapping chunks, fits and saves the BM25 encoder, generates dense embeddings, and upserts hybrid vectors to Pinecone in batches of 100.

```python
ingest_pdf(file_path: str) -> str
```

Each upserted vector record contains:
- `values` — 1536-dim OpenAI dense embedding
- `sparse_values` — BM25 term-frequency sparse vector (indices + values)
- `metadata.text` — raw chunk text
- `metadata.source` — original file path
- `metadata.chunk_index` — integer position in the chunk sequence

| Setting | Value |
|---|---|
| Splitter | `RecursiveCharacterTextSplitter` |
| `chunk_size` | 1000 characters |
| `chunk_overlap` | 200 characters |
| Embedding model | `text-embedding-3-small` (1536 dims) |
| Upsert batch size | 100 records per request |
| BM25 params saved to | `bm25_params.json` |

### 4.4 `_HybridSearchEngine` (internal class)

Internal class shared by both retrieval modes. Accepts a query string, produces a weighted dense+sparse query vector pair, and issues a single Pinecone hybrid query.

```python
search(query: str, top_k: int) -> list[Document]
```

**Implementation notes:**
- Performs a defensive `dict` copy before extracting `'text'` from Pinecone metadata to avoid corrupting the shared metadata object.
- Empty chunks are filtered out to prevent the LLM from receiving blank context.
- Each returned `Document` carries a `hybrid_score` metadata field with the raw Pinecone dot-product score.

### 4.5 `RAGFusion`

Implements the RAG Fusion strategy. Invokes GPT-4o-mini to produce N rephrased sub-queries, performs a hybrid search for each, then merges using Reciprocal Rank Fusion.

| Parameter | Type / Default | Description |
|---|---|---|
| `search_engine` | `_HybridSearchEngine` | Shared hybrid search engine instance |
| `num_queries` | `int` / 4 | Number of sub-queries GPT-4o-mini generates |
| `top_k_per_q` | `int` / 5 | Pinecone results to fetch per sub-query |
| `rrf_k` | `int` / 60 | RRF dampening constant |
| `rrf_top_n` | `int` / 10 | Maximum documents to keep after RRF merge |

**Sub-query post-processing:**
- Strips leading numbering/bullets GPT-4o-mini commonly adds (e.g., `'1.'`, `'-'`, `'•'`)
- Deduplicates queries while preserving order
- Always prepends the original question as the first query

### 4.6 `CohereReranker`

Wraps the Cohere cross-encoder rerank API. Unlike the bi-encoder approach used in initial retrieval, a cross-encoder jointly encodes the query and each candidate document, producing more accurate relevance scores.

| Parameter | Type / Default | Description |
|---|---|---|
| `top_n` | `int` / 4 | Number of highest-scoring documents to return |
| `model` | `str` | Cohere re-rank model (default: `rerank-english-v3.0`) |

Each returned `Document` gets a `rerank_score` metadata field (0–1 float, higher is better).

**Available re-rank models:**

| Model | Notes |
|---|---|
| `rerank-english-v3.0` | Fastest, English-only, recommended default |
| `rerank-multilingual-v3.0` | Cross-lingual support |
| `rerank-english-v2.0` | Legacy model, lower accuracy |

### 4.7 `UnifiedRetriever`

Top-level retriever class that orchestrates the full pipeline. Internally instantiates `_HybridSearchEngine`, `RAGFusion`, `CohereReranker`, and LangChain's `LongContextReorder`.

| Parameter | Type / Default | Description |
|---|---|---|
| `namespace` | `str` | Pinecone namespace (from `ingest_pdf` or `check_existing_namespace`) |
| `mode` | `str` / `'hybrid'` | `'hybrid'` or `'fusion'` |
| `alpha` | `float` / 0.5 | BM25 vs dense blend ratio |
| `initial_top_k` | `int` / 10 | Candidates to fetch (hybrid mode) |
| `final_top_n` | `int` / 4 | Documents to keep after re-ranking |
| `rerank_model` | `str` | Cohere re-rank model name |
| `num_queries` | `int` / 4 | Sub-queries for fusion mode |
| `top_k_per_q` | `int` / 5 | Per-sub-query top-k for fusion mode |
| `rrf_k` | `int` / 60 | RRF constant for fusion mode |

### 4.8 `get_rag_chain()`

Factory function. Instantiates `UnifiedRetriever` and assembles a LangChain LCEL chain: `retriever → prompt template → GPT-4o-mini → StrOutputParser`.

```python
chain = get_rag_chain(namespace, mode='hybrid', ...)
answer = chain.invoke('What is the main topic of the paper?')
```

**Chain structure (LCEL):**
```
{context: retriever(q) | format_docs, question: passthrough} | prompt | llm | parser
```

---

## 5. Module Reference — `app.py`

### 5.1 Session State

Streamlit session state is initialised once at startup with the `DEFAULTS` dictionary.

| Key | Default | Description |
|---|---|---|
| `namespace` | `None` | Pinecone namespace for the loaded PDF |
| `rag_chain` | `None` | Instantiated LangChain chain object |
| `messages` | `[]` | Chat history list of `{role, content}` dicts |
| `mode` | `'hybrid'` | Active retrieval mode |
| `alpha` | `0.5` | BM25 / dense blend ratio |
| `initial_top_k` | `10` | Pinecone candidates to retrieve |
| `final_top_n` | `4` | Docs kept after re-ranking |
| `rerank_model` | `rerank-english-v3.0` | Cohere re-rank model |
| `num_queries` | `4` | Sub-queries for fusion mode |
| `top_k_per_q` | `5` | Per-sub-query Pinecone top-k |
| `rrf_k` | `60` | RRF dampening constant |

### 5.2 Auto-load Behaviour

On startup, if no chain is loaded, the app checks for `bm25_params.json` on disk and calls `check_existing_namespace()` to see if a Pinecone namespace is available. If both exist, the chain is rebuilt automatically and a toast notification is shown, allowing users to resume sessions without re-uploading their PDF.

### 5.3 Sidebar Parameter Tuning

Every sidebar widget writes to a `new_state` dict. The app compares `new_state` with the current session state. If any value changed and a namespace is loaded, `_build_chain()` is called immediately to rebuild the chain with updated parameters.

### 5.4 PDF Upload and Indexing

Upload flow:

1. User uploads a PDF via `st.file_uploader`. The file is written to the `uploads/` directory.
2. On **'Process & Index PDF'**, `create_index_if_not_exists()` is called first.
3. The expected namespace is computed and compared with any existing Pinecone namespace.
4. If the namespace exists and `bm25_params.json` is present, the existing index is reused without re-ingestion.
5. If BM25 params are missing (even if namespace exists), re-ingestion is triggered to refit the encoder.
6. On success, the chain is built and the chat interface becomes active.

---

## 6. Configuration & Dependencies

### 6.1 `config.py` Requirements

The system requires a `config.py` file (not included in VCS) exporting the following constants:

```python
PINECONE_API_KEY    = 'your-pinecone-api-key'
PINECONE_INDEX_NAME = 'your-index-name'
COHERE_API_KEY      = 'your-cohere-api-key'
```

> OpenAI credentials are read from the `OPENAI_API_KEY` environment variable by LangChain automatically.

### 6.2 Python Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pinecone` | Vector database client |
| `pinecone-text` | BM25Encoder for sparse vectors |
| `cohere` | Cross-encoder re-ranking API |
| `langchain` | LCEL chain assembly and document abstractions |
| `langchain-openai` | ChatOpenAI and OpenAIEmbeddings wrappers |
| `langchain-community` | PyPDFLoader and LongContextReorder |
| `openai` | Underlying OpenAI API client |

---

## 7. Data Flow Details

### 7.1 Ingestion Flow

| # | Step | Output |
|---|---|---|
| 1 | `PyPDFLoader.load()` | List of `Document` objects (one per PDF page) |
| 2 | `RecursiveCharacterTextSplitter` | Smaller chunks (size=1000, overlap=200) |
| 3 | `BM25Encoder.fit(corpus)` | TF-IDF model fitted to chunk vocabulary |
| 4 | `BM25Encoder.dump()` | `bm25_params.json` saved to disk |
| 5 | `OpenAIEmbeddings.embed_documents()` | List of 1536-dim float vectors |
| 6 | `BM25Encoder.encode_documents()` | List of `{indices, values}` sparse vectors |
| 7 | `Pinecone.index.upsert()` | Records written to namespace in batches of 100 |

### 7.2 Query Flow

| # | Step | Output |
|---|---|---|
| 1 | `OpenAIEmbeddings.embed_query()` | Dense query vector (1536-dim) |
| 2 | `BM25Encoder.encode_queries()` | Sparse query vector `{indices, values}` |
| 3a *(hybrid)* | `Pinecone.index.query()` | Top `initial_top_k` matches by hybrid score |
| 3b *(fusion)* | GPT-4o-mini sub-query gen | N rephrased query strings |
| 3c *(fusion)* | N × `Pinecone.index.query()` | N ranked document lists |
| 3d *(fusion)* | `_reciprocal_rank_fusion()` | Single fused ranked list |
| 4 | `CohereReranker.rerank()` | Top `final_top_n` docs with `rerank_score` |
| 5 | `LongContextReorder` | Docs reordered (best at start & end) |
| 6 | `format_docs()` | Single string context (docs joined by `\n\n`) |
| 7 | `ChatPromptTemplate` | Formatted prompt with context + question |
| 8 | `ChatOpenAI` (GPT-4o-mini) | Final answer string |

---

## 8. Parameter Tuning Guide

| Scenario | Recommended Settings | Reasoning |
|---|---|---|
| **Technical documents with jargon** | `alpha=0.3`, Hybrid | BM25 better at matching exact terms and acronyms |
| **Natural-language questions** | `alpha=0.7`, Hybrid | Dense embeddings capture semantic meaning |
| **Vague or broad queries** | Fusion mode, `num_queries=5` | Multiple rephrasings improve recall |
| **Speed-sensitive usage** | Hybrid mode, `top_k=8`, `top_n=3` | Fewer Pinecone calls, fewer re-rank requests |
| **Maximum recall** | Fusion, `top_k_per_q=8`, `num_queries=6` | Highest coverage at cost of latency |
| **Multilingual PDF** | `rerank-multilingual-v3.0` | Cross-lingual re-ranking support |
| **Long dense PDF (>100 pages)** | `chunk_overlap=300` (code change), `top_n=5` | More overlap reduces context gaps |

---

## 9. Known Gotchas & Implementation Notes

### 9.1 Pinecone Metadata Read-Only Snapshot

Newer versions of the Pinecone SDK return match metadata as a read-only dict snapshot. Calling `.pop()` on this object silently fails or mutates a shared object, leaving `page_content` as an empty string. The implementation always creates a `dict` copy before extracting the `'text'` key.

### 9.2 `dotproduct` Metric Is Mandatory

Pinecone hybrid search requires the index to be created with `metric='dotproduct'`. If the index uses `'cosine'`, the alpha-scaled sparse values will not work correctly. **The index must be recreated if the wrong metric was used.**

### 9.3 BM25 Params Must Match the Indexed Corpus

The `BM25Encoder` is fitted on the exact chunk corpus at ingest time and saved to `bm25_params.json`. If this file is deleted, or if the same Pinecone namespace is reused with a different PDF, the encoder vocabulary will be mismatched. The app detects this condition and triggers re-ingestion.

### 9.4 GPT-4o-mini Sub-query Formatting

GPT-4o-mini sometimes prepends numbering or bullet characters to generated sub-queries (e.g., `'1. How does...'`). The `_generate_queries` method strips these using a regex and deduplicates the list before searching. The original question is always included as the first query.

### 9.5 Empty Chunks

Two guards exist against empty chunks reaching the LLM:
- `_HybridSearchEngine.search()` skips matches with empty text fields
- `RAGFusion._reciprocal_rank_fusion()` skips documents with empty `page_content`

Without these guards, the LLM would receive blank context and answer *"I don't know."*

---

## 10. File Structure

```
project/
├── rag_hybrid.py        # Core RAG library
├── app.py               # Streamlit UI
├── config.py            # API keys (not in VCS)
├── bm25_params.json     # BM25 encoder params (auto-generated)
└── uploads/             # Uploaded PDFs (auto-created)
```

### Pinecone Index Schema

```
Index name:  <PINECONE_INDEX_NAME>  (from config.py)
Dimension:   1536                   (text-embedding-3-small)
Metric:      dotproduct             (required for hybrid)
Cloud:       AWS us-east-1          (Serverless spec)
Namespace:   MD5(file_path)         (per-PDF isolation)

Record schema:
  id:            '{namespace}-chunk-{i}'
  values:        [float × 1536]           # dense OpenAI embedding
  sparse_values: {indices, values}        # BM25 sparse vector
  metadata:      {text, source, chunk_index}
```

---

*End of Document*