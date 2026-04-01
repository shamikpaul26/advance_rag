# advance_rag
Production ready RAG application

The Hybrid RAG Chatbot is a production-quality Retrieval-Augmented Generation system designed to answer questions over PDF documents. It integrates Pinecone for hybrid vector search, Cohere for cross-encoder re-ranking, and OpenAI GPT-4o-mini as the answer model. The system is served through an interactive Streamlit web interface with real-time parameter tuning.

System Highlights

Key capabilities:
  • Dual retrieval modes — Hybrid (BM25 + dense vector) and RAG Fusion (multi-query + RRF)
  • Cohere cross-encoder re-ranking for precision
  • LongContextReorder to mitigate LLM lost-in-the-middle bias
  • Live parameter tuning via sidebar (alpha, top-k, re-rank model, etc.)
  • Auto-detection and reuse of previously indexed PDFs

 Architecture Diagram

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

Pipeline Stage	Details

① User Query	Plain-text question entered in Streamlit chat UI
▼	
② Query Encoding	Dense: OpenAI text-embedding-3-small (1536-dim) Sparse: BM25Encoder (fitted on PDF corpus)
▼	
③ Retrieval Mode	HYBRID: Single Pinecone query, alpha-blended FUSION: N sub-queries via GPT-4o-mini → RRF merge
▼	
④ Pinecone Search	Serverless index (AWS us-east-1) dotproduct metric — required for sparse+dense hybrid
▼	
⑤ RRF Fusion [Fusion only]	score(d) = Σ 1/(k+rank)  across all sub-query lists Produces single unified ranked document list
▼	
⑥ Cohere Re-ranking	Cross-encoder relevance scoring Models: rerank-english-v3.0 / multilingual-v3.0
▼	
⑦ LongContextReorder	Best docs placed at start & end of context Reduces LLM lost-in-the-middle attention bias
▼	
⑧ GPT-4o-mini	Synthesises final answer from reordered context Temperature = 0, grounded strictly to provided docs

