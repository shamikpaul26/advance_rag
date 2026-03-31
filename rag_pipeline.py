import hashlib
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


# -----------------------------------------
# Generate Namespace
# -----------------------------------------

def generate_namespace(file_path: str):
    return hashlib.md5(file_path.encode()).hexdigest()


# -----------------------------------------
# Create Pinecone Index
# -----------------------------------------

def create_index_if_not_exists():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

# -----------------------------------------
# Check Existing Namespace
# -----------------------------------------
def check_existing_namespace(filename=None):
    """
    Check if a namespace already exists in Pinecone.
    If filename provided, return namespace if indexed.
    Otherwise return first available namespace.
    """
    from pinecone import Pinecone
    import os

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    stats = index.describe_index_stats()

    namespaces = stats.get("namespaces", {})

    if not namespaces:
        return None

    if filename:
        # Use filename-based namespace
        if filename in namespaces:
            return filename
        return None

    # Return first namespace if exists
    return list(namespaces.keys())[0]

# -----------------------------------------
# Ingest PDF
# -----------------------------------------

def ingest_pdf(file_path: str):

    namespace = generate_namespace(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["source"] = file_path

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=namespace
    )

    return namespace


# -----------------------------------------
# Format Documents for Prompt
# -----------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -----------------------------------------
# Modern LCEL RAG Chain
# -----------------------------------------

def get_rag_chain(namespace: str):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=namespace
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

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
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
