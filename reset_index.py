# reset_index.py
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME in pc.list_indexes().names():
    pc.delete_index(PINECONE_INDEX_NAME)
    print(f"✅ Deleted index '{PINECONE_INDEX_NAME}'")
else:
    print("Index not found.")