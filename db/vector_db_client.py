import chromadb
from chromadb.config import Settings
from config import CHROMA_COLLECTION_NAME

def get_chroma_client():
    client = chromadb.Client(Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory="./chroma_data"
    ))
    return client

def get_collection():
    client = get_chroma_client()
    collection = client.get_collection(CHROMA_COLLECTION_NAME)
    return collection
