from db.vector_db_client import get_collection
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
collection = get_collection()

def retrieve_top_k_chunks(query: str, k: int = 5):
    """
    Retrieve top-k most relevant support article chunks (with metadata) for a query.
    Args:
        query: The user's support question.
        k: Number of chunks to retrieve.
    Returns:
        List of dicts, each with 'text', 'score', and 'metadata'
    """
    # TODO: Complete efficient cosine similarity top-k retrieval using Chroma and query embedding
    pass
