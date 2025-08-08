import os
import sys
import time
from sentence_transformers import SentenceTransformer
from config import (CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP)
from db.vector_db_client import get_chroma_client
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

DATA_FILE = "data/full_doc.txt"

def parse_article_line(line):
    parts = line.strip().split('|', 3)
    if len(parts) != 4:
        return None
    return {
        'category': parts[0],
        'priority': int({'Low': 1,'Medium': 2,'High': 3}.get(parts[1], 0)),
        'priority_str': parts[1],
        'date': parts[2],
        'content': parts[3]
    }

def chunk_text(text, chunk_size=200, chunk_overlap=50):
    tokens = word_tokenize(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_txt = " ".join(chunk_tokens)
        chunks.append((chunk_txt, idx, start, end))
        idx += 1
        start += chunk_size - chunk_overlap
    return chunks

def main():
    print("[init_vector_db] Loading support articles...")
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Document file not found at {DATA_FILE}")
        sys.exit(1)
    with open(DATA_FILE, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    articles = [parse_article_line(l) for l in lines]
    articles = [a for a in articles if a]
    print(f"[init_vector_db] Parsed {len(articles)} articles.")

    print("[init_vector_db] Chunking and preparing metadata...")
    chunk_texts = []
    metadatas = []
    for art_idx, art in enumerate(articles):
        chunks = chunk_text(art['content'], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk, c_idx, c_start, c_end in chunks:
            meta = {
                'category': art['category'],
                'priority': art['priority'],
                'priority_str': art['priority_str'],
                'date': art['date'],
                'article_idx': art_idx,
                'chunk_idx': c_idx,
                'char_range': f"{c_start}-{c_end}"
            }
            metadatas.append(meta)
            chunk_texts.append(chunk)
    print(f"[init_vector_db] Total chunks prepared: {len(chunk_texts)}")

    print("[init_vector_db] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(chunk_texts), batch_size):
        batch_texts = chunk_texts[i:i+batch_size]
        all_embeddings.extend(model.encode(batch_texts, convert_to_numpy=True))
    print(f"[init_vector_db] Embeddings computed for {len(all_embeddings)} chunks.")

    client = get_chroma_client()
    existing_collections = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing_collections:
        print(f"[init_vector_db] Removing existing collection '{CHROMA_COLLECTION_NAME}'...")
        client.delete_collection(CHROMA_COLLECTION_NAME)
    collection = client.create_collection(CHROMA_COLLECTION_NAME)
    ids = [f"article{m['article_idx']}_chunk{m['chunk_idx']}" for m in metadatas]
    print("[init_vector_db] Inserting all vectors and metadata into Chroma...")
    collection.add(
        embeddings=all_embeddings,
        documents=chunk_texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"[init_vector_db] Total inserted in '{CHROMA_COLLECTION_NAME}': {len(chunk_texts)}")
    stats = collection.count()
    if stats != len(chunk_texts):
        print("[ERROR] Vector count mismatch! Initialization failed.")
        sys.exit(2)
    print("[init_vector_db] Vector DB ready.")
    sys.exit(0)

if __name__ == '__main__':
    main()
