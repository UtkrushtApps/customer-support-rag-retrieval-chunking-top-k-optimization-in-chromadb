# Customer Support RAG: Retrieval & Chunking Logic Task

## Task Overview
You are tasked with improving a customer support RAG system where support articles are embedded and stored in a Chroma vector database. Retrieval results are currently poor due to improper chunk size (too large, no overlap) and missing metadata. Your goals:
- Implement a chunking function to split source articles into 200-token chunks with 50-token overlap.
- During ingestion, attach metadata ('category', 'priority', 'date') to every chunk, and ensure this metadata is persistently stored in Chroma.
- Complete the retrieval pipeline so that given a query, the top-5 most relevant chunks (by cosine similarity) with all their metadata are fetched efficiently from ChromaDB.
- Use the provided recall@5 script and sample queries to verify and measure improvements.

## Guidance
- Focus on chunking logic, accurate metadata attachment, and correct top-k retrieval implementation only.
- The Chroma vector DB, embedding model, and ingestion pipeline are already operational.
- Retrieval must use cosine similarity and return all relevant metadata fields.
- Do not change pre-existing infrastructure or database setup scripts.
- Use the provided scripts and connection utilities – avoid writing infrastructure code.

## Database Access
- Vector DB: Chroma
- Host: `localhost`, Port: `8000` (or use Chroma Python SDK in your code)
- Collection: `support_articles`
- Metadata schema required: `category` (string), `priority` (int), `date` (yyyy-mm-dd)
- Vector dimension: 384 (sentence-transformers)
- Chunk count: ~10,000 after proper chunking (depends on source file length)

## Objectives
- Transform provided raw articles into properly chunked and embedded docs with correct metadata
- Implement accurate, efficient semantic retrieval (top-5) using cosine similarity
- Measure improvements using recall@5 on provided queries

## How to Verify
- Use the `sample_queries.txt` file for retrieval QA
- Run the provided recall@5 script to measure system improvements (results should show better relevant chunk retrieval)
- Manually inspect sample outputs to check that metadata is present and results are relevant

*Your work should focus exclusively on chunking, metadata, and retrieval code. Database setup and embedding infrastructure are pre-built and fully automated.*
