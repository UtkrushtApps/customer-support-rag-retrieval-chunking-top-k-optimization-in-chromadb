#!/bin/bash
set -e

echo "[+] Starting vector DB..."
docker-compose up -d
sleep 2

echo "[+] Running document chunking, embedding & ingestion..."
python3 db/init_vector_db.py

echo "✔ Vector DB is ready for retrieval tasks."
