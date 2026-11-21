# Hybrid RAG Assistant

Flask + ChromaDB + AWS Bedrock (Amazon Nova Pro, Titan Text Embeddings V2). Hybrid retrieval = dense vectors + BM25, fused with Reciprocal Rank Fusion.

```bash
cp .env.example .env        # then set AWS_REGION and credentials/profile
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py               # http://localhost:5050
```

Enable model access for Nova Pro and Titan Text Embeddings V2 in the Bedrock console first. Docker: `docker build -t rag . && docker run -p 5050:5050 --env-file .env rag`.
