---
name: flask-hybrid-rag-app
description: Build a lightweight Python Flask web application with a clean light theme and a hybrid RAG (dense + BM25 keyword, fused with Reciprocal Rank Fusion) pipeline on ChromaDB, using AWS Bedrock Amazon Nova Pro for generation and Amazon Titan Text Embeddings V2 for embeddings, all configured through a .env file and served on port 5050. Use this skill whenever the user wants a Flask RAG app, document Q&A or knowledge-base chat, a "chat with your documents" prototype, a ChromaDB + Bedrock demo, a Nova Pro / Titan embeddings web app, a pre-sales or client-facing RAG prototype, or says "hybrid search", "Chroma", "Bedrock RAG", or "lightweight Flask app", even if they do not name every component. Also use it to extend or re-theme an app that was already generated from this skill.
---

# Flask Hybrid RAG App

Generates a small, production-shaped Flask application: upload documents, index them in ChromaDB with Titan embeddings, search with **hybrid retrieval** (dense vectors + BM25, fused by RRF), and answer with **Nova Pro** through the Bedrock Converse API, with source citations. Light theme, `.env` driven, runs on **port 5050**.

## Fast path

1. Scaffold into the target folder (this copies the tested template and creates `.env` from `.env.example`):
   ```bash
   python /mnt/skills/user/flask-hybrid-rag-app/scripts/scaffold.py --dest ./my-rag-app --name "My RAG App"
   ```
   (Use the real skill path if it differs; the script lives next to this file.)
2. Customise (see "Customisation points" below): app name, accent colour, system prompt, sample documents.
3. Tell the user to fill in AWS settings in `.env`, then run:
   ```bash
   cd my-rag-app
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python app.py                                        # http://localhost:5050
   ```
4. Verify: `GET /api/health` returns `{"status":"ok", ...}`, upload a `.txt`/`.md`/`.pdf`, ask a question, confirm citations appear.

Deliver the finished project as a zip (and/or present the folder) rather than pasting every file into chat.

## What the template contains

```
app.py                 Flask routes, JSON errors, port from .env (default 5050)
config.py              All settings loaded from .env via python-dotenv
rag/bedrock.py         Titan embeddings (InvokeModel) + Nova Pro (Converse API)
rag/store.py           ChromaDB persistent store + in-memory BM25 + RRF fusion
rag/chunking.py        Paragraph-aware chunker with overlap
rag/loaders.py         txt / md / csv / pdf text extraction
rag/pipeline.py        retrieve -> build grounded prompt -> generate -> sources
templates/ static/     Single-page light-theme UI (vanilla JS, no build step)
.env.example           Every configurable value, documented
Dockerfile             Optional container, exposes 5050
```

## Hard requirements (do not change unless the user asks)

- **Port 5050**, read from `PORT` in `.env`; `.env.example` ships with `PORT=5050`.
- **All configuration in `.env`**. Never hard-code credentials, model IDs, paths or weights in Python. New settings get added to both `config.py` and `.env.example`.
- **Models**: generation `amazon.nova-pro-v1:0`, embeddings `amazon.titan-embed-text-v2:0` (1024 dims, normalized). Both are overridable in `.env`.
- **Hybrid retrieval** means both signals are always available: dense (Chroma, cosine) and keyword (BM25), fused with weighted Reciprocal Rank Fusion. The UI lets the user switch between `hybrid`, `dense` and `keyword`.
- **Light theme only**: white/near-white surfaces, dark slate text, one accent colour via CSS variables in `static/style.css`.
- **Grounded answers**: the system prompt tells Nova Pro to answer only from retrieved context, cite `[n]`, and say so when the context lacks the answer.
- **Credentials**: prefer the default AWS credential chain (profile, SSO, instance role). `.env` keys are optional and only for local use. Never echo secrets back to the user or commit `.env`.

## Customisation points

| Want to change | Where |
|---|---|
| App title / tagline | `APP_NAME`, `APP_TAGLINE` in `.env` |
| Accent colour, fonts | CSS variables at top of `static/style.css` |
| Assistant behaviour / domain wording | `SYSTEM_PROMPT` in `.env` |
| Chunk size / overlap | `CHUNK_SIZE`, `CHUNK_OVERLAP` |
| Dense vs keyword balance | `HYBRID_DENSE_WEIGHT`, `HYBRID_KEYWORD_WEIGHT`, `RRF_K` |
| Retrieval depth | `TOP_K` |
| Model IDs / region | `BEDROCK_CHAT_MODEL_ID`, `EMBED_MODEL_ID`, `AWS_REGION` |
| New file types | add an extractor in `rag/loaders.py` and its extension to `ALLOWED_EXTENSIONS` |
| Chroma location / collection | `CHROMA_DIR`, `CHROMA_COLLECTION` |

For domain re-theming (for example a media, broadcast or retail look), change the app name, accent colour, system prompt and sample questions; keep the architecture intact.

## Bedrock gotchas (tell the user when relevant)

- **Model access must be enabled** in the Bedrock console for Nova Pro and Titan Text Embeddings V2 in the chosen region, otherwise calls fail with `AccessDeniedException`.
- If Nova Pro rejects on-demand invocation in the region, set `BEDROCK_CHAT_MODEL_ID` to a cross-region inference profile such as `us.amazon.nova-pro-v1:0` (or `apac.` / `eu.` equivalent).
- Changing `EMBED_MODEL_ID` or `EMBED_DIMENSIONS` after documents are indexed requires a fresh Chroma folder, because vector sizes must match. Delete `CHROMA_DIR` and re-upload.
- Throttling is retried by boto3 (`standard` mode); large uploads embed chunks with a small thread pool (`EMBED_WORKERS`).

## Extending safely

Read `references/rag-design.md` before changing retrieval. Common extensions: reranking step after RRF, metadata filters (`where=` in Chroma), streaming answers (`converse_stream`), basic auth/login, S3 as the document source. Keep each addition behind a `.env` flag so the base app stays lightweight.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `AccessDeniedException` | Model access not enabled, or IAM missing `bedrock:InvokeModel` |
| `ValidationException` on chat | Use an inference-profile ID for Nova Pro in that region |
| `NoCredentialsError` | Set a profile/role, or the three `AWS_*` values in `.env` |
| Empty answers, "no context" | No documents uploaded, or scanned PDF with no extractable text |
| Port already in use | Another process on 5050; change `PORT` in `.env` |
| Dimension mismatch error | Embedding settings changed; wipe `CHROMA_DIR` |
