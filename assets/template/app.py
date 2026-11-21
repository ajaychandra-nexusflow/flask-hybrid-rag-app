"""Flask hybrid RAG app: ChromaDB + AWS Bedrock (Nova Pro, Titan embeddings)."""
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

from config import settings
from rag import pipeline
from rag.chunking import chunk_text
from rag.loaders import extract_text
from rag.store import get_store

app = Flask(__name__)
app.config["SECRET_KEY"] = settings.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = settings.MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def index():
    return render_template("index.html", app_name=settings.APP_NAME,
                           tagline=settings.APP_TAGLINE, top_k=settings.TOP_K)


@app.get("/api/health")
def health():
    return jsonify(status="ok", app=settings.APP_NAME, chunks=get_store().count(),
                   chat_model=settings.BEDROCK_CHAT_MODEL_ID,
                   embed_model=settings.EMBED_MODEL_ID, region=settings.AWS_REGION)


@app.get("/api/documents")
def documents():
    return jsonify(documents=get_store().list_documents())


@app.post("/api/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return jsonify(error="No files received."), 400
    added, skipped = [], []
    for f in files:
        name = secure_filename(f.filename or "")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in settings.ALLOWED_EXTENSIONS:
            skipped.append({"filename": name or "(unnamed)", "reason": "unsupported type"})
            continue
        text = extract_text(name, f.read())
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            skipped.append({"filename": name, "reason": "no extractable text"})
            continue
        added.append(get_store().add_document(name, chunks))
    return jsonify(added=added, skipped=skipped)


@app.delete("/api/documents/<doc_id>")
def delete_document(doc_id):
    get_store().delete_document(doc_id)
    return jsonify(deleted=doc_id)


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify(error="Question is required."), 400
    mode = data.get("mode", "hybrid")
    if mode not in {"hybrid", "dense", "keyword"}:
        mode = "hybrid"
    top_k = max(1, min(int(data.get("top_k") or settings.TOP_K), 20))
    return jsonify(pipeline.answer(question, mode, top_k, data.get("history")))


@app.errorhandler(Exception)
def on_error(e):
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    app.logger.exception("Unhandled error")
    return jsonify(error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG, use_reloader=False)
