"""All settings come from .env (loaded once at import)."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(v, default=False):
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    APP_NAME = os.getenv("APP_NAME", "Hybrid RAG Assistant")
    APP_TAGLINE = os.getenv("APP_TAGLINE", "Ask questions about your documents")
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "5050"))
    DEBUG = _bool(os.getenv("DEBUG"))
    SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
    ALLOWED_EXTENSIONS = {"txt", "md", "csv", "pdf"}

    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
    AWS_PROFILE = os.getenv("AWS_PROFILE") or None
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID") or None
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") or None
    AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN") or None
    BEDROCK_CHAT_MODEL_ID = os.getenv("BEDROCK_CHAT_MODEL_ID", "amazon.nova-pro-v1:0")
    EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
    EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1024"))
    EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

    CHROMA_DIR = str((BASE_DIR / os.getenv("CHROMA_DIR", "./data/chroma")).resolve())
    CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "documents")

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K = int(os.getenv("TOP_K", "5"))
    HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))
    HYBRID_KEYWORD_WEIGHT = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4"))
    RRF_K = int(os.getenv("RRF_K", "60"))

    SYSTEM_PROMPT = os.getenv(
        "SYSTEM_PROMPT",
        "Answer only from the numbered context, cite sources like [1], and say so if the answer is not in the context.",
    )


settings = Settings()
