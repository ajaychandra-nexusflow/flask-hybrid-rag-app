"""AWS Bedrock: Titan embeddings (InvokeModel) and Nova Pro (Converse)."""
import json
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

from config import settings

_client = None


def client():
    global _client
    if _client is None:
        cfg = Config(retries={"max_attempts": 6, "mode": "standard"}, read_timeout=120)
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            session = boto3.Session(
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                aws_session_token=settings.AWS_SESSION_TOKEN,
                region_name=settings.AWS_REGION,
            )
        else:
            session = boto3.Session(profile_name=settings.AWS_PROFILE,
                                    region_name=settings.AWS_REGION)
        _client = session.client("bedrock-runtime", config=cfg)
    return _client


def embed(text: str) -> list[float]:
    body = json.dumps({
        "inputText": text[:20000],
        "dimensions": settings.EMBED_DIMENSIONS,
        "normalize": True,
    })
    resp = client().invoke_model(
        modelId=settings.EMBED_MODEL_ID, body=body,
        contentType="application/json", accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def embed_many(texts: list[str]) -> list[list[float]]:
    with ThreadPoolExecutor(max_workers=max(1, settings.EMBED_WORKERS)) as pool:
        return list(pool.map(embed, texts))


def chat(system: str, messages: list[dict]) -> str:
    """messages: [{'role': 'user'|'assistant', 'content': '...'}]"""
    converse_msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]
    resp = client().converse(
        modelId=settings.BEDROCK_CHAT_MODEL_ID,
        system=[{"text": system}],
        messages=converse_msgs,
        inferenceConfig={"maxTokens": settings.MAX_TOKENS,
                         "temperature": settings.TEMPERATURE, "topP": 0.9},
    )
    parts = resp["output"]["message"]["content"]
    return "".join(p.get("text", "") for p in parts).strip()
