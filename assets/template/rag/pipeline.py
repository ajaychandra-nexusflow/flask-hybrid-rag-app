"""Retrieve -> grounded prompt -> Nova Pro -> answer + sources."""
from config import settings
from rag import bedrock
from rag.store import get_store


def answer(question: str, mode: str = "hybrid", top_k: int | None = None,
           history: list[dict] | None = None) -> dict:
    hits = get_store().search(question, mode=mode, k=top_k)
    if not hits:
        return {"answer": "I could not find anything relevant. Upload documents first, or try rephrasing.",
                "sources": []}
    context = "\n\n".join(f"[{n}] ({h['filename']})\n{h['text']}" for n, h in enumerate(hits, 1))
    user_msg = f"Context passages:\n{context}\n\nQuestion: {question}"
    messages = [m for m in (history or [])[-6:] if m.get("role") in {"user", "assistant"} and m.get("content")]
    messages.append({"role": "user", "content": user_msg})
    text = bedrock.chat(settings.SYSTEM_PROMPT, messages)
    sources = [{"n": n, "filename": h["filename"], "chunk_index": h["chunk_index"],
                "score": h["score"], "snippet": h["text"][:280]} for n, h in enumerate(hits, 1)]
    return {"answer": text, "sources": sources}
