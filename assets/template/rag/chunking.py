"""Paragraph-aware chunking with character overlap."""
import re


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        while len(p) > size:                      # hard-split very long paragraphs
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if buf:
                chunks.append(buf); buf = ""
            chunks.append(p[:cut].strip())
            p = p[max(cut - overlap, 1):].strip() if len(p) > cut else ""
        if not p:
            continue
        if buf and len(buf) + len(p) + 2 > size:
            chunks.append(buf)
            buf = (buf[-overlap:] + "\n\n" + p) if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [c for c in (c.strip() for c in chunks) if c]
