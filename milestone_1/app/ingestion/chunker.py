"""Simple character-based chunking with overlap."""
from typing import List

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be > overlap >= 0")
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a natural boundary when possible.
        if end < len(text):
            boundary = max(text.rfind("\n", start, end),
                           text.rfind(". ", start, end),
                           text.rfind(" ", start, end))
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + (1 if text[boundary] == "\n" else 2 if text[boundary:boundary+2] == ". " else 1)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
