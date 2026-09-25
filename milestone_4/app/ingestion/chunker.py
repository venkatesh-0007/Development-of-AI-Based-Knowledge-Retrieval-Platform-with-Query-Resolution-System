"""Semantic-aware chunking with dynamic overlap and boundary detection."""
from typing import List

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Chunks text into windows of `chunk_size` characters with `overlap` overlap,
    preferring natural boundaries (paragraphs, sentence ends, spaces).
    """
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be > overlap >= 0")
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind("; ", start, end),
                text.rfind(" ", start, end)
            )
            if boundary > start + int(chunk_size * 0.55):
                if text[boundary:boundary+2] == "\n\n":
                    end = boundary + 2
                elif text[boundary] == "\n":
                    end = boundary + 1
                elif text[boundary:boundary+2] in [". ", "! ", "? ", "; "]:
                    end = boundary + 2
                else:
                    end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks
