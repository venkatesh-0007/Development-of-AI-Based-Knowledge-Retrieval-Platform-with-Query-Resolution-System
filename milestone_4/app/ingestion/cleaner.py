"""Text cleaning and normalization utilities."""
import re

def clean_text(text: str) -> str:
    """Cleans null bytes, normalizes line breaks, and condenses excessive whitespace."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()
