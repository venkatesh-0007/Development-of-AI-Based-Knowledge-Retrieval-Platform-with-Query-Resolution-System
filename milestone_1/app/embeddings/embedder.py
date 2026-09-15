"""Embedding model wrapper using SentenceTransformers with lightweight fallback."""
import re
import math
import hashlib
from typing import List

DEFAULT_MODEL = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            self.model = None

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        
        # Lightweight fallback deterministic character/word n-gram vectorizer
        vectors = []
        dim = 384
        for text in texts:
            tokens = re.findall(r"\w+", text.lower())
            vec = [0.0] * dim
            
            # Word tokens
            for token in tokens:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign * 2.0

            # Character 3-grams for subword similarity
            clean_str = " ".join(tokens)
            for i in range(len(clean_str) - 2):
                gram = clean_str[i:i+3]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign * 0.5

            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vectors.append(vec)
        return vectors
