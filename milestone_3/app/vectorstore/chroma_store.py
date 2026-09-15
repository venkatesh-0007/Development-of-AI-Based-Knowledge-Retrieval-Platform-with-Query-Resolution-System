"""ChromaDB persistence and semantic search with in-memory cosine fallback."""
import math
from typing import List, Dict, Any, Optional

class ChromaStore:
    def __init__(self, path: str = "data/chroma", collection_name: str = "knowledge_base"):
        self.path = path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Fallback in-memory storage
        self._fallback_docs = []
        self._fallback_embeddings = []
        self._fallback_metadatas = []
        self._fallback_ids = []

        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except (ImportError, Exception):
            self.client = None
            self.collection = None

    def add(self, chunks: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]], ids: List[str]):
        if self.collection is not None:
            self.collection.upsert(
                ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
            )
            return

        # Fallback storage
        for c, emb, m, i in zip(chunks, embeddings, metadatas, ids):
            if i in self._fallback_ids:
                idx = self._fallback_ids.index(i)
                self._fallback_docs[idx] = c
                self._fallback_embeddings[idx] = emb
                self._fallback_metadatas[idx] = m
            else:
                self._fallback_ids.append(i)
                self._fallback_docs.append(c)
                self._fallback_embeddings.append(emb)
                self._fallback_metadatas.append(m)

    def search(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.collection is not None:
            n = min(top_k, self.collection.count())
            if n == 0:
                return []
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"]
            )
            rows = []
            if result.get("documents") and result["documents"][0]:
                for i, doc in enumerate(result["documents"][0]):
                    dist = result["distances"][0][i]
                    rows.append({
                        "content": doc,
                        "metadata": result["metadatas"][0][i],
                        "distance": dist,
                        "score": round(1 - dist, 4),
                    })
            return rows

        # Fallback cosine search with soft query-coverage scaling
        if not self._fallback_docs:
            return []

        scores = []
        for i, doc_emb in enumerate(self._fallback_embeddings):
            dot = sum(a * b for a, b in zip(embedding, doc_emb))
            norm_q = math.sqrt(sum(a * a for a in embedding))
            norm_d = math.sqrt(sum(b * b for b in doc_emb))
            raw_cosine = dot / (norm_q * norm_d) if (norm_q > 0 and norm_d > 0) else 0.0
            
            # Map positive alignment to 0.0 - 1.0 relevance scale
            scaled_score = max(0.0, min(1.0, raw_cosine * 2.5)) if raw_cosine > 0.01 else 0.0
            dist = max(0.0, 1.0 - scaled_score)
            scores.append((scaled_score, dist, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        top_items = scores[:top_k]

        rows = []
        for sim, dist, idx in top_items:
            rows.append({
                "content": self._fallback_docs[idx],
                "metadata": self._fallback_metadatas[idx],
                "distance": dist,
                "score": round(sim, 4)
            })
        return rows

    def count(self) -> int:
        if self.collection is not None:
            return self.collection.count()
        return len(self._fallback_docs)
