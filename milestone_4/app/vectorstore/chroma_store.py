"""ChromaDB persistence and semantic search with in-memory cosine fallback."""
import math
from typing import List, Dict, Any, Optional

class ChromaStore:
    """Manages persistent vector storage with ChromaDB, providing an in-memory cosine fallback."""

    def __init__(self, path: str = "data/chroma", collection_name: str = "knowledge_base_m4"):
        self.path = path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        # Fallback in-memory storage
        self._fallback_docs: List[str] = []
        self._fallback_embeddings: List[List[float]] = []
        self._fallback_metadatas: List[Dict[str, Any]] = []
        self._fallback_ids: List[str] = []

        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception:
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
                for doc, meta, dist in zip(
                    result["documents"][0],
                    result["metadatas"][0],
                    result["distances"][0]
                ):
                    similarity = max(0.0, min(1.0, 1.0 - float(dist)))
                    rows.append({
                        "document": doc,
                        "metadata": meta,
                        "score": round(similarity, 4),
                        "distance": dist
                    })
            return rows

        # Fallback cosine search
        if not self._fallback_embeddings:
            return []
        scores = []
        for i, emb in enumerate(self._fallback_embeddings):
            dot = sum(a * b for a, b in zip(embedding, emb))
            norm_q = math.sqrt(sum(a * a for a in embedding))
            norm_e = math.sqrt(sum(b * b for b in emb))
            sim = dot / (norm_q * norm_e) if (norm_q > 0 and norm_e > 0) else 0.0
            scores.append((sim, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]
        return [
            {
                "document": self._fallback_docs[i],
                "metadata": self._fallback_metadatas[i],
                "score": round(max(0.0, min(1.0, sim)), 4),
                "distance": 1.0 - sim
            }
            for sim, i in top
        ]

    def count(self) -> int:
        if self.collection is not None:
            try:
                return self.collection.count()
            except Exception:
                pass
        return len(self._fallback_ids)

    def clear(self):
        if self.collection is not None:
            try:
                self.client.delete_collection(self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                pass
        self._fallback_docs.clear()
        self._fallback_embeddings.clear()
        self._fallback_metadatas.clear()
        self._fallback_ids.clear()
