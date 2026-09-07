"""ChromaDB persistence and semantic search."""
import chromadb

class ChromaStore:
    def __init__(self, path="data/chroma", collection_name="knowledge_base"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks, embeddings, metadatas, ids):
        self.collection.upsert(
            ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
        )

    def search(self, embedding, top_k=5):
        n = min(top_k, self.collection.count())
        if n == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"]
        )
        rows = []
        for i, doc in enumerate(result["documents"][0]):
            rows.append({
                "content": doc,
                "metadata": result["metadatas"][0][i],
                "distance": result["distances"][0][i],
                "score": 1 - result["distances"][0][i],
            })
        return rows

    def count(self):
        return self.collection.count()
