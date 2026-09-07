"""Semantic retrieval pipeline."""
class RetrievalPipeline:
    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.encode([query])[0]
        return self.vectorstore.search(query_embedding, top_k=top_k)
