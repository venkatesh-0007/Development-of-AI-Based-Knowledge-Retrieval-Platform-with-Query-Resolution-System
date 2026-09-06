"""End-to-end ingestion pipeline."""
import hashlib
from .loaders import extract_text
from .cleaner import clean_text
from .chunker import chunk_text

class IngestionPipeline:
    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def ingest(self, file_bytes, filename, chunk_size=1000, overlap=150):
        raw = extract_text(file_bytes, filename)
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned, chunk_size, overlap)
        embeddings = self.embedder.encode(chunks)

        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        ids = [f"{file_hash}-{i}" for i in range(len(chunks))]
        metadatas = [{
            "source": filename,
            "chunk_index": i,
            "file_hash": file_hash
        } for i in range(len(chunks))]

        if chunks:
            self.vectorstore.add(chunks, embeddings, metadatas, ids)

        return {
            "filename": filename,
            "characters": len(cleaned),
            "chunks": len(chunks),
            "file_hash": file_hash
        }
