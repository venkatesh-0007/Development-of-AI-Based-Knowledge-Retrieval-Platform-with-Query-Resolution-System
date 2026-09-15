"""End-to-end ingestion pipeline with full metadata preservation."""
import hashlib
from typing import Dict, Any, List
from .loaders import extract_document_sections
from .cleaner import clean_text
from .chunker import chunk_text

class IngestionPipeline:
    def __init__(self, embedder, vectorstore):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def ingest(self, file_bytes: bytes, filename: str, chunk_size: int = 1000, overlap: int = 150) -> Dict[str, Any]:
        sections = extract_document_sections(file_bytes, filename)
        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

        all_chunks: List[str] = []
        all_metadatas: List[Dict[str, Any]] = []
        all_ids: List[str] = []
        global_chunk_index = 0

        for section in sections:
            raw_text = section.get("text", "")
            page_num = section.get("page")
            section_name = section.get("section") or "Document Body"
            
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            sec_chunks = chunk_text(cleaned, chunk_size, overlap)
            for chunk in sec_chunks:
                chunk_id = f"{file_hash}-{global_chunk_index}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                
                # ChromaDB requires primitive metadata types (int, str, float, bool)
                all_metadatas.append({
                    "document_name": filename,
                    "source": filename,
                    "page": page_num if page_num is not None else 0,
                    "section": section_name,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_index,
                    "file_hash": file_hash
                })
                global_chunk_index += 1

        if all_chunks:
            embeddings = self.embedder.encode(all_chunks)
            self.vectorstore.add(all_chunks, embeddings, all_metadatas, all_ids)

        return {
            "filename": filename,
            "characters": sum(len(c) for c in all_chunks),
            "chunks": len(all_chunks),
            "file_hash": file_hash
        }
