"""Streamlit UI for the Knowledge Retrieval Platform."""
import os
import streamlit as st
from dotenv import load_dotenv

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.pipeline import RetrievalPipeline
from app.agents.orchestrator import Orchestrator

load_dotenv()

st.set_page_config(page_title="AI Knowledge Retrieval Platform", layout="wide")
st.title("AI-Based Knowledge Retrieval Platform")
st.caption("Milestone 1 — Foundation & Knowledge Retrieval")

@st.cache_resource
def get_components():
    embedder = Embedder()
    store = ChromaStore()
    ingestion = IngestionPipeline(embedder, store)
    retrieval = RetrievalPipeline(embedder, store)
    orchestrator = Orchestrator(retrieval)
    return ingestion, retrieval, orchestrator, store

ingestion, retrieval, orchestrator, store = get_components()

with st.sidebar:
    st.header("Knowledge Base")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT or CSV",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    chunk_size = st.number_input("Chunk size", 300, 3000, 1000, 100)
    overlap = st.number_input("Chunk overlap", 0, 500, 150, 25)

    if st.button("Index Documents", disabled=not uploads):
        for f in uploads or []:
            try:
                result = ingestion.ingest(f.getvalue(), f.name, chunk_size, overlap)
                st.success(f"{f.name}: {result['chunks']} chunks indexed")
            except Exception as exc:
                st.error(f"{f.name}: {exc}")

    st.metric("Indexed chunks", store.count())

st.subheader("Ask the Knowledge Base")
query = st.text_input("Enter your question")
top_k = st.slider("Top-K retrieval", 1, 10, 5)

if st.button("Search / Answer", type="primary") and query.strip():
    output = orchestrator.run(query, top_k)
    st.markdown("### Retrieved Context")
    for i, item in enumerate(output["sources"], 1):
        st.write(f"**{i}. {item['metadata'].get('source')}** — similarity: {item['score']:.3f}")
        with st.expander("View chunk"):
            st.write(item["content"])

    st.markdown("### Response")
    st.write(output["answer"])

st.divider()
st.caption("Voice integration can be enabled in a later UI iteration using the browser Web Speech API.")
