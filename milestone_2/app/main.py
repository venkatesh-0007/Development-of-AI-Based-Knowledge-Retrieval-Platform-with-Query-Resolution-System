"""Streamlit UI for Milestone 2 Multi-Agent Query Resolution Platform."""
import os
import sys
from pathlib import Path

# Ensure milestone_2 root is in sys.path
milestone_2_root = str(Path(__file__).resolve().parent.parent)
if milestone_2_root not in sys.path:
    sys.path.insert(0, milestone_2_root)

import streamlit as st
from dotenv import load_dotenv

from app.embeddings.embedder import Embedder
from app.vectorstore.chroma_store import ChromaStore
from app.ingestion.pipeline import IngestionPipeline
from app.agents.query_understanding import QueryUnderstandingAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.response_generation import ResponseGenerationAgent
from app.agents.clarification import ClarificationAgent
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import ConfidenceLevel, QueryType

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Retrieval Platform — Milestone 2",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Based Knowledge Retrieval Platform")
st.caption("Milestone 2 — Multi-Agent Query Resolution & Response Generation")

@st.cache_resource
def get_system():
    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="knowledge_base_m2")
    ingestion = IngestionPipeline(embedder, store)
    
    query_agent = QueryUnderstandingAgent()
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.45)
    response_agent = ResponseGenerationAgent()
    clarification_agent = ClarificationAgent()
    
    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=query_agent,
        response_generation_agent=response_agent,
        clarification_agent=clarification_agent
    )
    return ingestion, store, orchestrator

ingestion, store, orchestrator = get_system()

# Sidebar: Knowledge Base Management
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploads = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, CSV)",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chunk_size = st.number_input("Chunk size", 200, 3000, 800, 100)
    with col_c2:
        overlap = st.number_input("Overlap", 0, 500, 150, 25)

    if st.button("📥 Index Uploaded Files", disabled=not uploads, use_container_width=True):
        for f in uploads or []:
            try:
                res = ingestion.ingest(f.getvalue(), f.name, chunk_size, overlap)
                st.success(f"{f.name}: {res['chunks']} chunks indexed")
            except Exception as exc:
                st.error(f"{f.name}: {exc}")

    st.divider()
    st.subheader("Benchmark Sample Datasets")
    if st.button("Load ML & Networks Samples", use_container_width=True):
        sample_dir = Path(milestone_2_root) / "data" / "sample_docs"
        if sample_dir.exists():
            for sample_file in sample_dir.glob("*.*"):
                if sample_file.suffix.lower() in [".txt", ".csv", ".pdf", ".docx"]:
                    try:
                        content = sample_file.read_bytes()
                        ingestion.ingest(content, sample_file.name, chunk_size, overlap)
                        st.info(f"Loaded: {sample_file.name}")
                    except Exception as e:
                        st.warning(f"Could not load {sample_file.name}: {e}")
            st.success("Sample datasets indexed successfully!")
        else:
            st.error("Sample directory not found.")

    st.metric("Total Indexed Chunks", store.count())
    
    st.divider()
    st.subheader("Retrieval Tuning")
    min_confidence = st.slider("Min Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    top_k_select = st.slider("Top-K Chunks", 1, 10, 5)

# Main Area: Query Input & Multi-Agent Resolution
st.subheader("💬 Ask the Multi-Agent System")
query_text = st.text_input(
    "Enter your query (Factual, Procedural, Comparative, or Ambiguous):",
    placeholder="e.g., 'Compare TCP vs UDP' or 'How to implement k-fold cross-validation?'"
)

col_b1, col_b2 = st.columns([1, 5])
with col_b1:
    search_clicked = st.button("🚀 Resolve Query", type="primary", use_container_width=True)

if search_clicked and query_text.strip():
    with st.spinner("Multi-Agent pipeline executing..."):
        result = orchestrator.run(
            query=query_text,
            top_k=top_k_select,
            confidence_threshold=min_confidence
        )

    resp = result.response
    qa = resp.query_analysis

    # Classification & Pipeline Trace Display
    st.markdown("### 🔍 Multi-Agent Query Analysis")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        type_color = {
            QueryType.FACTUAL: "blue",
            QueryType.PROCEDURAL: "orange",
            QueryType.COMPARATIVE: "green",
            QueryType.AMBIGUOUS: "red"
        }.get(qa.query_type, "gray")
        st.markdown(f"**Query Intent:** :{type_color}[**{qa.query_type.value.upper()}**]")

    with col_m2:
        st.metric("Classifier Confidence", f"{qa.confidence * 100:.1f}%")

    with col_m3:
        conf_color = {
            ConfidenceLevel.HIGH: "🟢 HIGH",
            ConfidenceLevel.MEDIUM: "🟡 MEDIUM",
            ConfidenceLevel.LOW: "🟠 LOW",
            ConfidenceLevel.NONE: "🔴 NONE"
        }.get(resp.confidence_level, "⚪ N/A")
        st.markdown(f"**Response Confidence:**\n### {conf_color} ({resp.confidence_score * 100:.1f}%)")

    with col_m4:
        st.metric("Execution Latency", f"{result.execution_time_ms} ms")

    if qa.entities:
        st.caption(f"**Extracted Entities / Keywords:** `{'`, `'.join(qa.entities)}`")

    st.divider()

    # Grounded Answer Display
    st.markdown("### 📝 Grounded Response")
    if resp.requires_clarification:
        st.warning(resp.answer)
    elif not resp.has_sufficient_evidence:
        st.info(resp.answer)
    else:
        st.markdown(resp.answer)

    # Source Attribution & Chunk Inspection
    if resp.sources:
        st.divider()
        st.markdown("### 📚 Source Citations & Evidence Attribution")
        for i, src in enumerate(resp.sources, 1):
            with st.expander(f"Citation {i}: {src.document_name} — Similarity Score: {src.relevance_score:.3f}"):
                st.markdown(f"**Chunk ID:** `{src.chunk_id}`")
                st.markdown(f"**Snippet:**\n> {src.snippet}")
                if result.retrieval_result and i <= len(result.retrieval_result.chunks):
                    st.markdown("**Full Chunk Content:**")
                    st.code(result.retrieval_result.chunks[i-1].content)
