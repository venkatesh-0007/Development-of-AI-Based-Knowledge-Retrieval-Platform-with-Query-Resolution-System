"""Streamlit UI for Milestone 3.1 — Clarification Agent Platform."""
import os
import sys
from pathlib import Path

# Ensure milestone_3 root is in sys.path
milestone_3_root = str(Path(__file__).resolve().parent.parent)
if milestone_3_root not in sys.path:
    sys.path.insert(0, milestone_3_root)

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
from app.agents.models import ConfidenceLevel, QueryType, ClarificationType, ClarificationStatus

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Retrieval Platform — Milestone 3.1",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Knowledge Retrieval & Multi-Agent Clarification Platform")
st.caption("Milestone 3.1 — Intelligent Clarification Agent, Ambiguity Detection & Query Refinement")

@st.cache_resource
def get_system():
    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="knowledge_base_m3")
    ingestion = IngestionPipeline(embedder, store)
    
    clarification_agent = ClarificationAgent()
    query_agent = QueryUnderstandingAgent(clarification_agent=clarification_agent)
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.45)
    response_agent = ResponseGenerationAgent()
    
    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=query_agent,
        response_generation_agent=response_agent,
        clarification_agent=clarification_agent
    )
    return ingestion, store, orchestrator, clarification_agent

ingestion, store, orchestrator, clarification_agent = get_system()

# Initialize session state for interactive clarification dialogues
if "active_clarification" not in st.session_state:
    st.session_state.active_clarification = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Sidebar: Knowledge Base Management & Benchmarks
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
        sample_dir = Path(milestone_3_root) / "data" / "sample_docs"
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

# Main Query Area
st.subheader("💬 Query Resolution & Clarification Interface")

# Sample Query Presets
st.markdown("**Try Example Queries:**")
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
sample_query = None
with col_p1:
    if st.button("🔍 Polysemous: 'Explain tree'", use_container_width=True):
        sample_query = "Explain tree"
with col_p2:
    if st.button("❓ Missing Context: 'How does it work?'", use_container_width=True):
        sample_query = "How does it work?"
with col_p3:
    if st.button("✂️ Incomplete: 'Compare SVM'", use_container_width=True):
        sample_query = "Compare SVM"
with col_p4:
    if st.button("⚡ Multi-Part Query", use_container_width=True):
        sample_query = "What is TCP, how does it differ from UDP, and what port does DNS use?"

query_input = st.text_input(
    "Enter your query:",
    value=sample_query or "",
    placeholder="e.g., 'Explain tree' or 'What is TCP?' or 'How does it work?'"
)

col_act1, col_act2 = st.columns([1, 5])
with col_act1:
    resolve_btn = st.button("🚀 Submit Query", type="primary", use_container_width=True)

if resolve_btn and query_input.strip():
    with st.spinner("Processing through Clarification & Multi-Agent pipeline..."):
        res = orchestrator.run(
            query=query_input,
            top_k=top_k_select,
            confidence_threshold=min_confidence
        )
        st.session_state.last_result = res
        if res.status == "clarification_requested":
            st.session_state.active_clarification = res.clarification_request
        else:
            st.session_state.active_clarification = None

# Display Clarification Box if clarification is active
if st.session_state.active_clarification:
    req = st.session_state.active_clarification
    st.markdown("---")
    st.warning("### ⚠️ Clarification Agent Triggered")
    st.markdown(f"**Original Query:** `{req.original_query}`")
    st.markdown(f"**Ambiguity Reason:** {req.reason}")
    st.info(f"**👉 Follow-up Question:** **{req.follow_up_question}**")

    st.markdown("#### Choose an Option or Type Your Clarification:")
    
    # Suggested Options as Interactive Buttons
    if req.suggested_options:
        opt_cols = st.columns(len(req.suggested_options))
        for idx, opt in enumerate(req.suggested_options):
            with opt_cols[idx]:
                if st.button(f"📌 {opt}", key=f"opt_btn_{idx}", use_container_width=True):
                    with st.spinner("Refining query and resolving..."):
                        resolved_res = orchestrator.resolve_clarification(
                            clarification_id=req.clarification_id,
                            user_response=opt,
                            top_k=top_k_select,
                            confidence_threshold=min_confidence
                        )
                        st.session_state.last_result = resolved_res
                        st.session_state.active_clarification = None
                        st.rerun()

    # Freeform custom clarification input
    with st.form("custom_clarification_form"):
        custom_input = st.text_input("Or enter custom clarification:", placeholder="e.g. 'I meant Decision Trees in Machine Learning'")
        submit_custom = st.form_submit_button("Submit Clarification")
        if submit_custom and custom_input.strip():
            with st.spinner("Refining query and resolving..."):
                resolved_res = orchestrator.resolve_clarification(
                    clarification_id=req.clarification_id,
                    user_response=custom_input,
                    top_k=top_k_select,
                    confidence_threshold=min_confidence
                )
                st.session_state.last_result = resolved_res
                st.session_state.active_clarification = None
                st.rerun()

# Display Orchestration Results
if st.session_state.last_result:
    result = st.session_state.last_result
    resp = result.response
    qa = resp.query_analysis

    if result.status in ("success", "clarified_success"):
        st.markdown("---")
        if resp.is_clarified_resolution and resp.refined_query:
            st.success(f"🔄 **Query Successfully Refined & Disambiguated:** `{resp.refined_query}`")

        st.markdown("### 🔍 Multi-Agent Query Analysis")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.markdown(f"**Query Intent:** :blue[**{qa.query_type.value.upper()}**]")

        with col_m2:
            st.metric("Classifier Confidence", f"{qa.classification_confidence * 100:.1f}%")

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
            st.caption(f"**Identified Entities:** `{'`, `'.join(qa.entities)}`")

        st.divider()

        # Grounded Answer Display
        st.markdown("### 📝 Grounded Response")
        if not resp.has_sufficient_evidence:
            st.info(resp.answer)
        else:
            st.markdown(resp.answer)

        # Citations & Evidence
        if resp.sources:
            st.divider()
            st.markdown("### 📚 Source Citations & Evidence Attribution")
            for i, src in enumerate(resp.sources, 1):
                with st.expander(f"Citation [{i}]: {src.document_name} — Relevance Score: {src.relevance_score:.3f}"):
                    st.markdown(f"**Chunk ID:** `{src.chunk_id}`")
                    st.markdown(f"**Snippet:**\n> {src.snippet}")
                    if result.retrieval_result and i <= len(result.retrieval_result.chunks):
                        st.markdown("**Full Chunk Content:**")
                        st.code(result.retrieval_result.chunks[i-1].content)
