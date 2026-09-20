"""Streamlit UI for Milestone 3 — AI Knowledge Retrieval Platform.

Features:
- M3.1: Clarification Agent (Interactive ambiguity resolution dialog & option suggestions)
- M3.2: Conversation Memory Agent (Multi-turn session tracking & anaphora resolution)
- M3.3: Voice Input & Text-to-Speech (Live Web Speech STT microphone & inline TTS player)
- M3.4: Response Transparency Panel (Chunk ranking, similarity score bars, confidence formula breakdown)
"""
import os
import sys
import uuid
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
from app.agents.memory import ConversationMemoryAgent
from app.agents.voice import VoiceModule
from app.agents.orchestrator import MultiAgentOrchestrator
from app.agents.models import ConfidenceLevel, QueryType, ClarificationType, ClarificationStatus
from app.components.voice_interface import render_voice_input_widget, render_tts_player
from app.components.transparency_panel import render_transparency_panel

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Retrieval Platform — Milestone 3",
    page_icon="🎙️",
    layout="wide"
)

# Custom header styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6b7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .chat-bubble-user {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .chat-bubble-bot {
        background-color: #f9fafb;
        border-left: 4px solid #8b5cf6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎙️ Multi-Agent Knowledge Retrieval & Voice Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Milestone 3 — Clarification, Conversation Memory, Voice Interaction & Transparency</div>', unsafe_allow_html=True)

@st.cache_resource
def get_system():
    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="knowledge_base_m3")
    ingestion = IngestionPipeline(embedder, store)
    
    clarification_agent = ClarificationAgent()
    query_agent = QueryUnderstandingAgent(clarification_agent=clarification_agent)
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.45)
    response_agent = ResponseGenerationAgent()
    memory_agent = ConversationMemoryAgent()
    voice_module = VoiceModule()
    
    orchestrator = MultiAgentOrchestrator(
        retrieval_agent=retrieval_agent,
        query_understanding_agent=query_agent,
        response_generation_agent=response_agent,
        clarification_agent=clarification_agent,
        memory_agent=memory_agent
    )
    return ingestion, store, orchestrator, clarification_agent, memory_agent, voice_module

ingestion, store, orchestrator, clarification_agent, memory_agent, voice_module = get_system()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_clarification" not in st.session_state:
    st.session_state.active_clarification = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Sidebar: Knowledge Base Management & Conversation Memory Controls
with st.sidebar:
    st.header("⚙️ Session & Memory")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    
    session_obj = memory_agent.get_or_create_session(st.session_state.session_id)
    st.metric("Memory Turn Count", len(session_obj.turns))
    
    if st.button("🧹 Reset Conversation Memory", use_container_width=True):
        memory_agent.clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.active_clarification = None
        st.session_state.last_result = None
        st.success("Conversation memory reset!")
        st.rerun()

    st.divider()
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
    if st.button("Load Sample Knowledge Docs", use_container_width=True):
        sample_dir = Path(milestone_3_root) / "data" / "sample_docs"
        if sample_dir.exists():
            count = 0
            for sample_file in sample_dir.glob("*.*"):
                if sample_file.suffix.lower() in [".txt", ".csv", ".pdf", ".docx"]:
                    try:
                        content = sample_file.read_bytes()
                        ingestion.ingest(content, sample_file.name, chunk_size, overlap)
                        count += 1
                    except Exception as e:
                        st.warning(f"Could not load {sample_file.name}: {e}")
            st.success(f"Indexed {count} benchmark documents!")
        else:
            st.error("Sample directory not found.")

    st.metric("Total Indexed Chunks", store.count())
    
    st.divider()
    st.subheader("Retrieval Tuning")
    min_confidence = st.slider("Min Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    top_k_select = st.slider("Top-K Chunks", 1, 10, 5)

# Main Workspace: 2-column layout (Chat/Voice on Left, Transparency on Right)
chat_col, trans_col = st.columns([1.1, 0.9])

with chat_col:
    st.subheader("💬 Multi-Turn Conversation & Voice Query")
    
    # Render Conversation History
    for chat_item in st.session_state.chat_history:
        with st.chat_message(chat_item["role"], avatar="🧑‍💻" if chat_item["role"] == "user" else "🤖"):
            st.markdown(chat_item["content"])
            if chat_item["role"] == "assistant" and "spoken_text" in chat_item and chat_item["spoken_text"]:
                render_tts_player(chat_item["spoken_text"], element_id=chat_item.get("msg_id", "tts_prev"))

    # Voice Input Microphone Component (Web Speech API)
    voice_transcript = render_voice_input_widget(element_id="voice_stt_main")

    # Interactive Query Input
    query_input = st.chat_input("Ask a question, follow up on previous context, or speak...")
    
    # Process Voice or Text Query
    active_query = None
    if query_input:
        active_query = query_input
    elif voice_transcript:
        active_query = voice_transcript

    if active_query and active_query.strip():
        # Record user query in display history
        st.session_state.chat_history.append({"role": "user", "content": active_query})
        
        with st.spinner("Analyzing context, retrieving knowledge, and generating response..."):
            res = orchestrator.run(
                query=active_query,
                session_id=st.session_state.session_id,
                top_k=top_k_select,
                confidence_threshold=min_confidence
            )
            st.session_state.last_result = res
            
            if res.status == "clarification_requested":
                st.session_state.active_clarification = res.clarification_request
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ **Clarification Needed**: {res.clarification_request.follow_up_question}",
                    "spoken_text": res.clarification_request.follow_up_question,
                    "msg_id": f"tts_clar_{len(st.session_state.chat_history)}"
                })
            else:
                st.session_state.active_clarification = None
                resp_text = res.response.answer
                if res.response.is_clarified_resolution and res.response.refined_query:
                    resp_text = f"🔄 *Refined Query: {res.response.refined_query}*\n\n" + resp_text
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": resp_text,
                    "spoken_text": res.response.spoken_text or resp_text,
                    "msg_id": f"tts_{len(st.session_state.chat_history)}"
                })
        st.rerun()

    # Active Clarification Dialog (M3.1)
    if st.session_state.active_clarification:
        req = st.session_state.active_clarification
        st.markdown("---")
        st.warning("### ⚠️ Clarification Agent Triggered")
        st.markdown(f"**Original Query:** `{req.original_query}`")
        st.markdown(f"**Ambiguity Reason:** {req.reason}")
        st.info(f"**👉 Follow-up Question:** **{req.follow_up_question}**")

        st.markdown("#### Choose an Option or Type Your Clarification:")
        if req.suggested_options:
            opt_cols = st.columns(len(req.suggested_options))
            for idx, opt in enumerate(req.suggested_options):
                with opt_cols[idx]:
                    if st.button(f"📌 {opt}", key=f"opt_btn_{idx}", use_container_width=True):
                        with st.spinner("Resolving clarification and synthesizing answer..."):
                            resolved_res = orchestrator.resolve_clarification(
                                clarification_id=req.clarification_id,
                                user_response=opt,
                                session_id=st.session_state.session_id,
                                top_k=top_k_select,
                                confidence_threshold=min_confidence
                            )
                            st.session_state.last_result = resolved_res
                            st.session_state.active_clarification = None
                            
                            st.session_state.chat_history.append({"role": "user", "content": opt})
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"🔄 *Refined:* `{resolved_res.response.refined_query}`\n\n" + resolved_res.response.answer,
                                "spoken_text": resolved_res.response.spoken_text,
                                "msg_id": f"tts_res_{len(st.session_state.chat_history)}"
                            })
                            st.rerun()

        # Freeform text clarification
        with st.form("freeform_clarification_form"):
            freeform_text = st.text_input("Custom Response:", placeholder="e.g. 'I am referring to TCP congestion control'")
            submit_clar = st.form_submit_button("Submit Clarification")
            if submit_clar and freeform_text.strip():
                with st.spinner("Resolving clarification..."):
                    resolved_res = orchestrator.resolve_clarification(
                        clarification_id=req.clarification_id,
                        user_response=freeform_text,
                        session_id=st.session_state.session_id,
                        top_k=top_k_select,
                        confidence_threshold=min_confidence
                    )
                    st.session_state.last_result = resolved_res
                    st.session_state.active_clarification = None
                    
                    st.session_state.chat_history.append({"role": "user", "content": freeform_text})
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"🔄 *Refined:* `{resolved_res.response.refined_query}`\n\n" + resolved_res.response.answer,
                        "spoken_text": resolved_res.response.spoken_text,
                        "msg_id": f"tts_res_{len(st.session_state.chat_history)}"
                    })
                    st.rerun()

with trans_col:
    st.subheader("🔍 Response Transparency & Verification")
    if st.session_state.last_result and st.session_state.last_result.response:
        resp = st.session_state.last_result.response
        if resp.transparency:
            render_transparency_panel(resp.transparency)
        else:
            st.info("No retrieval transparency data for current turn (e.g. Clarification pending).")
    else:
        st.info("Ask a query or speak into the microphone to see retrieval chunks, relevance scores, confidence formulas, and citation evidence.")

