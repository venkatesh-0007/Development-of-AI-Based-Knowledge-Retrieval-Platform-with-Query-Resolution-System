"""Streamlit Web Application for Milestone 4 — Complete AI Knowledge Retrieval Platform.

Features:
- M4.1: Query Analytics & Automated Knowledge Gap Detection Module
- M4.2: End-to-End Multi-Domain Resolution (Machine Learning, Computer Networks, Cybersecurity)
- M4.3: Optimized Retrieval, Dynamic Top-K, Calibrated Prompts, Speech Sanitization & Transparency
- M4.4: Integrated Demonstration Mode & System Reporting
"""
import os
import sys
import uuid
from pathlib import Path

# Ensure milestone_4 root is on sys.path
m4_root = str(Path(__file__).resolve().parent.parent)
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

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
from app.analytics.storage import AnalyticsStorage
from app.analytics.gap_detector import KnowledgeGapDetector
from app.analytics.tracker import AnalyticsTracker
from app.components.voice_interface import render_voice_input_widget, render_tts_player
from app.components.transparency_panel import render_transparency_panel
from app.components.analytics_dashboard import render_analytics_dashboard

load_dotenv()

st.set_page_config(
    page_title="AI Knowledge Platform — Milestone 4",
    page_icon="🧠",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1d4ed8, #7c3aed, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🧠 AI-Based Knowledge Retrieval Platform & Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Milestone 4 — Multi-Domain Resolution, Gap Detection & Optimization</div>', unsafe_allow_html=True)

@st.cache_resource
def get_system():
    embedder = Embedder()
    store = ChromaStore(path="data/chroma", collection_name="knowledge_base_m4")
    ingestion = IngestionPipeline(embedder, store)
    
    analytics_storage = AnalyticsStorage(db_path="data/analytics/analytics.db")
    gap_detector = KnowledgeGapDetector(analytics_storage, confidence_threshold=0.45)
    analytics_tracker = AnalyticsTracker(analytics_storage, low_confidence_threshold=0.45)

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
        memory_agent=memory_agent,
        analytics_tracker=analytics_tracker
    )
    return ingestion, store, orchestrator, clarification_agent, memory_agent, voice_module, analytics_storage, gap_detector

ingestion, store, orchestrator, clarification_agent, memory_agent, voice_module, analytics_storage, gap_detector = get_system()

# Session State Initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_clarification" not in st.session_state:
    st.session_state.active_clarification = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Navigation & View")
    view_mode = st.radio(
        "Select Application Mode:",
        ["💬 Query & Voice Resolution", "📊 Query Analytics & Gap Radar"],
        label_visibility="collapsed"
    )

    st.divider()
    st.header("⚙️ Session & Memory")
    st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")
    session_obj = memory_agent.get_or_create_session(st.session_state.session_id)
    st.metric("Memory Turns", len(session_obj.turns))

    if st.button("🧹 Reset Conversation Memory", use_container_width=True):
        memory_agent.clear_session(st.session_state.session_id)
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.active_clarification = None
        st.session_state.last_result = None
        st.success("Session memory cleared!")
        st.rerun()

    st.divider()
    st.header("📂 Multi-Domain Knowledge Base")
    st.caption("Domains: Machine Learning • Computer Networks • Cybersecurity")

    if st.button("⚡ Index 3-Domain Benchmark Docs", use_container_width=True):
        sample_dir = Path(m4_root) / "data" / "sample_docs"
        if sample_dir.exists():
            count = 0
            for sf in sample_dir.glob("*.*"):
                if sf.suffix.lower() in [".txt", ".csv", ".pdf", ".docx"]:
                    try:
                        content = sf.read_bytes()
                        ingestion.ingest(content, sf.name, chunk_size=800, overlap=150)
                        count += 1
                    except Exception as e:
                        st.warning(f"Could not load {sf.name}: {e}")
            st.success(f"Indexed {count} documents across 3 domains!")
        else:
            st.error("Sample directory not found.")

    st.metric("Total Indexed Chunks", store.count())

    st.divider()
    st.subheader("Retrieval Tuning")
    min_confidence = st.slider("Min Confidence Threshold", 0.0, 1.0, 0.45, 0.05)
    top_k_select = st.slider("Top-K Chunks", 1, 10, 5)

# Render Selected View
if view_mode == "📊 Query Analytics & Gap Radar":
    render_analytics_dashboard(analytics_storage, gap_detector)

else:
    # 2-Column layout: Left Chat/Voice, Right Transparency
    chat_col, trans_col = st.columns([1.1, 0.9])

    with chat_col:
        st.subheader("💬 Multi-Domain Query & Voice Interaction")

        for chat_item in st.session_state.chat_history:
            with st.chat_message(chat_item["role"], avatar="🧑‍💻" if chat_item["role"] == "user" else "🤖"):
                st.markdown(chat_item["content"])
                if chat_item["role"] == "assistant" and "spoken_text" in chat_item and chat_item["spoken_text"]:
                    render_tts_player(chat_item["spoken_text"], element_id=chat_item.get("msg_id", "tts_prev"))

        # Voice Input STT
        render_voice_input_widget(element_id="voice_stt_m4")

        # Chat Input
        query_input = st.chat_input("Ask a question across ML, Networks, or Cybersecurity, or dictate...")

        if query_input and query_input.strip():
            active_q = query_input.strip()
            st.session_state.chat_history.append({"role": "user", "content": active_q})

            with st.spinner("Processing multi-agent query and updating telemetry..."):
                res = orchestrator.run(
                    query=active_q,
                    session_id=st.session_state.session_id,
                    top_k=top_k_select,
                    confidence_threshold=min_confidence,
                    input_modality="text"
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

        # Active Clarification Proposal
        if st.session_state.active_clarification:
            req = st.session_state.active_clarification
            st.markdown("---")
            st.warning("### ⚠️ Clarification Agent Triggered")
            st.markdown(f"**Original Query:** `{req.original_query}`")
            st.markdown(f"**Reason:** {req.reason}")
            st.info(f"**👉 Follow-up Question:** **{req.follow_up_question}**")

            if req.suggested_options:
                opt_cols = st.columns(len(req.suggested_options))
                for idx, opt in enumerate(req.suggested_options):
                    with opt_cols[idx]:
                        if st.button(f"Option {idx+1}: {opt}", key=f"opt_btn_{idx}"):
                            with st.spinner("Resolving clarification..."):
                                res = orchestrator.resolve_clarification(
                                    clarification_id=req.clarification_id,
                                    user_response=opt,
                                    session_id=st.session_state.session_id,
                                    top_k=top_k_select,
                                    confidence_threshold=min_confidence
                                )
                                st.session_state.last_result = res
                                st.session_state.active_clarification = None
                                st.session_state.chat_history.append({"role": "user", "content": opt})
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": f"🔄 *Refined Query: {res.response.refined_query}*\n\n" + res.response.answer,
                                    "spoken_text": res.response.spoken_text or res.response.answer,
                                    "msg_id": f"tts_res_{len(st.session_state.chat_history)}"
                                })
                            st.rerun()

            custom_clar = st.text_input("Or provide specific details:", key="custom_clar_input")
            if st.button("Submit Clarification", key="submit_clar_btn"):
                if custom_clar.strip():
                    with st.spinner("Resolving clarification..."):
                        res = orchestrator.resolve_clarification(
                            clarification_id=req.clarification_id,
                            user_response=custom_clar.strip(),
                            session_id=st.session_state.session_id,
                            top_k=top_k_select,
                            confidence_threshold=min_confidence
                        )
                        st.session_state.last_result = res
                        st.session_state.active_clarification = None
                        st.session_state.chat_history.append({"role": "user", "content": custom_clar})
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"🔄 *Refined Query: {res.response.refined_query}*\n\n" + res.response.answer,
                            "spoken_text": res.response.spoken_text or res.response.answer,
                            "msg_id": f"tts_res_{len(st.session_state.chat_history)}"
                        })
                    st.rerun()

    with trans_col:
        st.subheader("🔍 Response Transparency")
        if st.session_state.last_result and st.session_state.last_result.response:
            payload = getattr(st.session_state.last_result.response, "transparency", None)
            render_transparency_panel(payload)
        else:
            st.info("Ask a question to inspect retrieved chunks, similarity scores, and confidence breakdown.")
