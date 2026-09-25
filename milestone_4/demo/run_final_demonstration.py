"""Final Multi-Domain Demonstration Script for Milestone 4 (M4.4).

Demonstrates the complete, integrated AI-based Knowledge Retrieval Platform across:
1. Document ingestion across 3 distinct knowledge domains (ML, Networks, Cybersecurity)
2. Grounded multi-agent query resolution (Factual, Procedural, Comparative)
3. Interactive clarification dialogues and query refinement
4. Multi-turn conversation memory with anaphora pronoun resolution
5. Clean domain switching without cross-domain context bleeding
6. Voice TTS sanitization and Web Speech playback readiness
7. Response Transparency Panel payload with mathematical confidence formula
8. Query Analytics telemetry and Automated Knowledge Gap Detection
"""
import os
import sys
import time
from pathlib import Path

# Add milestone_4 root to path
m4_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(m4_root))

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

# ANSI color formatting for terminal demonstration
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_section(title: str, step: str = ""):
    print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{MAGENTA}[{step}] {title.upper()}{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}")

def main():
    print(f"\n{BOLD}{GREEN}================================================================================{RESET}")
    print(f"{BOLD}{GREEN} 🧠 AI KNOWLEDGE RETRIEVAL PLATFORM — MILESTONE 4 FINAL DEMONSTRATION{RESET}")
    print(f"{BOLD}{GREEN}================================================================================{RESET}")

    # Initialize Subsystems
    embedder = Embedder()
    store = ChromaStore(collection_name="demo_m4_kb")
    store.clear()
    pipeline = IngestionPipeline(embedder, store)

    db_path = str(m4_root / "data" / "analytics" / "demo_analytics.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    analytics_storage = AnalyticsStorage(db_path=db_path)
    gap_detector = KnowledgeGapDetector(analytics_storage, confidence_threshold=0.45)
    analytics_tracker = AnalyticsTracker(analytics_storage, low_confidence_threshold=0.45)

    clarification_agent = ClarificationAgent()
    query_agent = QueryUnderstandingAgent(clarification_agent=clarification_agent)
    retrieval_agent = RetrievalAgent(embedder, store, confidence_threshold=0.35)
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

    # STEP 1: Multi-Domain Knowledge Ingestion
    print_section("Ingesting 3 Distinct Knowledge Domains", "STAGE 1")
    sample_dir = m4_root / "data" / "sample_docs"
    docs = sorted(list(sample_dir.glob("*.*")))
    print(f"Loading {len(docs)} documents across Machine Learning, Computer Networks, and Cybersecurity...")
    
    total_chunks = 0
    for doc in docs:
        res = pipeline.ingest(doc.read_bytes(), doc.name, chunk_size=800, overlap=150)
        total_chunks += res["chunks"]
        print(f"  📥 Indexed {doc.name:<34} | Chunks: {res['chunks']} | Domain: {res['domain']}")

    print(f"{GREEN}✓ Ingestion complete. Total indexed chunks: {total_chunks}{RESET}")

    # STAGE 2: Multi-Domain Factual Queries
    print_section("Multi-Domain Factual Query Resolution", "STAGE 2")
    factual_queries = [
        ("Machine Learning", "What is Supervised Learning and what are its key applications?"),
        ("Computer Networks", "What is the function of the Transport Layer in the OSI 7-layer model?"),
        ("Cybersecurity", "What are the three pillars of the CIA Triad in information security?")
    ]
    for domain_name, q in factual_queries:
        print(f"\n{BOLD}Domain: {domain_name}{RESET}")
        print(f"Query: {YELLOW}\"{q}\"{RESET}")
        res = orchestrator.run(q)
        print(f"  • Status: {GREEN}{res.status}{RESET} | Latency: {res.execution_time_ms} ms")
        print(f"  • Confidence: {res.response.confidence_score*100:.1f}% ({res.response.confidence_level.value})")
        print(f"  • Sources: {', '.join(res.response.sources)}")
        preview = res.response.answer.split('\n\n')[0]
        print(f"  • Answer Summary: {preview[:140]}...")

    # STAGE 3: Procedural & Comparative Queries
    print_section("Procedural & Comparative Query Synthesis", "STAGE 3")
    procedural_q = "Explain the step by step process of the TCP three-way handshake"
    print(f"{BOLD}Procedural Query:{RESET} {YELLOW}\"{procedural_q}\"{RESET}")
    res_proc = orchestrator.run(procedural_q)
    print(f"  • Query Type: {res_proc.response.query_analysis.query_type.value.upper()}")
    print(f"  • Generated Steps Preview:\n    " + "\n    ".join(res_proc.response.answer.splitlines()[:5]))

    comparative_q = "Compare AES symmetric encryption with RSA asymmetric encryption"
    print(f"\n{BOLD}Comparative Query:{RESET} {YELLOW}\"{comparative_q}\"{RESET}")
    res_comp = orchestrator.run(comparative_q)
    print(f"  • Query Type: {res_comp.response.query_analysis.query_type.value.upper()}")
    print(f"  • Answer Summary: {res_comp.response.answer.splitlines()[0]}")

    # STAGE 4: Clarification Agent & Query Refinement
    print_section("Clarification Dialogues & Ambiguity Resolution", "STAGE 4")
    ambiguous_q = "Explain tree"
    print(f"Ambiguous Query: {YELLOW}\"{ambiguous_q}\"{RESET}")
    res_amb = orchestrator.run(ambiguous_q)
    print(f"  • Status: {RED}{res_amb.status}{RESET}")
    print(f"  • Reason: {res_amb.clarification_request.reason}")
    print(f"  • Follow-up Question: {BOLD}{res_amb.clarification_request.follow_up_question}{RESET}")
    print("  • Suggested Options:")
    for idx, opt in enumerate(res_amb.clarification_request.suggested_options, 1):
        print(f"    [{idx}] {opt}")

    # Simulate user choosing Option 1
    selected_option = "Decision Trees (Machine Learning / Classification)"
    print(f"\n{BOLD}User Selects Option 1:{RESET} {GREEN}\"{selected_option}\"{RESET}")
    res_clarified = orchestrator.resolve_clarification(
        clarification_id=res_amb.clarification_request.clarification_id,
        user_response=selected_option
    )
    print(f"  • Disambiguated Query: {BOLD}{res_clarified.response.refined_query}{RESET}")
    print(f"  • Status: {GREEN}{res_clarified.status}{RESET} | Confidence: {res_clarified.response.confidence_score*100:.1f}%")
    print(f"  • Sources: {', '.join(res_clarified.response.sources)}")

    # STAGE 5: Multi-Turn Conversation & Pronoun Resolution
    print_section("Multi-Turn Memory & Anaphora Resolution", "STAGE 5")
    session_id = "demo_session_1"
    turn1_q = "What is Support Vector Machines (SVM)?"
    print(f"Turn 1 Query: {YELLOW}\"{turn1_q}\"{RESET}")
    res_t1 = orchestrator.run(turn1_q, session_id=session_id)
    print(f"  • Answer Preview: {res_t1.response.answer[:120]}...")

    turn2_q = "What are its key characteristics and how does it use kernels?"
    print(f"\nTurn 2 Query (Anaphoric): {YELLOW}\"{turn2_q}\"{RESET}")
    res_t2 = orchestrator.run(turn2_q, session_id=session_id)
    print(f"  • Rewritten Query Context: {BOLD}{res_t2.response.transparency.anaphora_rewritten_query or turn2_q}{RESET}")
    print(f"  • Confidence: {res_t2.response.confidence_score*100:.1f}% | Sources: {', '.join(res_t2.response.sources)}")

    # STAGE 6: Domain Switching Validation
    print_section("Cross-Domain Context Switching (Isolation Check)", "STAGE 6")
    print("User abruptly switches from Machine Learning session to Cybersecurity query...")
    switch_q = "What is Zero Trust Security Architecture?"
    print(f"Domain Switch Query: {YELLOW}\"{switch_q}\"{RESET}")
    res_sw = orchestrator.run(switch_q, session_id=session_id)
    print(f"  • Detected Domain: {BOLD}{res_sw.response.query_analysis.domain}{RESET}")
    print(f"  • Correctly Isolated Context: {'SVM' not in res_sw.response.answer}")
    print(f"  • Sources: {', '.join(res_sw.response.sources)}")

    # STAGE 7: Voice TTS Readability
    print_section("Voice Interaction & Speech Sanitization", "STAGE 7")
    raw_markdown = "### Zero Trust Model\nZero Trust enforces **Never Trust, Always Verify** [1].\n- Policy 1\n- Policy 2\n```bash\nip route\n```"
    spoken_text = voice_module.clean_for_speech(raw_markdown)
    print(f"Raw Markdown Input:\n{raw_markdown}")
    print(f"\nSanitized Spoken TTS Output:\n{GREEN}\"{spoken_text}\"{RESET}")

    # STAGE 8: Knowledge Gap Detection Simulation
    print_section("Simulating Unanswered Queries & Knowledge Gap Radar", "STAGE 8")
    print("Injecting repeated out-of-domain queries to simulate knowledge gaps...")
    gap_queries = [
        "What are Quantum Key Distribution (QKD) and post-quantum cryptographic standards?",
        "Explain Quantum Key Distribution algorithms in cryptography",
        "How do post-quantum lattice cryptography and Kyber work?",
        "What is Reinforcement Learning from Human Feedback (RLHF) in LLMs?",
        "How does RLHF align language models using reward models?"
    ]
    for gq in gap_queries:
        res_g = orchestrator.run(gq)
        print(f"  • Query: \"{gq[:55]}...\" -> Status: {res_g.status} | Conf: {res_g.response.confidence_score:.2f}")

    print("\nRunning automated Knowledge Gap Detector...")
    detected_gaps = gap_detector.detect_gaps()
    print(f"{GREEN}✓ Detected {len(detected_gaps)} distinct knowledge base gaps:{RESET}")

    for idx, gap in enumerate(detected_gaps, 1):
        print(f"\n  [{idx}] {BOLD}{gap.severity.value} GAP:{RESET} {YELLOW}{gap.topic}{RESET} (Domain: {gap.domain.upper()})")
        print(f"      • Affected Query Count: {gap.frequency}")
        print(f"      • Average Confidence: {gap.average_confidence*100:.1f}%")
        print(f"      • Recommended Action: {gap.suggested_actions[0]}")

    # STAGE 9: Analytics Summary Report
    print_section("Operational Telemetry Summary Report", "STAGE 9")
    summary = analytics_storage.get_summary()
    print(f"Total Telemetry Log Entries:   {BOLD}{summary.total_queries}{RESET}")
    print(f"Resolution Rate:               {BOLD}{summary.resolution_rate}%{RESET}")
    print(f"Unanswered Queries:            {BOLD}{summary.unanswered_count}{RESET}")
    print(f"Low Confidence Rate:           {BOLD}{summary.low_confidence_count}{RESET}")
    print(f"Average System Latency:        {BOLD}{summary.avg_latency_ms} ms{RESET}")
    print(f"Domain Distribution:           {summary.domain_breakdown}")
    print(f"Active Knowledge Gaps:         {summary.active_gaps_count}")

    print(f"\n{BOLD}{GREEN}================================================================================{RESET}")
    print(f"{BOLD}{GREEN} 🎉 MILESTONE 4 FINAL DEMONSTRATION COMPLETED SUCCESSFULLY!{RESET}")
    print(f"{BOLD}{GREEN}================================================================================{RESET}\n")

if __name__ == "__main__":
    main()
