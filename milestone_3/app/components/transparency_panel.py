"""Response Transparency Panel Component for Streamlit (M3.4).

Renders deep evidence inspection, source chunk viewer, relevance score distribution,
confidence formula breakdown, and low-confidence / insufficient evidence diagnostics.
"""
import streamlit as st
from typing import Optional, List
from app.agents.models import (
    TransparencyPanelPayload,
    ConfidenceLevel,
    SourceAttribution,
    RetrievalChunk
)

def render_transparency_panel(payload: Optional[TransparencyPanelPayload], key_suffix: str = "0") -> None:
    """
    Renders an interactive, expandable tabbed Transparency Panel.
    """
    if not payload:
        st.info("No transparency data available for this response.")
        return

    # Confidence Indicator Banner
    conf_color_map = {
        ConfidenceLevel.HIGH: "🟢 **HIGH CONFIDENCE**",
        ConfidenceLevel.MEDIUM: "🟡 **MEDIUM CONFIDENCE**",
        ConfidenceLevel.LOW: "🟠 **LOW CONFIDENCE**",
        ConfidenceLevel.NONE: "🔴 **NO EVIDENCE / UNCERTAIN**"
    }
    conf_badge = conf_color_map.get(payload.confidence_level, "⚪ UNKNOWN")

    with st.expander(f"🔍 Response Transparency & Evidence Inspector — {conf_badge} ({payload.confidence_score * 100:.1f}%)", expanded=False):
        # Top summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Combined Confidence", f"{payload.confidence_score * 100:.1f}%")
        with col2:
            st.metric("Top Chunk Score", f"{payload.score_breakdown.top_chunk_score * 100:.1f}%")
        with col3:
            st.metric("Top-K Avg Score", f"{payload.score_breakdown.avg_top_k_score * 100:.1f}%")
        with col4:
            st.metric("Active Threshold", f"{payload.score_breakdown.threshold * 100:.1f}%")

        if payload.low_confidence_reason:
            st.warning(f"⚠️ **Low Confidence Diagnostic:** {payload.low_confidence_reason}")

        # Tabbed Inspection Interface
        tab_citations, tab_chunks, tab_diagnostics = st.tabs([
            "📚 Source Citations & Scores",
            "🧩 Retrieved Knowledge Chunks",
            "⚙️ Confidence & Formula Diagnostics"
        ])

        with tab_citations:
            if payload.citations:
                st.markdown("#### Supporting Sources Cited in Grounded Answer")
                for i, citation in enumerate(payload.citations, 1):
                    col_info, col_bar = st.columns([3, 2])
                    with col_info:
                        st.markdown(f"**[{i}] {citation.document_name}** `(Chunk ID: {citation.chunk_id})`")
                        page_str = f"Page {citation.page}" if citation.page else "Section: " + (citation.section or "General")
                        st.caption(f"📍 Location: `{page_str}`")
                        st.markdown(f"> *\"{citation.snippet}\"*")
                    with col_bar:
                        st.write(f"Relevance: **{citation.relevance_score * 100:.1f}%**")
                        st.progress(min(1.0, max(0.0, citation.relevance_score)))
                    st.divider()
            else:
                st.info("No supporting citations were retained above the confidence threshold.")

        with tab_chunks:
            if payload.retrieved_chunks:
                st.markdown("#### Full Ranked Evidence Chunks")
                for i, chunk in enumerate(payload.retrieved_chunks, 1):
                    with st.expander(f"Rank #{i}: {chunk.document_name} — Similarity: {chunk.similarity_score:.4f}"):
                        st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                        st.markdown(f"**Metadata:** `{chunk.metadata}`")
                        st.markdown("**Full Raw Text:**")
                        st.code(chunk.content, language="markdown")
            else:
                st.info("No raw chunks available.")

        with tab_diagnostics:
            st.markdown("#### Application-Level Confidence Formula")
            st.code(
                f"Confidence Score = 70% * (Top Chunk: {payload.score_breakdown.top_chunk_score:.3f}) + "
                f"30% * (Avg Top-K: {payload.score_breakdown.avg_top_k_score:.3f})\n"
                f"Computed Score = {payload.confidence_score:.4f} -> Categorized as {payload.confidence_level.value}",
                language="text"
            )
            st.markdown(f"**Retrieved Chunks Passing Filter:** `{len(payload.retrieved_chunks)}`")
            st.markdown(f"**Noise / Low-Score Chunks Filtered Out:** `{payload.filtered_out_count}`")
            st.markdown(f"**Source Document References:** `{', '.join(payload.source_documents) if payload.source_documents else 'None'}`")
