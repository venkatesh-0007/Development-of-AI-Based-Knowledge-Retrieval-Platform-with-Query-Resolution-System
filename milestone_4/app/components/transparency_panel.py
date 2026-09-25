"""Response Transparency Panel Component for Streamlit (M4.2/M4.3).

Renders evidence inspection, source chunk viewer, relevance score distribution,
confidence formula breakdown, and low-confidence / insufficient evidence diagnostics.
"""
import streamlit as st
from typing import Optional
from ..agents.models import (
    TransparencyPanelPayload,
    ConfidenceLevel
)

def render_transparency_panel(payload: Optional[TransparencyPanelPayload], key_suffix: str = "0") -> None:
    if not payload:
        st.info("No transparency evidence payload available for this query.")
        return

    conf_color_map = {
        ConfidenceLevel.HIGH: "🟢 **HIGH CONFIDENCE**",
        ConfidenceLevel.MEDIUM: "🟡 **MEDIUM CONFIDENCE**",
        ConfidenceLevel.LOW: "🟠 **LOW CONFIDENCE**",
        ConfidenceLevel.NONE: "🔴 **NO EVIDENCE / UNCERTAIN**"
    }
    badge = conf_color_map.get(payload.score_breakdown.confidence_level, "⚪ UNKNOWN")
    score_pct = payload.score_breakdown.combined_score * 100

    with st.expander(f"🔍 Response Transparency & Evidence Panel — {badge} ({score_pct:.1f}%)", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Combined Confidence", f"{score_pct:.1f}%")
        with c2:
            st.metric("Top Chunk Score", f"{payload.score_breakdown.top_chunk_score * 100:.1f}%")
        with c3:
            st.metric("Avg Top-K Score", f"{payload.score_breakdown.avg_top_k_score * 100:.1f}%")
        with c4:
            st.metric("Query Type", payload.query_type.value.upper() if hasattr(payload.query_type, "value") else str(payload.query_type))

        if payload.anaphora_rewritten_query:
            st.info(f"🔄 **Anaphora Resolution Context:** Query rewritten to `{payload.anaphora_rewritten_query}`")

        if payload.clarification_occurred:
            st.success("✅ **Clarification Resolved:** Response synthesized from user disambiguation.")

        t_chunks, t_formula = st.tabs(["🧩 Retrieved Chunks & Sources", "📐 Mathematical Confidence Formula"])

        with t_chunks:
            if payload.ranked_chunks:
                for chunk in payload.ranked_chunks:
                    score_pct_chunk = chunk.score * 100
                    st.markdown(f"**#{chunk.rank} — `{chunk.document_name}`** (Similarity: **{score_pct_chunk:.1f}%** | Chunk ID: `{chunk.chunk_id}`)")
                    st.progress(min(1.0, max(0.0, chunk.score)))
                    st.caption(chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""))
                    st.divider()
            else:
                st.warning("No chunks met the minimum relevance threshold.")

        with t_formula:
            st.markdown(r"""
            **Confidence Score Formulation:**
            $$C = 0.70 \times s_{\text{top}} + 0.30 \times \bar{s}_{\text{top-}k}$$
            """)
            st.write({
                "Formula": "0.70 * top_score + 0.30 * avg_score",
                "Top Chunk Score": round(payload.score_breakdown.top_chunk_score, 4),
                "Average Top-K Score": round(payload.score_breakdown.avg_top_k_score, 4),
                "Combined Confidence Score": round(payload.score_breakdown.combined_score, 4),
                "Classification": payload.score_breakdown.confidence_level.value
            })
