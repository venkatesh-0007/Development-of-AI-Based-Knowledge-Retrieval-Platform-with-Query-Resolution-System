"""Query Analytics and Knowledge Gap Detection Dashboard Component (M4.1).

Renders comprehensive telemetry dashboards, knowledge gap radar, query audit trails,
and domain/type/status filtering interfaces.
"""
import streamlit as st
import pandas as pd
from typing import Optional

from ..analytics.storage import AnalyticsStorage
from ..analytics.gap_detector import KnowledgeGapDetector
from ..analytics.models import AnalyticsFilter, GapSeverity

def render_analytics_dashboard(storage: AnalyticsStorage, gap_detector: KnowledgeGapDetector):
    st.markdown("### 📊 Query Analytics & Knowledge Gap Detection")
    st.caption("Operational telemetry, query resolution performance, and automated knowledge base gap analysis.")

    summary = storage.get_summary()

    # KPI Top Bar
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("Total Queries", summary.total_queries)
    with c2:
        st.metric("Resolution Rate", f"{summary.resolution_rate}%")
    with c3:
        st.metric("Unanswered Queries", summary.unanswered_count)
    with c4:
        st.metric("Low Confidence", summary.low_confidence_count)
    with c5:
        st.metric("Avg Latency", f"{summary.avg_latency_ms:.1f} ms")
    with c6:
        st.metric("Active Gaps", summary.active_gaps_count)

    st.divider()

    # Dashboard Tabs
    tab_overview, tab_gaps, tab_explorer, tab_export = st.tabs([
        "📈 Overview & Trends",
        "🎯 Knowledge Gap Radar",
        "🔍 Query Explorer & Logs",
        "📥 Data Export & Control"
    ])

    with tab_overview:
        st.subheader("Operational Telemetry & Distribution")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown("#### Domain Distribution")
            if summary.domain_breakdown:
                st.bar_chart(summary.domain_breakdown)
            else:
                st.info("No queries logged yet.")

        with row1_col2:
            st.markdown("#### Query Type Breakdown")
            if summary.query_type_breakdown:
                st.bar_chart(summary.query_type_breakdown)
            else:
                st.info("No queries logged yet.")

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.markdown("#### Resolution Status")
            if summary.status_breakdown:
                st.bar_chart(summary.status_breakdown)
            else:
                st.info("No status data available.")

        with row2_col2:
            st.markdown("#### Confidence Level Distribution")
            if summary.confidence_distribution:
                st.bar_chart(summary.confidence_distribution)
            else:
                st.info("No confidence distribution data available.")

    with tab_gaps:
        st.subheader("🎯 Automated Knowledge Gap Detection")
        st.caption("Identifies recurring query themes with low confidence or missing supporting documentation.")

        c_scan, c_space = st.columns([1, 3])
        with c_scan:
            if st.button("🔄 Run Knowledge Gap Analysis", use_container_width=True):
                with st.spinner("Clustering queries and computing gap severity..."):
                    gap_detector.detect_gaps()
                st.success("Knowledge gap analysis updated!")
                st.rerun()

        gaps = storage.get_gap_reports()
        if gaps:
            for gap in gaps:
                sev_badges = {
                    GapSeverity.CRITICAL: "🔴 **CRITICAL GAP**",
                    GapSeverity.HIGH: "🟠 **HIGH GAP**",
                    GapSeverity.MEDIUM: "🟡 **MEDIUM GAP**",
                    GapSeverity.LOW: "⚪ **LOW GAP**"
                }
                badge = sev_badges.get(gap.severity, "⚪ GAP")
                with st.expander(f"{badge} — {gap.topic} (Domain: {gap.domain.upper()} | Frequency: {gap.frequency})"):
                    g_c1, g_c2, g_c3 = st.columns(3)
                    with g_c1:
                        st.metric("Frequency (Queries)", gap.frequency)
                    with g_c2:
                        st.metric("Avg Confidence", f"{gap.average_confidence * 100:.1f}%")
                    with g_c3:
                        st.metric("Unanswered Count", gap.unanswered_count)

                    st.markdown("##### 💬 Sample User Queries:")
                    for sq in gap.sample_queries:
                        st.markdown(f"- *\"{sq}\"*")

                    st.markdown("##### 🛠️ Recommended Remedial Actions:")
                    for sa in gap.suggested_actions:
                        st.markdown(f"- ✅ {sa}")
        else:
            st.info("No knowledge base gaps currently detected. Ingest more queries or click 'Run Knowledge Gap Analysis'.")

    with tab_explorer:
        st.subheader("🔍 Query Explorer & Audit Logs")
        
        # Filter controls
        f_c1, f_c2, f_c3, f_c4 = st.columns(4)
        with f_c1:
            sel_domain = st.selectbox("Filter Domain", ["ALL", "machine_learning", "computer_networks", "cybersecurity", "general"])
        with f_c2:
            sel_type = st.selectbox("Filter Query Type", ["ALL", "factual", "procedural", "comparative", "ambiguous", "multi_part"])
        with f_c3:
            sel_conf = st.selectbox("Filter Confidence", ["ALL", "HIGH", "MEDIUM", "LOW", "NONE"])
        with f_c4:
            sel_status = st.selectbox("Filter Status", ["ALL", "RESOLVED", "LOW_CONFIDENCE", "UNANSWERED", "CLARIFICATION_REQUESTED", "CLARIFICATION_RESOLVED"])

        search_txt = st.text_input("Search Query or Response text...", "")

        flt = AnalyticsFilter(
            domain=sel_domain,
            query_type=sel_type,
            confidence_level=sel_conf,
            status=sel_status,
            search_query=search_txt if search_txt.strip() else None
        )

        entries = storage.get_queries(filter_criteria=flt, limit=100)
        st.caption(f"Showing {len(entries)} matching queries:")

        if entries:
            table_data = []
            for e in entries:
                table_data.append({
                    "Timestamp": e.timestamp[:19].replace("T", " "),
                    "Query": e.query_text,
                    "Domain": e.domain,
                    "Type": e.query_type,
                    "Confidence": f"{e.confidence_score * 100:.1f}% ({e.confidence_level})",
                    "Status": e.status.value,
                    "Latency (ms)": f"{e.latency_ms:.1f}",
                    "Modality": e.input_modality
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            with st.expander("🔎 Deep Log Inspector (Single Query Details)"):
                q_ids = [e.query_id for e in entries]
                sel_qid = st.selectbox("Select Query ID to inspect", q_ids)
                selected_entry = storage.get_query_by_id(sel_qid)
                if selected_entry:
                    st.json(selected_entry.to_dict())
        else:
            st.info("No queries found matching the selected filters.")

    with tab_export:
        st.subheader("📥 Analytics Data Export & Management")
        
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            csv_data = storage.export_csv()
            st.download_button(
                label="📄 Export Query Telemetry (CSV)",
                data=csv_data,
                file_name="query_analytics_telemetry.csv",
                mime="text/csv",
                use_container_width=True
            )
        with c_exp2:
            json_data = storage.export_json()
            st.download_button(
                label="📦 Export Full Analytics & Gap Report (JSON)",
                data=json_data,
                file_name="query_analytics_report.json",
                mime="application/json",
                use_container_width=True
            )

        st.divider()
        if st.button("🗑️ Clear Analytics Logs", type="secondary"):
            storage.clear()
            st.success("Analytics logs cleared successfully.")
            st.rerun()
