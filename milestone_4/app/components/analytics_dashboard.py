"""Query Analytics and Knowledge Gap Detection Dashboard Component (M4.1 & Phase 6).

Renders comprehensive telemetry dashboards, knowledge gap radar, query audit trails,
and domain/type/status/date filtering interfaces.
"""
import streamlit as st
import pandas as pd
from typing import Optional
from datetime import datetime

from ..analytics.storage import AnalyticsStorage
from ..analytics.gap_detector import KnowledgeGapDetector
from ..analytics.models import AnalyticsFilter, GapSeverity


def render_analytics_dashboard(storage: AnalyticsStorage, gap_detector: KnowledgeGapDetector):
    st.markdown("### 📊 Query Analytics & Knowledge Gap Detection")
    st.caption("Operational telemetry, query resolution performance, and automated knowledge base gap analysis.")

    summary = storage.get_summary()
    answered_count = summary.total_queries - summary.unanswered_count

    # KPI Top Bar (All 7 required KPI cards)
    row_kpi_1 = st.columns(4)
    with row_kpi_1[0]:
        st.metric("Total Queries", summary.total_queries)
    with row_kpi_1[1]:
        st.metric("Answered Queries", answered_count)
    with row_kpi_1[2]:
        st.metric("Unanswered Queries", summary.unanswered_count)
    with row_kpi_1[3]:
        st.metric("Low Confidence", summary.low_confidence_count)

    row_kpi_2 = st.columns(4)
    with row_kpi_2[0]:
        st.metric("Clarification Requests", summary.clarification_count)
    with row_kpi_2[1]:
        st.metric("Average Confidence", f"{summary.avg_confidence * 100:.1f}%")
    with row_kpi_2[2]:
        st.metric("Average Latency", f"{summary.avg_latency_ms:.1f} ms")
    with row_kpi_2[3]:
        st.metric("Resolution Rate", f"{summary.resolution_rate:.1f}%")

    st.divider()

    # Dashboard Tabs
    tab_overview, tab_gaps, tab_explorer, tab_export = st.tabs([
        "📈 Overview & Charts",
        "🎯 Knowledge Gap Radar",
        "🔍 Query Explorer & Audit Logs",
        "📥 Data Export & Control"
    ])

    with tab_overview:
        st.subheader("Operational Telemetry & System Analytics")

        # Row 1: Queries over Time and Domain Distribution
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            st.markdown("#### Queries Over Time (Daily)")
            if summary.daily_query_volume:
                st.bar_chart(summary.daily_query_volume)
            else:
                st.info("No query volume time-series recorded yet.")

        with r1_c2:
            st.markdown("#### Domain Distribution")
            if summary.domain_breakdown:
                st.bar_chart(summary.domain_breakdown)
            else:
                st.info("No domain queries recorded yet.")

        # Row 2: Query Type Distribution and Confidence Distribution
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            st.markdown("#### Query Type Distribution")
            if summary.query_type_breakdown:
                st.bar_chart(summary.query_type_breakdown)
            else:
                st.info("No query type data available.")

        with r2_c2:
            st.markdown("#### Confidence Level Distribution")
            if summary.confidence_distribution:
                st.bar_chart(summary.confidence_distribution)
            else:
                st.info("No confidence distribution data available.")

        # Row 3: Resolution Status Distribution and Clarification Frequency
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            st.markdown("#### Resolution Status Distribution")
            if summary.status_breakdown:
                st.bar_chart(summary.status_breakdown)
            else:
                st.info("No status data available.")

        with r3_c2:
            st.markdown("#### Clarification Frequency & Metrics")
            clar_resolved = summary.status_breakdown.get("CLARIFICATION_RESOLVED", 0)
            clar_requested = summary.status_breakdown.get("CLARIFICATION_REQUESTED", 0)
            clar_data = {
                "Clarification Requested": clar_requested,
                "Clarification Resolved": clar_resolved,
                "Direct Resolved": summary.status_breakdown.get("RESOLVED", 0)
            }
            st.bar_chart(clar_data)

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
                    g_c1, g_c2, g_c3, g_c4 = st.columns(4)
                    with g_c1:
                        st.metric("Query Count", gap.frequency)
                    with g_c2:
                        st.metric("Avg Confidence", f"{gap.average_confidence * 100:.1f}%")
                    with g_c3:
                        st.metric("Avg Retrieval Score", f"{gap.average_retrieval_score:.3f}")
                    with g_c4:
                        st.metric("Unanswered Count", gap.unanswered_count)

                    st.markdown(
                        f"**Affected Domain:** `{gap.domain}` | "
                        f"**First Seen:** `{gap.first_detected[:19].replace('T', ' ')}` | "
                        f"**Last Seen:** `{gap.last_detected[:19].replace('T', ' ')}`"
                    )

                    st.markdown("##### 💬 Sample User Queries:")
                    for sq in gap.sample_queries:
                        st.markdown(f"- *\"{sq}\"*")

                    st.markdown("##### 🛠️ Recommended Remedial Actions:")
                    for sa in gap.suggested_actions:
                        st.markdown(f"- ✅ {sa}")
        else:
            st.info("No knowledge base gaps currently detected. Ingest queries or click 'Run Knowledge Gap Analysis'.")

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

        # Date and text filter controls
        f_d1, f_d2, f_d3 = st.columns([1, 1, 2])
        with f_d1:
            date_from = st.text_input("Date From (YYYY-MM-DD)", "")
        with f_d2:
            date_to = st.text_input("Date To (YYYY-MM-DD)", "")
        with f_d3:
            search_txt = st.text_input("Search Query or Response text...", "")

        flt = AnalyticsFilter(
            domain=sel_domain,
            query_type=sel_type,
            confidence_level=sel_conf,
            status=sel_status,
            search_query=search_txt if search_txt.strip() else None,
            date_from=date_from if date_from.strip() else None,
            date_to=date_to if date_to.strip() else None
        )

        entries = storage.get_queries(filter_criteria=flt, limit=200)
        st.caption(f"Showing {len(entries)} matching queries from persistent analytics storage:")

        if entries:
            table_data = []
            for e in entries:
                table_data.append({
                    "Timestamp": e.timestamp[:19].replace("T", " "),
                    "Query": e.query_text,
                    "Domain": e.domain,
                    "Query Type": e.query_type,
                    "Confidence": f"{e.confidence_score * 100:.1f}% ({e.confidence_level})",
                    "Resolution Status": e.status.value,
                    "Response Time (ms)": f"{e.latency_ms:.1f}",
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
