"""
Florida Tax Law Assistant - Streamlit Frontend

A simple web interface for querying the Florida Tax RAG API.
"""

import streamlit as st
import requests
from typing import Optional

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Florida Tax Law Assistant",
    page_icon="🏛️",
    layout="wide",
)


def check_health() -> dict:
    """Check API health status."""
    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "unavailable", "error": "Cannot connect to API"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def query_tax_law(question: str, options: Optional[dict] = None) -> dict:
    """Send a query to the RAG API."""
    try:
        payload = {"query": question}
        if options:
            payload["options"] = options

        response = requests.post(
            f"{API_URL}/api/v1/query",
            json=payload,
            timeout=120  # Long timeout for complex queries
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "CONNECTION_ERROR", "message": "Cannot connect to API. Is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "TIMEOUT", "message": "Request timed out"}
    except Exception as e:
        return {"error": "ERROR", "message": str(e)}


def get_source_details(chunk_id: str) -> dict:
    """Get full details for a source chunk."""
    try:
        response = requests.get(f"{API_URL}/api/v1/sources/{chunk_id}", timeout=10)
        return response.json()
    except Exception:
        return None


def display_confidence_badge(confidence: float):
    """Display a colored confidence badge."""
    if confidence >= 0.8:
        color = "green"
        label = "High"
    elif confidence >= 0.5:
        color = "orange"
        label = "Medium"
    else:
        color = "red"
        label = "Low"

    st.markdown(
        f'<span style="background-color: {color}; color: white; padding: 4px 8px; '
        f'border-radius: 4px; font-size: 14px;">{label} Confidence: {confidence:.0%}</span>',
        unsafe_allow_html=True
    )


def main():
    # Title
    st.title("🏛️ Florida Tax Law Assistant")
    st.markdown("Ask questions about Florida tax law and get answers with legal citations.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Options")

        # Document type filters
        st.subheader("Document Types")
        include_statutes = st.checkbox("Statutes", value=True)
        include_rules = st.checkbox("Administrative Rules", value=True)
        include_cases = st.checkbox("Court Cases", value=True)
        include_taas = st.checkbox("TAAs", value=True)

        # Build doc_types list
        doc_types = []
        if include_statutes:
            doc_types.append("statute")
        if include_rules:
            doc_types.append("rule")
        if include_cases:
            doc_types.append("case")
        if include_taas:
            doc_types.append("taa")

        # Tax year
        st.subheader("Tax Year")
        tax_year = st.number_input(
            "Year",
            min_value=1990,
            max_value=2030,
            value=2024,
            help="Filter results to a specific tax year"
        )

        # Advanced options
        st.subheader("Advanced")
        include_reasoning = st.checkbox(
            "Show reasoning steps",
            value=False,
            help="Include detailed reasoning in the response"
        )

        st.divider()

        # Health status
        st.subheader("API Status")
        health = check_health()
        if health.get("status") == "healthy":
            st.success("🟢 API Healthy")
            if "services" in health:
                for svc in health["services"]:
                    status_icon = "✅" if svc.get("healthy") else "❌"
                    st.caption(f"{status_icon} {svc['name']}: {svc.get('latency_ms', 'N/A')}ms")
        elif health.get("status") == "degraded":
            st.warning("🟡 API Degraded")
        else:
            st.error("🔴 API Unavailable")
            if "error" in health:
                st.caption(health["error"])

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        question = st.text_area(
            "Ask a question about Florida tax law:",
            placeholder="e.g., What is the Florida sales tax rate?",
            height=100
        )

        # Submit button
        submitted = st.button("🔍 Search", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Example questions:**")
        examples = [
            "What is the Florida sales tax rate?",
            "Are groceries taxable in Florida?",
            "What exemptions exist for manufacturers?",
            "How is use tax calculated?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                question = ex
                submitted = True

    # Process query
    if submitted and question:
        # Build options
        options = {
            "timeout_seconds": 90,
            "include_reasoning": include_reasoning,
        }
        if doc_types:
            options["doc_types"] = doc_types
        if tax_year:
            options["tax_year"] = tax_year

        # Show spinner while processing
        with st.spinner("Searching Florida tax law..."):
            result = query_tax_law(question, options)

        # Check for errors
        if "error" in result and "answer" not in result:
            st.error(f"❌ Error: {result.get('message', result.get('error'))}")
            return

        # Display results
        st.divider()

        # Answer section
        st.subheader("📝 Answer")
        st.markdown(result.get("answer", "No answer available"))

        # Metrics row
        col_conf, col_time, col_valid = st.columns(3)

        with col_conf:
            confidence = result.get("confidence", 0)
            display_confidence_badge(confidence)

        with col_time:
            proc_time = result.get("processing_time_ms", 0)
            st.metric("Processing Time", f"{proc_time/1000:.1f}s")

        with col_valid:
            validation = result.get("validation_passed", True)
            if validation:
                st.success("✅ Validated")
            else:
                st.warning("⚠️ Needs Review")

        # Warnings
        warnings = result.get("warnings", [])
        for warning in warnings:
            st.warning(f"⚠️ {warning}")

        # Reasoning steps (if requested)
        if include_reasoning and result.get("reasoning_steps"):
            with st.expander("🧠 Reasoning Steps", expanded=False):
                for i, step in enumerate(result["reasoning_steps"], 1):
                    st.markdown(f"**Step {i}:** {step}")

        # Sources section
        sources = result.get("sources", [])
        if sources:
            st.divider()
            st.subheader(f"📚 Sources ({len(sources)})")

            for source in sources:
                citation = source.get("citation", source.get("doc_id", "Unknown"))
                doc_type = source.get("doc_type", "unknown")
                relevance = source.get("relevance_score", 0)
                text = source.get("text", "")

                # Type badge color
                type_colors = {
                    "statute": "🔵",
                    "rule": "🟢",
                    "case": "🟠",
                    "taa": "🟣"
                }
                type_badge = type_colors.get(doc_type, "⚪")

                with st.expander(f"{type_badge} {citation} (relevance: {relevance:.0%})"):
                    st.markdown(f"**Document Type:** {doc_type.title()}")
                    if source.get("effective_date"):
                        st.markdown(f"**Effective Date:** {source['effective_date'][:10]}")
                    st.markdown("---")
                    st.markdown(text[:2000] + ("..." if len(text) > 2000 else ""))

        # Citations section
        citations = result.get("citations", [])
        if citations and citations != sources:
            st.divider()
            st.subheader("📎 Citations")
            for cit in citations:
                st.markdown(f"- {cit.get('citation', cit.get('doc_id'))}")

        # Stage timings (debug info)
        stage_timings = result.get("stage_timings")
        if stage_timings:
            with st.expander("⏱️ Performance Breakdown", expanded=False):
                for stage, timing in stage_timings.items():
                    st.markdown(f"- **{stage}:** {timing:.0f}ms")

    elif submitted:
        st.warning("Please enter a question.")

    # Footer
    st.divider()
    st.caption(
        "Florida Tax Law Assistant | Powered by RAG | "
        "This tool provides general information and should not be considered legal advice."
    )


if __name__ == "__main__":
    main()
