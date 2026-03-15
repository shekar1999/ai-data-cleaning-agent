import streamlit as st
import pandas as pd

from utils.loader import DataLoader
from utils.profiler import DatasetProfiler
from utils.column_intelligence import ColumnIntelligenceEngine
from utils.missing_value_policy import MissingValuePolicyEngine
from utils.duplicate_policy import DuplicatePolicyEngine
from utils.cleaner import CleaningActionEngine
from utils.anomaly import AnomalyDetectionEngine
from utils.explainability import ExplainabilityEngine
from utils.insights import InsightsEngine

st.set_page_config(page_title="AI Data Quality Engine", page_icon="🧠", layout="wide")

def main():
    # --- Custom CSS Injection ---
    st.markdown("""
        <style>
        /* Sleek typography & neon accents */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }
        
        /* Main glowing title */
        h1 {
            background: linear-gradient(90deg, #a855f7, #ec4899, #ef4444);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
            letter-spacing: -1.5px;
            margin-bottom: 0.2rem;
        }

        /* Metric Cards */
        [data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* Neon Button */
        .stButton>button {
            background: linear-gradient(135deg, #6366f1, #a855f7) !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            box-shadow: 0 4px 15px -3px rgba(99, 102, 241, 0.5) !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(168, 85, 247, 0.6) !important;
        }

        /* Sidebar upload area styling */
        [data-testid="stSidebar"] {
            border-right: 1px solid #334155 !important;
        }
        
        /* DataFrame Tables */
        .dataframe {
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧠 AI Data Quality Engine")
    st.markdown("<p style='color: #cbd5e1; font-size: 1.1rem; margin-bottom: 2rem;'>An autonomous, intelligence-driven engine that understands your data semantics before safely cleaning it.</p>", unsafe_allow_html=True)

    # 1. Upload
    st.sidebar.header("1. Upload Settings")
    has_header = st.sidebar.checkbox("File contains header row", value=True)
    manual_delimiter = st.sidebar.selectbox(
        "Delimiter for TXT / DATA / TSV files",
        options=["Auto Detect", "Comma (,)", "Tab (\\t)", "Pipe (|)", "Semicolon (;)", "Space"],
        index=0
    )
    uploaded_file = st.sidebar.file_uploader("Upload structured dataset", type=["csv", "xlsx", "txt", "tsv", "json", "data"])

    if not uploaded_file:
        st.info("Please upload a dataset from the sidebar to begin autonomous profiling.")
        return

    # 2. Parse
    loader = DataLoader()
    raw_df, parse_report = loader.parse_file(uploaded_file, uploaded_file.name, manual_delimiter, has_header)
    
    if raw_df is None:
        st.error("Failed to parse the file.")
        for warn in parse_report["warnings"]:
            st.error(warn)
        return

    st.header("1. Parsing Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Parsing Confidence", parse_report["parsing_confidence"])
    col2.metric("Detected Delimiter", repr(parse_report.get("detected_delimiter", "N/A")))
    col3.metric("Detected Encoding", parse_report.get("encoding", "N/A"))
    
    if parse_report["warnings"]:
        st.warning("Warnings during parsing:")
        for w in parse_report["warnings"]:
            st.write(f"- {w}")

    # 3. Profile & Intelligence
    st.header("2. Dataset Understanding & Intelligence")
    with st.spinner("Profiling dataset and inferring column semantics..."):
        profiler = DatasetProfiler()
        initial_profile = profiler.profile(raw_df)
        
        intel_engine = ColumnIntelligenceEngine()
        intelligence = intel_engine.infer_semantics(raw_df, initial_profile)
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", initial_profile["row_count"])
    m2.metric("Columns", initial_profile["column_count"])
    m3.metric("Missing Values", initial_profile["total_missing"])
    m4.metric("Duplicate Rows", initial_profile["total_duplicate_rows"])

    st.subheader("Inferred Semantic Types")
    intel_data = []
    for col, intel in intelligence.items():
        intel_data.append({
            "Column": col,
            "Inferred Type": intel["semantic_type"],
            "Confidence": intel["confidence"],
            "Modification Risk": "High" if intel["risk_score"] >= 0.8 else ("Medium" if intel["risk_score"] > 0.4 else "Low"),
            "Reasoning": intel["reasoning"]
        })
    st.dataframe(pd.DataFrame(intel_data), use_container_width=True)

    # 4. Granular Safety Policy Generation
    st.header("3. Safety Protocols & Overrides")
    missing_engine = MissingValuePolicyEngine()
    dup_engine = DuplicatePolicyEngine()
    
    missing_policies = {}
    for col in intelligence.keys():
        stats = initial_profile["columns"].get(col, {})
        missing_policies[col] = missing_engine.generate_policy(col, intelligence[col], stats)
        
    dup_policy = dup_engine.generate_global_policy(intelligence, initial_profile)
    
    st.markdown("**Duplicate Handling Policy:**")
    st.info(" ".join(dup_policy["reasoning"]))

    st.markdown("**Column Modification Risk Protocols:**")
    for col, intel in intelligence.items():
        m_pol = missing_policies.get(col, {})
        if intel["risk_score"] > 0.8:
            st.error(f"**{col}**: {intel['semantic_type'].upper()} (CRITICAL RISK). Reason: {m_pol.get('reasoning', ['Protected'])[0]}")
        elif m_pol.get("action") == "preserve":
            st.warning(f"**{col}**: PRESERVED. Reason: {m_pol.get('reasoning', ['Intentional skip'])[0]}")

    # Settings Override
    st.subheader("Manual Overrides")
    force_drop_dupes = st.checkbox("Recklessly force drop all exact duplicates (Overrides safe deduplication)", value=False)

    # 5. Clean & Explain
    if st.button("Execute Safe Cleaning", type="primary"):
        with st.spinner("Applying safety-controlled transformations..."):
            cleaner = CleaningActionEngine()
            user_overrides = {"force_drop_duplicates": force_drop_dupes}
            cleaned_df, action_log = cleaner.clean(raw_df, intelligence, missing_policies, dup_policy, user_overrides)
            
            anomaly_engine = AnomalyDetectionEngine()
            # Feed numeric columns that aren't IDs directly
            numeric_cols = [c for c, i in intelligence.items() if i["semantic_type"] == "numeric_measure"]
            anomalies, anomaly_meta = None, []
            
            if numeric_cols:
                 numeric_df = cleaned_df[numeric_cols].copy()
                 anomalies, anomaly_meta = anomaly_engine.detect_anomalies(numeric_df, cleaned_df)
            
            final_profile = profiler.profile(cleaned_df)
            
            explain_engine = ExplainabilityEngine()
            report = explain_engine.generate_report(initial_profile, final_profile, intelligence, missing_policies, dup_policy, action_log)
            
            insights_engine = InsightsEngine()

        st.success("Cleaning Completed Successfully.")

        st.header("4. Cleaning Results & Explainability")
        st.info(insights_engine.get_llm_placeholder_summary(report))
        
        # Before / After
        c1, c2 = st.columns(2)
        c1.metric("Missing Values (Before)", report["summary"]["total_missing_before"])
        c2.metric("Missing Values (After)", report["summary"]["total_missing_after"])
        
        c3, c4 = st.columns(2)
        c3.metric("Rows (Before)", report["summary"]["rows_before"])
        c4.metric("Rows (After)", report["summary"]["rows_after"])
        
        st.subheader("Action Log")
        if action_log:
            st.table(pd.DataFrame(action_log))
        else:
            st.write("No modifications were deemed necessary based on the intelligent policies.")
            
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(cleaned_df.head(20), use_container_width=True)
        
        if anomalies is not None and not anomalies.empty:
            st.warning("⚠️ Statistical Anomalies Detected (Not Removed)")
            for m in anomaly_meta:
                 st.write(m)
            st.dataframe(anomalies.head(20), use_container_width=True)

        st.header("5. Download Center")
        csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Cleaned Dataset", data=csv_data, file_name=f"cleaned_{uploaded_file.name}.csv", mime="text/csv")
        
    st.markdown("---")
    st.markdown("*LLM integration module is prepared in `insights.py` for future natural-language extensions.*")

if __name__ == "__main__":
    main()