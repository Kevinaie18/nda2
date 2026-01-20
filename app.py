"""
NDA Analyzer - Main Streamlit Application

A tool for analyzing Non-Disclosure Agreements using AI.
Updated: January 2025
"""

import os
from typing import Any, Dict, List
from pathlib import Path

import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from backend.loader import load_document, validate_upload
from backend.embedder import build_index, search
from backend.analyzer import run_full_analysis

load_dotenv()

# === Constants & Model Mapping ===
SUPPORTED_FORMATS = [".pdf", ".docx"]

# Updated models - January 2025
AVAILABLE_MODELS = ["DeepSeek V3.2", "DeepSeek V3.1", "DeepSeek R1"]
MODEL_MAP = {
    "DeepSeek V3.2": "accounts/fireworks/models/deepseek-v3p2",
    "DeepSeek V3.1": "accounts/fireworks/models/deepseek-v3p1",
    "DeepSeek R1": "accounts/fireworks/models/deepseek-r1-0528",
}
DEFAULT_MODEL = "DeepSeek V3.2"
DEFAULT_TOP_K = 8


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "index" not in st.session_state:
        st.session_state.index = None
    if "docs_meta" not in st.session_state:
        st.session_state.docs_meta = {}
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}


def display_usage_instructions():
    """Display welcome message and usage instructions."""
    st.title("📄 NDA Analyzer")
    st.markdown("""
### Welcome to NDA Analyzer!

This tool helps you analyze Non-Disclosure Agreements using AI. Here's how to use it:

1. **Upload Documents**: Use the sidebar to upload one or more NDA documents (PDF or DOCX)  
2. **Configure Analysis**: Adjust the analysis parameters in the sidebar  
3. **Run Analysis**: Click the "Run Analysis" button to process your documents  
4. **Review Results**: View the summary, risk score, and extracted clauses  
5. **Search**: Use the semantic search feature to find specific information  

Get started by uploading your documents in the sidebar!
    """)


def display_results(file_name: str):
    """Display analysis results for a document."""
    result = st.session_state.analysis_results[file_name]
    st.markdown(f"### Results for `{file_name}`")
    
    # 1. Summary
    st.markdown("**Executive Summary**")
    st.write(result["summary"])
    
    # 2. Risk score
    risk_score = result["risk_score"]
    risk_color = "🔴" if risk_score >= 70 else "🟡" if risk_score >= 40 else "🟢"
    st.metric("Risk Score", f"{risk_color} {risk_score}/100")
    
    # 3. Clauses
    clauses = result["clauses"]
    if clauses:
        clauses_df = pd.DataFrame(clauses)
        
        # Warning if placeholder
        if clauses[0].get("clause_type") == "Unknown":
            st.warning("⚠️ Clause extraction failed, using placeholder.")
        
        st.markdown("**Critical Clauses**")
        
        # Configure column display
        column_config = {
            "clause_type": "Clause Type",
            "risk_level": "Risk Level",
            "page": "Page",
            "excerpt": "Excerpt",
            "justification": "Justification",
        }
        
        # Add recommendations column if present
        if "recommendations" in clauses_df.columns:
            column_config["recommendations"] = "Recommendations"
        
        st.dataframe(
            clauses_df,
            column_config=column_config,
            hide_index=True
        )
        
        # 4. Export CSV
        csv_data = clauses_df.to_csv(index=False)
        st.download_button(
            "📥 Download clauses CSV",
            csv_data,
            file_name=f"clauses_{file_name}.csv",
            mime="text/csv"
        )
    
    # 5. Risk reasoning
    if "risk_reasoning" in result and result["risk_reasoning"]:
        with st.expander("📊 Risk Assessment Details"):
            st.write(result["risk_reasoning"])


def display_search_interface():
    """Display semantic search interface."""
    st.markdown("---")
    st.markdown("### 🔍 Semantic Search")
    
    if not st.session_state.docs_meta:
        st.info("Upload and analyze documents to enable search.")
        return
    
    available_docs = list(st.session_state.docs_meta.keys())
    selected_doc = st.selectbox("Select document to search", available_docs)
    
    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of results", 1, 10, 5)
    
    if st.button("Search") and query and selected_doc:
        with st.spinner("Searching..."):
            results = search(
                st.session_state.index,
                query,
                st.session_state.docs_meta,
                selected_doc,
                top_k
            )
            
            if results:
                for i, res in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {res['score']}, Page: {res['page']})"):
                        st.write(res["excerpt"])
            else:
                st.warning("No results found.")


def process_documents(uploaded_files, model_name: str, temperature: float, top_k: int):
    """Process uploaded documents through the analysis pipeline."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    
    for idx, up in enumerate(uploaded_files):
        name = up.name
        ext = Path(name).suffix.lower()
        
        if ext not in SUPPORTED_FORMATS:
            st.error(f"Unsupported format: {ext}")
            continue
        
        try:
            # Validate upload
            status_text.text(f"Validating {name}...")
            validate_upload(up)
            
            # Load document
            status_text.text(f"Loading {name}...")
            progress_bar.progress((idx * 3 + 1) / (total_files * 3))
            chunks = load_document(up)
            
            if not chunks:
                st.error(f"No content extracted from {name}")
                continue
            
            # Build index
            status_text.text(f"Building index for {name}...")
            progress_bar.progress((idx * 3 + 2) / (total_files * 3))
            st.session_state.index, st.session_state.docs_meta = build_index(
                chunks, st.session_state.index, st.session_state.docs_meta, name
            )
            
            # Run analysis
            status_text.text(f"Analyzing {name}...")
            api_model = MODEL_MAP.get(model_name, MODEL_MAP[DEFAULT_MODEL])
            res = run_full_analysis(chunks, api_model, temperature, top_k)
            st.session_state.analysis_results[name] = res
            
            progress_bar.progress((idx + 1) / total_files)
            
        except Exception as e:
            st.error(f"Error processing {name}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    st.success("Analysis complete!")


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="NDA Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    initialize_session_state()
    
    # API key check
    api_key = st.secrets.get("fireworks", {}).get("api_key") or os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        st.error("🔑 FIREWORKS_API_KEY is not set. Edit your Streamlit secrets or .env file.")
        st.info("Get your API key at: https://fireworks.ai/")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    uploaded = st.sidebar.file_uploader(
        "Upload NDA(s)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Upload PDF or DOCX files (max 10MB each)"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Model Settings**")
    
    model = st.sidebar.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL),
        help="DeepSeek V3.2 is recommended for general use. R1 is better for complex reasoning."
    )
    
    temp = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, 0.0, 0.1,
        help="Lower = more deterministic, Higher = more creative"
    )
    
    top_k = st.sidebar.number_input(
        "Top K chunks",
        1, 20, DEFAULT_TOP_K,
        help="Number of document chunks to analyze"
    )
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary") and uploaded:
        process_documents(uploaded, model, temp, top_k)
    
    # Main content area
    if not uploaded:
        display_usage_instructions()
        return
    
    # Display results
    if st.session_state.analysis_results:
        tabs = st.tabs(list(st.session_state.analysis_results.keys()) + ["🔍 Search"])
        
        for i, fname in enumerate(st.session_state.analysis_results):
            with tabs[i]:
                display_results(fname)
        
        with tabs[-1]:
            display_search_interface()
    else:
        st.info("👆 Click 'Run Analysis' in the sidebar to analyze your documents.")


if __name__ == "__main__":
    main()
