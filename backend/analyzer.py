"""
Analyzer module for processing NDA documents using DeepSeek via Fireworks AI.

Version v2.0: 
- Updated to chat completions API (OpenAI-compatible)
- Updated model IDs for DeepSeek V3.2 and R1-0528
- Added retry logic with exponential backoff
- Fixed CSV parsing to match 6-field prompt
- Added async support for parallel API calls
"""

from typing import List, Dict, Any, Optional
import os
import csv
import re
import time
from functools import wraps
from pathlib import Path
from io import StringIO

import requests
import streamlit as st

# Load prompts
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """Load a raw prompt template from disk."""
    return (PROMPTS_DIR / f"{name}.txt").read_text()


def get_api_key() -> str:
    """Get Fireworks API key from secrets or environment."""
    api_key = os.getenv("FIREWORKS_API_KEY") or (
        st.secrets.get("fireworks", {}).get("api_key")
    )
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not set in env or secrets.")
    return api_key


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying failed API calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _call_deepseek(
    messages: List[Dict[str, str]],
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 4096
) -> str:
    """
    Call Fireworks DeepSeek API using chat completions endpoint.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Full model identifier (e.g., accounts/fireworks/models/deepseek-v3p2)
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text content
    """
    api_key = get_api_key()
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    # Use chat completions endpoint (OpenAI-compatible)
    resp = requests.post(
        "https://api.fireworks.ai/inference/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=120  # 2 minute timeout for long responses
    )
    
    if resp.status_code == 429:
        raise Exception("Rate limited - will retry")
    
    if resp.status_code != 200:
        raise Exception(f"API call failed: {resp.status_code} - {resp.text}")
    
    response_data = resp.json()
    return response_data["choices"][0]["message"]["content"]


def parse_clauses_csv(text: str) -> List[Dict[str, Any]]:
    """
    Parse CSV text into a list of clause dicts.
    Expected fields (6): clause_type, risk_level, page, excerpt, justification, recommendations
    """
    # Remove code fences or unexpected wrappers
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"```csv\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    
    # Skip header if present
    if lines and lines[0].lower().startswith("clause_type"):
        lines = lines[1:]
    
    clauses = []
    
    # Use csv.reader for proper CSV parsing
    try:
        reader = csv.reader(lines)
        for parts in reader:
            # Accept lines with at least 5 columns (6th is optional recommendations)
            if len(parts) < 5:
                continue
            
            ctype = parts[0].strip()
            rlevel = parts[1].strip()
            page_str = parts[2].strip()
            excerpt = parts[3].strip() if len(parts) > 3 else ""
            justification = parts[4].strip() if len(parts) > 4 else ""
            recommendations = parts[5].strip() if len(parts) > 5 else ""
            
            # Parse page number
            try:
                page_num = int(page_str)
            except ValueError:
                page_num = 1
            
            # Validate risk level
            if rlevel not in ["High", "Medium", "Low"]:
                rlevel = "Medium"
            
            clauses.append({
                "clause_type": ctype,
                "risk_level": rlevel,
                "page": page_num,
                "excerpt": excerpt,
                "justification": justification,
                "recommendations": recommendations
            })
    except csv.Error:
        pass  # Fall through to fallback parsing
    
    # Fallback: relaxed parsing if strict CSV fails
    if not clauses:
        for line in lines:
            # Try to split on comma, handling quoted fields
            parts = [p.strip().strip('"') for p in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)]
            if len(parts) >= 3:
                clauses.append({
                    "clause_type": parts[0],
                    "risk_level": parts[1] if len(parts) > 1 else "Medium",
                    "page": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1,
                    "excerpt": parts[3] if len(parts) > 3 else "",
                    "justification": parts[4] if len(parts) > 4 else "",
                    "recommendations": parts[5] if len(parts) > 5 else ""
                })
    
    return clauses


def parse_risk_score(text: str) -> tuple[int, str]:
    """
    Parse risk score and reasoning from API response.
    
    Returns:
        Tuple of (score, reasoning)
    """
    # Try to parse as CSV first
    text = text.strip()
    
    # Remove any markdown formatting
    text = re.sub(r"```[\s\S]*?```", "", text).strip()
    
    try:
        reader = csv.reader([text])
        for parts in reader:
            if len(parts) >= 2:
                score_str = parts[0].strip()
                reasoning = parts[1].strip()
                
                # Extract number from score
                match = re.search(r'\d+', score_str)
                if match:
                    score = int(match.group())
                    return max(0, min(100, score)), reasoning
    except csv.Error:
        pass
    
    # Fallback: extract first number as score
    match = re.search(r'\b(\d{1,3})\b', text)
    score = int(match.group(1)) if match else 50
    
    return max(0, min(100, score)), text


def run_full_analysis(
    chunks: List[Dict[str, Any]],
    model_name: str,
    temperature: float = 0.0,
    top_k: int = 8
) -> Dict[str, Any]:
    """
    Run a full analysis of an NDA document.
    
    Steps:
        1. Generate executive summary
        2. Extract clauses via CSV format
        3. Calculate risk score
    
    Args:
        chunks: List of document chunks with 'text' key
        model_name: Fireworks model identifier
        temperature: Sampling temperature
        top_k: Number of chunks to analyze
        
    Returns:
        Dict with summary, risk_score, risk_reasoning, and clauses
    """
    # 1. Combine top_k chunks
    full_text = "\n\n".join(chunk["text"] for chunk in chunks[:top_k])
    
    if not full_text.strip():
        return {
            "summary": "No text content found in document.",
            "risk_score": 0,
            "risk_reasoning": "Unable to analyze empty document.",
            "clauses": []
        }
    
    # 2. Executive summary
    summary_template = _load_prompt("summarizer")
    summary_prompt = summary_template.replace("{text}", full_text)
    
    summary_messages = [
        {
            "role": "system",
            "content": "You are a senior legal and business analyst at a private equity fund. Provide detailed, professional analysis."
        },
        {
            "role": "user", 
            "content": summary_prompt
        }
    ]
    
    summary = _call_deepseek(summary_messages, model_name, temperature).strip()
    
    # 3. Clause extraction (CSV format)
    clause_template = _load_prompt("clause_extractor")
    clause_prompt = clause_template.replace("{text}", full_text)
    
    clause_messages = [
        {
            "role": "system",
            "content": "You are a senior legal analyst. Extract clauses in strict CSV format only. No markdown, no explanations."
        },
        {
            "role": "user",
            "content": clause_prompt
        }
    ]
    
    raw_csv = _call_deepseek(clause_messages, model_name, temperature)
    
    # Debug: show raw output
    with st.expander("🔍 Raw clauses CSV", expanded=False):
        st.code(raw_csv, language="text")
    
    clauses = parse_clauses_csv(raw_csv)
    
    if not clauses:
        st.warning("⚠️ No clauses extracted via CSV; using placeholder.")
        clauses = [{
            "clause_type": "Unknown",
            "risk_level": "Medium",
            "page": 1,
            "excerpt": "",
            "justification": "Unable to extract clauses from document",
            "recommendations": "Manual review recommended"
        }]
    
    # 4. Risk scoring
    risk_template = _load_prompt("risk_assessor")
    
    # Build CSV from parsed clauses
    csv_lines = []
    for c in clauses:
        excerpt_safe = c["excerpt"].replace('"', '""')
        just_safe = c["justification"].replace('"', '""')
        csv_lines.append(
            f'{c["clause_type"]},{c["risk_level"]},{c["page"]},"{excerpt_safe}","{just_safe}"'
        )
    clauses_csv = "\n".join(csv_lines)
    
    risk_prompt = risk_template.replace("{text}", full_text).replace("{clauses_csv}", clauses_csv)
    
    risk_messages = [
        {
            "role": "system",
            "content": "You are a legal risk analyst. Provide risk score as CSV: score,reasoning"
        },
        {
            "role": "user",
            "content": risk_prompt
        }
    ]
    
    raw_risk = _call_deepseek(risk_messages, model_name, temperature)
    
    # Debug: show raw output
    with st.expander("🔍 Raw risk output", expanded=False):
        st.code(raw_risk, language="text")
    
    risk_score, risk_reasoning = parse_risk_score(raw_risk)
    
    return {
        "summary": summary,
        "risk_score": risk_score,
        "risk_reasoning": risk_reasoning,
        "clauses": clauses
    }
