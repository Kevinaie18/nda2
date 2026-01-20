"""
Shared utility functions for NDA Analyzer.

This module contains common helper functions used across the application.
"""

import re
from typing import Optional


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove control characters.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to max_length, preserving word boundaries.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: String to append when truncating
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length - len(suffix)]
    
    # Try to break at word boundary
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.5:
        truncated = truncated[:last_space]
    
    return truncated + suffix


def extract_number(text: str, default: int = 50) -> int:
    """
    Extract first number from text, with default fallback.
    
    Args:
        text: Text containing a number
        default: Default value if no number found
        
    Returns:
        Extracted number or default
    """
    match = re.search(r'\b(\d{1,3})\b', text)
    return int(match.group(1)) if match else default


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe filesystem operations.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and null bytes
    filename = re.sub(r'[/\\:\x00]', '_', filename)
    # Remove other potentially dangerous characters
    filename = re.sub(r'[<>"|?*]', '_', filename)
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def format_risk_level(score: int) -> tuple[str, str]:
    """
    Format risk score into level and color.
    
    Args:
        score: Risk score (0-100)
        
    Returns:
        Tuple of (level name, color emoji)
    """
    if score >= 70:
        return "High", "🔴"
    elif score >= 40:
        return "Medium", "🟡"
    else:
        return "Low", "🟢"
