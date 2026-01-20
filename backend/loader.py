"""
Document loader module for processing PDF and DOCX files.

This module provides functionality to load and chunk documents from various formats
into a format suitable for embedding and analysis.

Version 2.0:
- Added file validation (size, content)
- Improved chunking with overlap for better context
- Better error handling
"""

from typing import List, Dict, Any, BinaryIO
import re
from pathlib import Path

import PyPDF2
from docx import Document

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters


def validate_upload(file: BinaryIO) -> None:
    """
    Validate uploaded file before processing.
    
    Args:
        file: File-like object to validate
        
    Raises:
        ValueError: If file is invalid (too large, empty, etc.)
    """
    # Check file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if size > MAX_FILE_SIZE:
        raise ValueError(f"File too large: {size / 1024 / 1024:.1f}MB (max {MAX_FILE_SIZE / 1024 / 1024:.0f}MB)")
    
    if size == 0:
        raise ValueError("File is empty")


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary within last 20% of chunk
        if end < len(text):
            search_start = end - int(chunk_size * 0.2)
            
            # Look for sentence endings
            for sep in ['. ', '.\n', '? ', '!\n', '\n\n']:
                last_sep = text.rfind(sep, search_start, end)
                if last_sep > search_start:
                    end = last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start, accounting for overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks


def load_document(file: BinaryIO) -> List[Dict[str, Any]]:
    """
    Load and chunk a document from a file object.
    
    Args:
        file: A file-like object containing the document
        
    Returns:
        List of document chunks, each containing:
        - text: The chunk text
        - page: Page number
        - metadata: Additional metadata
        
    Raises:
        ValueError: If file format is unsupported
    """
    file_ext = Path(file.name).suffix.lower()
    
    if file_ext == ".pdf":
        return _load_pdf(file)
    elif file_ext == ".docx":
        return _load_docx(file)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def _load_pdf(file: BinaryIO) -> List[Dict[str, Any]]:
    """
    Load and chunk a PDF file.
    
    Args:
        file: PDF file object
        
    Returns:
        List of document chunks with page numbers
    """
    chunks = []
    
    try:
        pdf_reader = PyPDF2.PdfReader(file)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {str(e)}")
    
    for page_num in range(len(pdf_reader.pages)):
        try:
            page = pdf_reader.pages[page_num]
            text = page.extract_text() or ""
        except Exception:
            continue
        
        # Clean the text
        text = clean_text(text)
        
        if not text:
            continue
        
        # Chunk the page text
        page_chunks = chunk_text(text)
        
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "text": chunk,
                "page": page_num + 1,
                "metadata": {
                    "chunk_id": f"p{page_num + 1}_{i}",
                    "source": file.name,
                    "type": "pdf"
                }
            })
    
    return chunks


def _load_docx(file: BinaryIO) -> List[Dict[str, Any]]:
    """
    Load and chunk a DOCX file.
    
    Args:
        file: DOCX file object
        
    Returns:
        List of document chunks with estimated page numbers
    """
    chunks = []
    
    try:
        doc = Document(file)
    except Exception as e:
        raise ValueError(f"Failed to read DOCX: {str(e)}")
    
    # Collect all text
    all_paragraphs = []
    for para in doc.paragraphs:
        text = clean_text(para.text)
        if text:
            all_paragraphs.append(text)
    
    # Join paragraphs and chunk
    full_text = "\n\n".join(all_paragraphs)
    text_chunks = chunk_text(full_text)
    
    # Estimate page numbers (rough: ~3000 chars per page)
    chars_per_page = 3000
    
    current_pos = 0
    for i, chunk in enumerate(text_chunks):
        page_num = (current_pos // chars_per_page) + 1
        
        chunks.append({
            "text": chunk,
            "page": page_num,
            "metadata": {
                "chunk_id": f"chunk_{i}",
                "source": file.name,
                "type": "docx"
            }
        })
        
        current_pos += len(chunk)
    
    return chunks
