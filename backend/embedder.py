"""
Embedder module for transforming text chunks into vector embeddings.

Uses FAISS for efficient similarity search and sentence-transformers for embeddings.

Version 2.0:
- Lazy model loading to improve startup time
- Better error handling
- Improved search result formatting
"""

from typing import List, Dict, Tuple, Optional
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Model configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Global model instance (lazy loaded)
_model: Optional[SentenceTransformer] = None


@st.cache_resource(show_spinner="Loading embedding model...")
def _load_model() -> SentenceTransformer:
    """Load the sentence transformer model with caching."""
    return SentenceTransformer(MODEL_NAME)


def get_model() -> SentenceTransformer:
    """Get or load the embedding model (lazy loading)."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generate vector embeddings for a list of text chunks.

    Args:
        texts: A list of strings (chunks)

    Returns:
        A NumPy array of shape (len(texts), embedding_dim)
    """
    model = get_model()
    return np.array(
        model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=False,
            batch_size=32
        )
    )


def build_index(
    chunks: List[Dict[str, str]],
    existing_index: Optional[faiss.IndexFlatIP] = None,
    docs_meta: Optional[Dict[str, List[Dict]]] = None,
    file_name: str = "default"
) -> Tuple[faiss.IndexFlatIP, Dict[str, List[Dict]]]:
    """
    Build or update a FAISS index from text chunks.

    Args:
        chunks: List of dicts with 'text' key
        existing_index: Existing FAISS index to update (optional)
        docs_meta: Metadata associated with the index (optional)
        file_name: Document name for metadata

    Returns:
        Tuple of (updated FAISS index, metadata dictionary)
    """
    if not chunks:
        raise ValueError("No chunks provided for indexing")
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    if existing_index is None:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    else:
        index = existing_index

    # Add embeddings to index
    index.add(embeddings.astype(np.float32))

    # Initialize metadata dict if needed
    if docs_meta is None:
        docs_meta = {}

    if file_name not in docs_meta:
        docs_meta[file_name] = []

    # Store metadata for each chunk
    base_idx = len(docs_meta[file_name])
    for i, chunk in enumerate(chunks):
        docs_meta[file_name].append({
            "chunk_id": base_idx + i,
            "text": chunk["text"],
            "page": chunk.get("page", 1),
            "metadata": chunk.get("metadata", {})
        })

    return index, docs_meta


def search(
    index: faiss.IndexFlatIP,
    query: str,
    docs_meta: Dict[str, List[Dict]],
    file_name: str,
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Perform semantic search over indexed document chunks.

    Args:
        index: A FAISS index with document embeddings
        query: The user search query
        docs_meta: Metadata associated with document chunks
        file_name: Target document to search within
        top_k: Number of top matches to return

    Returns:
        List of matched chunks with score, page, and excerpt
    """
    if index is None or index.ntotal == 0:
        return []
    
    if file_name not in docs_meta:
        return []
    
    # Generate query embedding
    query_vector = embed_texts([query]).astype(np.float32)
    
    # Search index
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    # Build results
    results = []
    doc_chunks = docs_meta[file_name]
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(doc_chunks):
            continue
            
        chunk_meta = doc_chunks[idx]
        text = chunk_meta["text"]
        
        # Truncate long excerpts
        excerpt = text[:500] + "..." if len(text) > 500 else text
        
        results.append({
            "score": round(float(score), 4),
            "page": chunk_meta["page"],
            "excerpt": excerpt,
            "chunk_id": chunk_meta["chunk_id"]
        })

    return results
