"""
Utility functions for Embed Service: text chunking, token estimation, and similarity math.
"""

import re
import math
import hashlib
from typing import List, Dict, Any
import numpy as np


def clean_text(text: str) -> str:
    """Normalize whitespace and clean up raw string input."""
    if not text:
        return ""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Approximate token count (1 token ~= 4 characters / 0.75 words in English)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(1, int((words * 1.3 + chars / 4.0) / 2))


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks respecting paragraph and sentence boundaries.
    
    Returns a list of dicts containing:
      - chunk_index: int
      - content: str
      - token_count: int
      - char_count: int
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # First split by double newlines (paragraphs)
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0

    for para in paragraphs:
        # If paragraph itself is too large, split by sentences
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sent_len = len(sentence)
                if current_length + sent_len > chunk_size and current_chunk:
                    chunk_str = " ".join(current_chunk).strip()
                    chunks.append({
                        "chunk_index": chunk_idx,
                        "content": chunk_str,
                        "token_count": estimate_tokens(chunk_str),
                        "char_count": len(chunk_str)
                    })
                    chunk_idx += 1
                    # Keep overlap from the tail of previous chunk
                    overlap_chars = 0
                    overlap_chunk = []
                    for s in reversed(current_chunk):
                        if overlap_chars + len(s) <= chunk_overlap:
                            overlap_chunk.insert(0, s)
                            overlap_chars += len(s)
                        else:
                            break
                    current_chunk = overlap_chunk + [sentence]
                    current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                else:
                    current_chunk.append(sentence)
                    current_length += sent_len + 1
        else:
            para_len = len(para)
            if current_length + para_len > chunk_size and current_chunk:
                chunk_str = "\n\n".join(current_chunk).strip()
                chunks.append({
                    "chunk_index": chunk_idx,
                    "content": chunk_str,
                    "token_count": estimate_tokens(chunk_str),
                    "char_count": len(chunk_str)
                })
                chunk_idx += 1
                current_chunk = [para]
                current_length = para_len
            else:
                current_chunk.append(para)
                current_length += para_len + 2

    if current_chunk:
        chunk_str = "\n\n".join(current_chunk).strip()
        if chunk_str:
            chunks.append({
                "chunk_index": chunk_idx,
                "content": chunk_str,
                "token_count": estimate_tokens(chunk_str),
                "char_count": len(chunk_str)
            })

    return chunks


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two 1D vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def generate_fallback_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Generate high-entropy deterministic 384-dimensional dense pseudo-embedding for testing
    when PyTorch or transformer models are in fallback mode.
    """
    cleaned = clean_text(text).lower()
    words = cleaned.split()
    vector = np.zeros(dim, dtype=np.float32)

    for i, word in enumerate(words):
        h = int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
        weight = 1.0 / math.sqrt(i + 1)
        vector[idx] += sign * weight

    # Normalize vector to unit length
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    else:
        vector[0] = 1.0
    return vector.tolist()
