"""
Rerank Service: FastAPI microservice for deep cross-encoder reranking of retrieved passages.
Default Model: cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import os
import re
import time
import math
import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rerank-service")

MODEL_NAME = os.getenv("MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
cross_encoder_instance = None
model_loaded = False


def sigmoid(x: float) -> float:
    """Standard sigmoid function for mapping logits to 0..1 probabilities."""
    return 1.0 / (1.0 + math.exp(-max(min(x, 15.0), -15.0)))


def fallback_bm25_rerank(query: str, texts: List[str]) -> List[float]:
    """
    High-precision BM25/TF-IDF lexical relevance scorer used when deep learning models
    are in fallback mode.
    """
    query_tokens = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 1]
    if not query_tokens:
        return [0.5] * len(texts)

    scores = []
    doc_token_lists = [[w.lower() for w in re.findall(r'\w+', t)] for t in texts]
    total_docs = len(texts)

    # Document frequency
    df = {}
    for qt in query_tokens:
        df[qt] = sum(1 for d in doc_token_lists if qt in d)

    avg_dl = sum(len(d) for d in doc_token_lists) / max(1, total_docs)
    k1 = 1.5
    b = 0.75

    for doc_tokens in doc_token_lists:
        doc_len = len(doc_tokens)
        score = 0.0
        # Exact phrase bonus
        doc_str = " ".join(doc_tokens)
        query_str = " ".join(query_tokens)
        if query_str in doc_str:
            score += 2.0

        for qt in query_tokens:
            tf = doc_tokens.count(qt)
            if tf > 0:
                doc_freq = df.get(qt, 0)
                idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / max(1.0, avg_dl))))
                score += term_score
        scores.append(score)

    # Normalize to 0.1 - 0.99
    max_s = max(scores) if scores and max(scores) > 0 else 1.0
    normalized = [min(0.99, max(0.05, s / max_s)) for s in scores]
    return normalized


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cross_encoder_instance, model_loaded
    logger.info(f"Initializing Rerank Service with model: {MODEL_NAME}")
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder model...")
        cross_encoder_instance = CrossEncoder(MODEL_NAME)
        model_loaded = True
        logger.info("CrossEncoder model loaded successfully!")
    except Exception as e:
        logger.warning(f"Could not load CrossEncoder directly ({e}). Running in resilient fallback mode.")
        model_loaded = False
    yield
    logger.info("Shutting down Rerank Service.")


app = FastAPI(
    title="Verified RAG - Rerank Service",
    description="Cross-Encoder Deep Reranking Microservice",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class PassageItem(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None
    initial_score: Optional[float] = None
    initial_rank: Optional[int] = None


class RerankRequest(BaseModel):
    query: str = Field(..., description="The user question or retrieval query")
    passages: List[Union[PassageItem, str]] = Field(..., description="List of passages (objects or strings)")
    top_n: Optional[int] = Field(default=None, description="Number of top reranked passages to return")


class RerankResultItem(BaseModel):
    id: Optional[str] = None
    content: str
    rerank_score: float
    confidence_score: float
    initial_rank: Optional[int] = None
    new_rank: int
    rank_delta: int
    metadata: Optional[Dict[str, Any]] = None


class RerankResponse(BaseModel):
    query: str
    results: List[RerankResultItem]
    total_input_passages: int
    returned_passages: int
    model: str
    latency_ms: float
    is_fallback: bool


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "rerank-service",
        "model": MODEL_NAME,
        "model_loaded": model_loaded
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank_passages(payload: RerankRequest):
    start_time = time.time()
    query = payload.query.strip()
    raw_passages = payload.passages

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not raw_passages:
        return RerankResponse(
            query=query,
            results=[],
            total_input_passages=0,
            returned_passages=0,
            model=MODEL_NAME,
            latency_ms=0.0,
            is_fallback=False
        )

    # Normalize passages to standard structure
    normalized_passages: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_passages):
        if isinstance(item, str):
            normalized_passages.append({
                "id": str(idx),
                "content": item,
                "metadata": {},
                "initial_rank": idx + 1,
                "initial_score": None
            })
        elif isinstance(item, PassageItem):
            normalized_passages.append({
                "id": item.id or str(idx),
                "content": item.content,
                "metadata": item.metadata or {},
                "initial_rank": item.initial_rank or (idx + 1),
                "initial_score": item.initial_score
            })
        elif isinstance(item, dict):
            normalized_passages.append({
                "id": item.get("id", str(idx)),
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
                "initial_rank": item.get("initial_rank", idx + 1),
                "initial_score": item.get("initial_score")
            })

    passage_texts = [p["content"] for p in normalized_passages]
    pairs = [[query, text] for text in passage_texts]

    is_fallback = False
    raw_scores: List[float] = []

    if model_loaded and cross_encoder_instance is not None:
        try:
            scores_array = cross_encoder_instance.predict(pairs)
            raw_scores = [float(s) for s in scores_array]
        except Exception as e:
            logger.error(f"Error during cross-encoder prediction: {e}")
            is_fallback = True
            raw_scores = fallback_bm25_rerank(query, passage_texts)
    else:
        is_fallback = True
        raw_scores = fallback_bm25_rerank(query, passage_texts)

    # Pair each passage with its score
    scored_items = []
    for idx, passage in enumerate(normalized_passages):
        score = raw_scores[idx]
        conf = sigmoid(score) if not is_fallback else float(score)
        scored_items.append({
            "passage": passage,
            "raw_score": score,
            "confidence": conf
        })

    # Sort descending by score
    scored_items.sort(key=lambda x: x["raw_score"], reverse=True)

    # Apply top_n slice
    top_n = payload.top_n if payload.top_n is not None and payload.top_n > 0 else len(scored_items)
    selected_items = scored_items[:top_n]

    # Build final response objects
    results = []
    for new_idx, item in enumerate(selected_items):
        p = item["passage"]
        new_rank = new_idx + 1
        initial_rank = p["initial_rank"]
        rank_delta = initial_rank - new_rank if initial_rank else 0

        results.append(RerankResultItem(
            id=p["id"],
            content=p["content"],
            rerank_score=round(item["raw_score"], 4),
            confidence_score=round(item["confidence"], 4),
            initial_rank=initial_rank,
            new_rank=new_rank,
            rank_delta=rank_delta,
            metadata=p["metadata"]
        ))

    latency = (time.time() - start_time) * 1000

    return RerankResponse(
        query=query,
        results=results,
        total_input_passages=len(raw_passages),
        returned_passages=len(results),
        model=MODEL_NAME if not is_fallback else f"{MODEL_NAME}-lexical-fallback",
        latency_ms=round(latency, 2),
        is_fallback=is_fallback
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
