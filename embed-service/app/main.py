"""
Embed Service: FastAPI microservice for computing vector embeddings & semantic search.
Default Model: sentence-transformers/all-MiniLM-L6-v2 (384-d output)
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

try:
    from app.utils import chunk_text, generate_fallback_embedding, compute_cosine_similarity
except ImportError:
    from utils import chunk_text, generate_fallback_embedding, compute_cosine_similarity


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embed-service")

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
model_instance = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance, model_loaded
    logger.info(f"Initializing Embed Service with model: {MODEL_NAME}")
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading SentenceTransformer model...")
        model_instance = SentenceTransformer(MODEL_NAME)
        model_loaded = True
        logger.info("SentenceTransformer model loaded successfully!")
    except Exception as e:
        logger.warning(f"Could not load SentenceTransformer directly ({e}). Running in resilient fallback mode.")
        model_loaded = False
    yield
    logger.info("Shutting down Embed Service.")


app = FastAPI(
    title="Verified RAG - Embed Service",
    description="Vector Embedding Generation & Similarity Search Microservice",
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
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., description="List of text strings to embed", min_length=1)
    normalize: bool = Field(default=True, description="Whether to L2-normalize embeddings")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    model: str
    count: int
    latency_ms: float
    is_fallback: bool


class ChunkAndEmbedRequest(BaseModel):
    text: str = Field(..., description="Raw text document to chunk and embed")
    chunk_size: int = Field(default=500, description="Max character length per chunk")
    chunk_overlap: int = Field(default=80, description="Overlap characters between chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata to attach to chunks")


class ChunkAndEmbedResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    total_chunks: int
    dimensions: int
    latency_ms: float


class SimilarityRequest(BaseModel):
    query: str
    documents: List[str]


class SimilarityResponse(BaseModel):
    similarities: List[float]
    top_index: int
    top_similarity: float


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "embed-service",
        "model": MODEL_NAME,
        "model_loaded": model_loaded,
        "dimensions": 384
    }


@app.post("/embed", response_model=EmbedResponse)
def get_embeddings(payload: EmbedRequest):
    start_time = time.time()
    texts = payload.texts
    
    if not texts:
        raise HTTPException(status_code=400, detail="Text list cannot be empty")

    embeddings_list = []
    is_fallback = False

    if model_loaded and model_instance is not None:
        try:
            raw_embeddings = model_instance.encode(
                texts,
                normalize_embeddings=payload.normalize,
                show_progress_bar=False
            )
            embeddings_list = raw_embeddings.tolist()
        except Exception as e:
            logger.error(f"Error during transformer inference: {e}")
            is_fallback = True
            embeddings_list = [generate_fallback_embedding(t, dim=384) for t in texts]
    else:
        is_fallback = True
        embeddings_list = [generate_fallback_embedding(t, dim=384) for t in texts]

    latency = (time.time() - start_time) * 1000
    dims = len(embeddings_list[0]) if embeddings_list else 384

    return EmbedResponse(
        embeddings=embeddings_list,
        dimensions=dims,
        model=MODEL_NAME if not is_fallback else f"{MODEL_NAME}-fallback",
        count=len(embeddings_list),
        latency_ms=round(latency, 2),
        is_fallback=is_fallback
    )


@app.post("/chunk-and-embed", response_model=ChunkAndEmbedResponse)
def chunk_and_embed_document(payload: ChunkAndEmbedRequest):
    start_time = time.time()
    
    # 1. Chunk document
    raw_chunks = chunk_text(
        text=payload.text,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap
    )
    
    if not raw_chunks:
        return ChunkAndEmbedResponse(
            chunks=[],
            total_chunks=0,
            dimensions=384,
            latency_ms=0.0
        )

    # 2. Extract contents
    texts = [c["content"] for c in raw_chunks]
    
    # 3. Compute embeddings
    if model_loaded and model_instance is not None:
        try:
            embeddings = model_instance.encode(texts, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"Inference error in chunk-and-embed: {e}")
            embeddings = [generate_fallback_embedding(t, 384) for t in texts]
    else:
        embeddings = [generate_fallback_embedding(t, 384) for t in texts]

    # 4. Merge back
    meta = payload.metadata or {}
    enriched_chunks = []
    for idx, chunk_data in enumerate(raw_chunks):
        enriched_chunks.append({
            "chunk_index": chunk_data["chunk_index"],
            "content": chunk_data["content"],
            "token_count": chunk_data["token_count"],
            "char_count": chunk_data["char_count"],
            "embedding": embeddings[idx],
            "metadata": meta
        })

    latency = (time.time() - start_time) * 1000

    return ChunkAndEmbedResponse(
        chunks=enriched_chunks,
        total_chunks=len(enriched_chunks),
        dimensions=len(embeddings[0]) if embeddings else 384,
        latency_ms=round(latency, 2)
    )


@app.post("/similarity", response_model=SimilarityResponse)
def calculate_similarity(payload: SimilarityRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="Documents list cannot be empty")

    # Embed query and documents
    all_texts = [payload.query] + payload.documents
    if model_loaded and model_instance is not None:
        try:
            all_vecs = model_instance.encode(all_texts, normalize_embeddings=True).tolist()
        except Exception:
            all_vecs = [generate_fallback_embedding(t, 384) for t in all_texts]
    else:
        all_vecs = [generate_fallback_embedding(t, 384) for t in all_texts]

    query_vec = all_vecs[0]
    doc_vecs = all_vecs[1:]

    sims = [compute_cosine_similarity(query_vec, dv) for dv in doc_vecs]
    top_idx = int(np.argmax(sims)) if sims else 0
    top_sim = float(sims[top_idx]) if sims else 0.0

    return SimilarityResponse(
        similarities=[round(s, 4) for s in sims],
        top_index=top_idx,
        top_similarity=round(top_sim, 4)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
