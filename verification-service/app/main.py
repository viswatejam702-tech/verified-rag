"""
Verification Service: FastAPI microservice for NLI Claim-Level Verification and Hallucination Scoring.
Default Model: cross-encoder/nli-deberta-v3-base (Contradiction, Entailment, Neutral)
"""

import os
import re
import time
import math
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verification-service")

MODEL_NAME = os.getenv("MODEL_NAME", "cross-encoder/nli-deberta-v3-base")
nli_model = None
nli_tokenizer = None
model_loaded = False

# Mapping for NLI label indices in cross-encoder/nli-deberta-v3-base:
# 0: contradiction, 1: entailment, 2: neutral (or similar standard NLI configs)
LABEL_MAPPING = {0: "contradiction", 1: "entailment", 2: "neutral"}


def softmax(logits: List[float]) -> List[float]:
    """Compute softmax probabilities from raw logits."""
    max_l = max(logits)
    exp_vals = [math.exp(l - max_l) for l in logits]
    sum_exp = sum(exp_vals)
    return [round(e / sum_exp, 4) for e in exp_vals]


def extract_claims_from_text(text: str) -> List[str]:
    """
    Extract discrete atomic claims from text.
    Filters out conversational filler (e.g. 'Sure! Here is the summary:')
    and breaks complex paragraphs into verifiable sentences.
    """
    if not text:
        return []

    # Strip markdown headers, lists, code blocks
    cleaned = re.sub(r'```[\s\S]*?```', '', text)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^\s*[-*•\d+.]\s*', '', cleaned, flags=re.MULTILINE)

    # Split by sentence delimiters
    raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', cleaned)
    
    claims = []
    filler_patterns = [
        r'^(certainly|sure|here is|here are|as requested|in summary|based on|hope this helps)',
        r'^(let me know|feel free|i hope|to summarize|in conclusion|if you need|please let me know)'
    ]

    for sent in raw_sentences:
        sent = sent.strip()
        # Clean inline citation marks like [1], [doc2], [3, 4]
        sent_no_citations = re.sub(r'\[(?:\w+[-_]?\w*|\d+(?:\s*,\s*\d+)*)\]', '', sent).strip()
        
        if len(sent_no_citations) < 10:
            continue
            
        # Check if sentence is pure conversational filler
        is_filler = any(re.search(pat, sent_no_citations.lower()) for pat in filler_patterns)
        if is_filler and len(sent_no_citations.split()) < 12:
            continue
            
        claims.append(sent_no_citations)

    return claims



def fallback_heuristic_nli(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Semantic token overlap and negation-aware NLI fallback when transformer weights are not loaded.
    """
    p_lower = premise.lower()
    h_lower = hypothesis.lower()

    # Extract alphanumeric words (> 2 chars)
    h_words = [w for w in re.findall(r'\b\w+\b', h_lower) if len(w) > 2]
    if not h_words:
        return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

    # Negation words detection
    negations = {"not", "never", "no", "cannot", "hardly", "barely", "failed", "neither", "none"}
    h_has_neg = any(w in negations for w in h_words)
    p_has_neg = any(w in negations for w in re.findall(r'\b\w+\b', p_lower))

    # Numbers / entities extracted
    h_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', h_lower))
    p_numbers = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', p_lower))

    # Overlap calculation
    matched_words = [w for w in h_words if w in p_lower]
    overlap_ratio = len(matched_words) / max(1, len(h_words))

    # Check for direct numerical contradiction
    if h_numbers and p_numbers:
        if not h_numbers.issubset(p_numbers) and overlap_ratio > 0.4:
            return {"entailment": 0.08, "neutral": 0.12, "contradiction": 0.80}

    # Check for polarity mismatch (negation flip)
    if (h_has_neg and not p_has_neg and overlap_ratio > 0.5) or (not h_has_neg and p_has_neg and overlap_ratio > 0.5):
        return {"entailment": 0.10, "neutral": 0.20, "contradiction": 0.70}

    # High overlap entailment
    if overlap_ratio >= 0.65:
        p_entail = min(0.95, 0.5 + overlap_ratio * 0.45)
        p_contra = 0.05
        p_neut = 1.0 - p_entail - p_contra
        return {"entailment": round(p_entail, 4), "neutral": round(p_neut, 4), "contradiction": round(p_contra, 4)}
    elif overlap_ratio >= 0.35:
        p_entail = 0.45
        p_neut = 0.45
        p_contra = 0.10
        return {"entailment": round(p_entail, 4), "neutral": round(p_neut, 4), "contradiction": round(p_contra, 4)}
    else:
        # Unsupported claim
        return {"entailment": 0.10, "neutral": 0.80, "contradiction": 0.10}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global nli_model, nli_tokenizer, model_loaded
    logger.info(f"Initializing Verification Service with NLI model: {MODEL_NAME}")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        logger.info("Loading HuggingFace NLI model & tokenizer...")
        nli_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        nli_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        nli_model.eval()
        model_loaded = True
        logger.info("NLI Verification Model loaded successfully!")
    except Exception as e:
        logger.warning(f"Could not load NLI transformers directly ({e}). Running in resilient fallback mode.")
        model_loaded = False
    yield
    logger.info("Shutting down Verification Service.")


app = FastAPI(
    title="Verified RAG - Verification & Hallucination Service",
    description="Natural Language Inference (NLI) Claim-Level Verification and Hallucination Scoring",
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
class ExtractClaimsRequest(BaseModel):
    text: str = Field(..., description="Answer or text from which to extract verifiable claims")


class ExtractClaimsResponse(BaseModel):
    claims: List[str]
    count: int


class VerifyRequest(BaseModel):
    answer: Optional[str] = Field(default=None, description="Full answer text to verify (will be split into claims)")
    claims: Optional[List[str]] = Field(default=None, description="Explicit list of atomic claims")
    evidence_passages: List[str] = Field(..., description="Retrieved context passages to verify against", min_length=1)
    threshold_entailment: float = Field(default=0.60, description="Minimum confidence to mark as VERIFIED")
    threshold_contradiction: float = Field(default=0.50, description="Minimum confidence to mark as CONTRADICTION")


class ClaimVerificationResult(BaseModel):
    claim: str
    verdict: str  # 'VERIFIED' | 'CONTRADICTION' | 'UNSUPPORTED'
    confidence: float
    entailment_prob: float
    contradiction_prob: float
    neutral_prob: float
    supporting_evidence: Optional[str] = None
    supporting_passage_index: Optional[int] = None


class VerifyResponse(BaseModel):
    faithfulness_score: float  # 0.0 to 1.0 (verified / total)
    hallucination_rate: float  # 0.0 to 1.0 (contradicted + unsupported / total)
    total_claims: int
    verified_claims_count: int
    contradicted_claims_count: int
    unsupported_claims_count: int
    claims: List[ClaimVerificationResult]
    model: str
    latency_ms: float
    is_fallback: bool


# --- Endpoints ---
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "verification-service",
        "model": MODEL_NAME,
        "model_loaded": model_loaded
    }


@app.post("/extract-claims", response_model=ExtractClaimsResponse)
def extract_claims(payload: ExtractClaimsRequest):
    claims = extract_claims_from_text(payload.text)
    return ExtractClaimsResponse(
        claims=claims,
        count=len(claims)
    )


@app.post("/verify", response_model=VerifyResponse)
def verify_answer(payload: VerifyRequest):
    start_time = time.time()
    
    # 1. Determine claims
    claims_to_verify: List[str] = []
    if payload.claims and len(payload.claims) > 0:
        claims_to_verify = [c.strip() for c in payload.claims if c.strip()]
    elif payload.answer:
        claims_to_verify = extract_claims_from_text(payload.answer)
        
    if not claims_to_verify:
        return VerifyResponse(
            faithfulness_score=1.0,
            hallucination_rate=0.0,
            total_claims=0,
            verified_claims_count=0,
            contradicted_claims_count=0,
            unsupported_claims_count=0,
            claims=[],
            model=MODEL_NAME,
            latency_ms=0.0,
            is_fallback=False
        )

    evidence_list = [e.strip() for e in payload.evidence_passages if e.strip()]
    if not evidence_list:
        raise HTTPException(status_code=400, detail="Evidence passages list cannot be empty")

    results: List[ClaimVerificationResult] = []
    is_fallback = not (model_loaded and nli_model is not None and nli_tokenizer is not None)

    for claim in claims_to_verify:
        best_verdict = "UNSUPPORTED"
        best_entailment = 0.0
        best_contradiction = 0.0
        best_neutral = 1.0
        best_evidence = None
        best_evidence_idx = None

        for idx, evidence in enumerate(evidence_list):
            if is_fallback:
                probs = fallback_heuristic_nli(premise=evidence, hypothesis=claim)
                p_contra = probs["contradiction"]
                p_entail = probs["entailment"]
                p_neut = probs["neutral"]
            else:
                try:
                    import torch
                    inputs = nli_tokenizer(
                        evidence,
                        claim,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                    with torch.no_grad():
                        outputs = nli_model(**inputs)
                        logits = outputs.logits[0].tolist()
                        probs_list = softmax(logits)
                        # DeBERTa NLI standard: [contradiction, entailment, neutral]
                        p_contra = probs_list[0]
                        p_entail = probs_list[1]
                        p_neut = probs_list[2]
                except Exception as e:
                    logger.error(f"Inference error during claim verification: {e}")
                    is_fallback = True
                    probs = fallback_heuristic_nli(premise=evidence, hypothesis=claim)
                    p_contra = probs["contradiction"]
                    p_entail = probs["entailment"]
                    p_neut = probs["neutral"]

            # Keep track of highest entailment or contradiction
            if p_entail > best_entailment:
                best_entailment = p_entail
                best_contradiction = p_contra
                best_neutral = p_neut
                best_evidence = evidence
                best_evidence_idx = idx

            # If strong contradiction found
            if p_contra >= payload.threshold_contradiction and p_contra > best_contradiction:
                best_contradiction = p_contra
                best_evidence = evidence
                best_evidence_idx = idx

        # Determine final verdict for this claim
        if best_contradiction >= payload.threshold_contradiction and best_contradiction > best_entailment:
            verdict = "CONTRADICTION"
            confidence = best_contradiction
        elif best_entailment >= payload.threshold_entailment:
            verdict = "VERIFIED"
            confidence = best_entailment
        else:
            verdict = "UNSUPPORTED"
            confidence = best_neutral

        # Extract snippet of evidence (first 180 chars)
        evidence_snippet = None
        if best_evidence:
            evidence_snippet = best_evidence[:200] + ("..." if len(best_evidence) > 200 else "")

        results.append(ClaimVerificationResult(
            claim=claim,
            verdict=verdict,
            confidence=round(confidence, 4),
            entailment_prob=round(best_entailment, 4),
            contradiction_prob=round(best_contradiction, 4),
            neutral_prob=round(best_neutral, 4),
            supporting_evidence=evidence_snippet,
            supporting_passage_index=best_evidence_idx
        ))

    # Calculate Aggregate Metrics
    total = len(results)
    verified_cnt = sum(1 for r in results if r.verdict == "VERIFIED")
    contradicted_cnt = sum(1 for r in results if r.verdict == "CONTRADICTION")
    unsupported_cnt = sum(1 for r in results if r.verdict == "UNSUPPORTED")

    faithfulness = round(verified_cnt / total, 4) if total > 0 else 1.0
    hallucination_rate = round((contradicted_cnt + unsupported_cnt) / total, 4) if total > 0 else 0.0

    latency = (time.time() - start_time) * 1000

    return VerifyResponse(
        faithfulness_score=faithfulness,
        hallucination_rate=hallucination_rate,
        total_claims=total,
        verified_claims_count=verified_cnt,
        contradicted_claims_count=contradicted_cnt,
        unsupported_claims_count=unsupported_cnt,
        claims=results,
        model=MODEL_NAME if not is_fallback else f"{MODEL_NAME}-heuristic-fallback",
        latency_ms=round(latency, 2),
        is_fallback=is_fallback
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
