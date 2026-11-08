"""
Verified RAG — Telemetry, Observability & Request Tracing Engine.

Tracks every RAG transaction with:
1. Stage-by-stage latency breakdowns (retrieval, reranking, generation, verification)
2. Token usage and model identifiers
3. Verification verdicts & faithfulness scoring
4. Privacy-by-design: Sanitizes or hashes raw query text and never logs sensitive document bodies by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("verified-rag.observability")


@dataclass
class RAGRequestTrace:
    request_id: str
    timestamp_utc: str
    query_hash: str
    retrieval_mode: str
    retrieved_doc_ids: List[str]
    retrieved_chunk_count: int
    duration_retrieval_ms: float
    duration_rerank_ms: float
    duration_generation_ms: float
    duration_verification_ms: float
    total_latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_identifier: str = "local-synthesizer"
    total_claims_extracted: int = 0
    faithfulness_score: float = 0.0
    hallucination_detected: bool = False
    abstained: bool = False
    client_ip_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGTelemetryTracker:
    """In-memory telemetry collector and request tracer."""

    def __init__(self, max_traces: int = 1000):
        self._traces: List[RAGRequestTrace] = []
        self._max_traces = max_traces

    def hash_text(self, text: str) -> str:
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]

    def record_trace(
        self,
        request_id: str,
        query: str,
        retrieval_mode: str,
        retrieved_doc_ids: List[str],
        retrieved_chunk_count: int,
        duration_retrieval_ms: float,
        duration_rerank_ms: float,
        duration_generation_ms: float,
        duration_verification_ms: float,
        total_latency_ms: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model_identifier: str = "default-model",
        total_claims_extracted: int = 0,
        faithfulness_score: float = 1.0,
        hallucination_detected: bool = False,
        abstained: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RAGRequestTrace:
        trace = RAGRequestTrace(
            request_id=request_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            query_hash=self.hash_text(query),
            retrieval_mode=retrieval_mode,
            retrieved_doc_ids=retrieved_doc_ids,
            retrieved_chunk_count=retrieved_chunk_count,
            duration_retrieval_ms=round(duration_retrieval_ms, 2),
            duration_rerank_ms=round(duration_rerank_ms, 2),
            duration_generation_ms=round(duration_generation_ms, 2),
            duration_verification_ms=round(duration_verification_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_identifier=model_identifier,
            total_claims_extracted=total_claims_extracted,
            faithfulness_score=round(faithfulness_score, 4),
            hallucination_detected=hallucination_detected,
            abstained=abstained,
            metadata=metadata or {},
        )

        self._traces.append(trace)
        if len(self._traces) > self._max_traces:
            self._traces.pop(0)

        logger.info(
            f"[RAG-TRACE {trace.request_id}] Mode={trace.retrieval_mode} Total={trace.total_latency_ms}ms "
            f"Faithfulness={trace.faithfulness_score:.2f} Abstained={trace.abstained}"
        )
        return trace

    def get_recent_traces(self, limit: int = 50) -> List[RAGRequestTrace]:
        return list(reversed(self._traces[-limit:]))

    def get_summary_metrics(self) -> Dict[str, Any]:
        if not self._traces:
            return {
                "total_requests": 0,
                "avg_latency_ms": 0.0,
                "avg_faithfulness": 0.0,
                "hallucination_rate_pct": 0.0,
                "abstention_rate_pct": 0.0,
            }
        N = len(self._traces)
        avg_lat = sum(t.total_latency_ms for t in self._traces) / N
        avg_faith = sum(t.faithfulness_score for t in self._traces) / N
        hallucinated = sum(1 for t in self._traces if t.hallucination_detected)
        abstained = sum(1 for t in self._traces if t.abstained)

        return {
            "total_requests": N,
            "avg_latency_ms": round(avg_lat, 2),
            "avg_faithfulness": round(avg_faith, 4),
            "hallucination_rate_pct": round(hallucinated / N * 100.0, 2),
            "abstention_rate_pct": round(abstained / N * 100.0, 2),
        }

