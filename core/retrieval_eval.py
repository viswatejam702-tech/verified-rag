"""
Verified RAG — Information Retrieval Evaluation Suite.

Computes:
1. Recall@K & Precision@K
2. Mean Reciprocal Rank (MRR)
3. Normalized Discounted Cumulative Gain (nDCG@K)
4. Context Precision & Context Recall (RAGAS-aligned)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set


@dataclass
class RetrievalEvalQuery:
    query_id: str
    query: str
    relevant_chunk_ids: Set[str]
    target_topic: str


@dataclass
class RetrievalMetricResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    context_precision: float
    context_recall: float
    k: int
    total_queries: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "k": self.k,
            "total_queries": self.total_queries,
        }


# ─── Metric Calculations ──────────────────────────────────────────────────────

def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if not ground_truth_ids:
        return 1.0
    top_k_retrieved = set(retrieved_ids[:k])
    hits = len(top_k_retrieved & ground_truth_ids)
    return hits / len(ground_truth_ids)


def compute_precision_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k_retrieved = set(retrieved_ids[:k])
    hits = len(top_k_retrieved & ground_truth_ids)
    return hits / k


def compute_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> float:
    for idx, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in ground_truth_ids:
            return 1.0 / idx
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = 1.0 if doc_id in ground_truth_ids else 0.0
        dcg += rel / math.log2(idx + 1)

    # Ideal DCG (all relevant items ranked at the top)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ground_truth_ids), k) + 1))
    if idcg <= 0.0:
        return 1.0 if not ground_truth_ids else 0.0
    return dcg / idcg


def evaluate_retrieval_suite(
    evaluation_queries: List[RetrievalEvalQuery],
    retrieval_fn: Any,  # Callable[[str], List[str]] -> returns list of retrieved chunk_ids
    k: int = 3,
) -> RetrievalMetricResult:
    """
    Evaluates a retrieval configuration over a labelled evaluation dataset.
    """
    total_q = len(evaluation_queries)
    if total_q == 0:
        return RetrievalMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, k, 0)

    recalls, precisions, mrrs, ndcgs = [], [], [], []

    for item in evaluation_queries:
        retrieved_ids = retrieval_fn(item.query)
        recalls.append(compute_recall_at_k(retrieved_ids, item.relevant_chunk_ids, k))
        precisions.append(compute_precision_at_k(retrieved_ids, item.relevant_chunk_ids, k))
        mrrs.append(compute_reciprocal_rank(retrieved_ids, item.relevant_chunk_ids))
        ndcgs.append(compute_ndcg_at_k(retrieved_ids, item.relevant_chunk_ids, k))

    avg_recall = sum(recalls) / total_q
    avg_prec = sum(precisions) / total_q
    avg_mrr = sum(mrrs) / total_q
    avg_ndcg = sum(ndcgs) / total_q

    # Context precision (weighted precision by rank) and Context recall
    context_precision = avg_prec * 0.9 + avg_mrr * 0.1
    context_recall = avg_recall

    return RetrievalMetricResult(
        recall_at_k=avg_recall,
        precision_at_k=avg_prec,
        mrr=avg_mrr,
        ndcg_at_k=avg_ndcg,
        context_precision=context_precision,
        context_recall=context_recall,
        k=k,
        total_queries=total_q,
    )


# ─── Curated Labelled Retrieval Benchmark Dataset ────────────────────────────

def get_labelled_retrieval_benchmark() -> List[RetrievalEvalQuery]:
    """Returns verified queries mapped to ground truth chunk IDs."""
    return [
        RetrievalEvalQuery(
            query_id="ret_q1",
            query="What dilution refrigerator temperature is needed for transmon qubits?",
            relevant_chunk_ids={"chunk-q1"},
            target_topic="Quantum Computing",
        ),
        RetrievalEvalQuery(
            query_id="ret_q2",
            query="What is the physical to logical qubit ratio demonstrated in 2024?",
            relevant_chunk_ids={"chunk-q2"},
            target_topic="Quantum Error Correction",
        ),
        RetrievalEvalQuery(
            query_id="ret_q3",
            query="What is the average hydrodynamic diameter of mRNA lipid nanoparticles?",
            relevant_chunk_ids={"chunk-q3"},
            target_topic="mRNA Vaccine Delivery",
        ),
        RetrievalEvalQuery(
            query_id="ret_q4",
            query="Which NIST publication formalizes the Zero Trust Architecture?",
            relevant_chunk_ids={"chunk-q4"},
            target_topic="Zero Trust Security",
        ),
    ]

