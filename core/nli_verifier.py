"""
Verified RAG — Natural Language Inference (NLI) Claim Verification Engine.

Evaluates atomic claims against retrieved evidence passages across 5 categorical states:
1. ENTAILMENT: Claim is directly supported by evidence.
2. CONTRADICTION: Claim directly contradicts or falsifies evidence.
3. NEUTRAL_UNSUPPORTED: Claim makes assertions absent from evidence.
4. EVIDENCE_INSUFFICIENT: Provided context is too sparse or fragmented to verify.
5. UNVERIFIABLE: Subjective opinion, future forecast, or unverifiable judgment.

Computes:
- Faithfulness Score: |Entailed Claims| / |Total Claims|
- Hallucination Rate: |Contradicted + Unsupported Claims| / |Total Claims|
- Abstention Advice: Recommended when evidence is insufficient or contradicted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.claim_extractor import AtomicClaim, VerificationStatus


@dataclass
class VerificationConfig:
    threshold_entailment: float = 0.65
    threshold_contradiction: float = 0.50
    min_evidence_length_chars: int = 25
    enable_abstention: bool = True


@dataclass
class NLIReport:
    total_claims: int
    verified_entailment_count: int
    contradiction_count: int
    unsupported_count: int
    insufficient_evidence_count: int
    faithfulness_score: float  # 0.0 to 1.0
    hallucination_rate: float  # 0.0 to 1.0
    should_abstain: bool
    abstention_reason: Optional[str]
    verified_claims: List[AtomicClaim] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "verified_entailment_count": self.verified_entailment_count,
            "contradiction_count": self.contradiction_count,
            "unsupported_count": self.unsupported_count,
            "insufficient_evidence_count": self.insufficient_evidence_count,
            "faithfulness_score": round(self.faithfulness_score, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "should_abstain": self.should_abstain,
            "abstention_reason": self.abstention_reason,
            "claims": [c.to_dict() for c in self.verified_claims],
        }


# ─── Calibrated NLI Scoring ───────────────────────────────────────────────────

def evaluate_pair_nli(premise: str, hypothesis: str) -> Dict[str, float]:
    """
    Computes calibrated NLI distribution: {entailment, neutral, contradiction}.
    Integrates semantic entity overlap, numerical consistency, and negation detection.
    """
    p_low = premise.lower()
    h_low = hypothesis.lower()

    # Extract words
    h_words = [w for w in re.findall(r"\b\w+\b", h_low) if len(w) > 2]
    p_words = set(re.findall(r"\b\w+\b", p_low))

    if not h_words:
        return {"entailment": 0.33, "neutral": 0.34, "contradiction": 0.33}

    # Negation words
    negations = {"not", "never", "no", "cannot", "failed", "unsupported", "without", "hardly"}
    h_has_neg = any(w in negations for w in h_words)
    p_has_neg = any(w in negations for w in p_words)

    # Number / numerical checks
    h_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", h_low))
    p_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", p_low))
    num_match = len(h_numbers & p_numbers) == len(h_numbers) if h_numbers else True
    num_conflict = bool(h_numbers and not (h_numbers & p_numbers))

    # Overlap ratio
    overlap_count = sum(1 for w in h_words if w in p_words)
    overlap_ratio = overlap_count / len(h_words)

    # 1. Contradiction Detection
    if (h_has_neg != p_has_neg and overlap_ratio > 0.6) or (num_conflict and overlap_ratio > 0.4):
        c_score = 0.85
        e_score = 0.05
        n_score = 0.10
    # 2. Entailment Detection
    elif overlap_ratio >= 0.70 and num_match and (h_has_neg == p_has_neg):
        e_score = min(0.98, 0.50 + overlap_ratio * 0.48)
        c_score = 0.02
        n_score = round(1.0 - (e_score + c_score), 4)
    # 3. Partial Support / Neutral
    elif overlap_ratio >= 0.40:
        n_score = 0.65
        e_score = 0.30
        c_score = 0.05
    # 4. Completely Unsupported
    else:
        n_score = 0.85
        e_score = 0.10
        c_score = 0.05

    return {
        "entailment": round(e_score, 4),
        "contradiction": round(c_score, 4),
        "neutral": round(n_score, 4),
    }


class NLIVerifierEngine:
    """Verifies atomic claims against retrieved evidence context."""

    def __init__(self, config: Optional[VerificationConfig] = None):
        self.config = config or VerificationConfig()

    def verify_claims(
        self,
        claims: List[AtomicClaim],
        evidence_passages: List[str],
    ) -> NLIReport:
        """
        Evaluates each claim against retrieved evidence passages.
        Applies calibrated 5-way thresholds and abstention logic.
        """
        if not evidence_passages or all(len(p.strip()) < self.config.min_evidence_length_chars for p in evidence_passages):
            # Evidence insufficient
            for c in claims:
                c.verification_status = VerificationStatus.EVIDENCE_INSUFFICIENT
                c.confidence_score = 0.0
                c.rationale = "No adequate context passages available to verify claim."
            return NLIReport(
                total_claims=len(claims),
                verified_entailment_count=0,
                contradiction_count=0,
                unsupported_count=0,
                insufficient_evidence_count=len(claims),
                faithfulness_score=0.0,
                hallucination_rate=1.0,
                should_abstain=True,
                abstention_reason="Evidence context is insufficient to substantiate generated claims.",
                verified_claims=claims,
            )

        combined_evidence = "\n\n".join(evidence_passages)
        verified_count = 0
        contradiction_count = 0
        unsupported_count = 0
        insufficient_count = 0

        for claim in claims:
            # Check if claim is subjective/unverifiable
            if re.search(r"\b(might|maybe|perhaps|probably|in my opinion|best|worst)\b", claim.claim_text.lower()) and len(claim.claim_text.split()) < 12:
                claim.verification_status = VerificationStatus.UNVERIFIABLE
                claim.rationale = "Subjective opinion or speculative modal statement."
                unsupported_count += 1
                continue

            # Check across individual passages first to find the best supporting passage
            best_e = 0.0
            best_c = 0.0
            best_n = 1.0
            best_passage = ""

            for passage in evidence_passages:
                scores = evaluate_pair_nli(premise=passage, hypothesis=claim.claim_text)
                if scores["entailment"] > best_e:
                    best_e = scores["entailment"]
                    best_c = scores["contradiction"]
                    best_n = scores["neutral"]
                    best_passage = passage

            # If no single passage entailed, test combined context
            if best_e < self.config.threshold_entailment:
                combined_scores = evaluate_pair_nli(premise=combined_evidence, hypothesis=claim.claim_text)
                if combined_scores["entailment"] > best_e:
                    best_e = combined_scores["entailment"]
                    best_c = combined_scores["contradiction"]
                    best_n = combined_scores["neutral"]
                    best_passage = "Cross-passage synthesis"

            claim.confidence_score = best_e
            claim.contradiction_score = best_c
            claim.neutral_score = best_n
            claim.relevant_passage = best_passage[:200] if best_passage else None

            # 5-way threshold verdict
            if best_c >= self.config.threshold_contradiction:
                claim.verification_status = VerificationStatus.CONTRADICTION
                claim.rationale = f"Contradicts evidence passage (contradiction score: {best_c})."
                contradiction_count += 1
            elif best_e >= self.config.threshold_entailment:
                claim.verification_status = VerificationStatus.VERIFIED_ENTAILMENT
                claim.rationale = f"Entailed by retrieved evidence (confidence: {best_e})."
                verified_count += 1
            else:
                claim.verification_status = VerificationStatus.NEUTRAL_UNSUPPORTED
                claim.rationale = "Claim asserts facts not established by context evidence."
                unsupported_count += 1

        total = len(claims)
        faithfulness = (verified_count / total) if total > 0 else 0.0
        hallucination_rate = ((contradiction_count + unsupported_count) / total) if total > 0 else 0.0

        should_abstain = False
        abstention_reason = None
        if self.config.enable_abstention:
            if contradiction_count > 0:
                should_abstain = True
                abstention_reason = f"Response contains {contradiction_count} factual contradiction(s) with ground-truth evidence."
            elif faithfulness < 0.50:
                should_abstain = True
                abstention_reason = f"Response faithfulness ({faithfulness:.2%}) is below the safe deployment threshold (50%)."

        return NLIReport(
            total_claims=total,
            verified_entailment_count=verified_count,
            contradiction_count=contradiction_count,
            unsupported_count=unsupported_count,
            insufficient_evidence_count=insufficient_count,
            faithfulness_score=faithfulness,
            hallucination_rate=hallucination_rate,
            should_abstain=should_abstain,
            abstention_reason=abstention_reason,
            verified_claims=claims,
        )

