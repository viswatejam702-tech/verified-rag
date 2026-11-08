"""
Verified RAG — Atomic Propositional Claim Extraction Engine.

Decomposes complex generative responses into independent, verifiable factual claims.
Preserves explicit bracketed citations, maps potential source references,
and discards conversational boilerplate.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    VERIFIED_ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL_UNSUPPORTED = "neutral_unsupported"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    UNVERIFIABLE = "unverifiable"


@dataclass
class AtomicClaim:
    """An independent factual assertion extracted from text."""
    claim_id: str
    claim_text: str
    evidence_references: List[str] = field(default_factory=list)
    source_document: Optional[str] = None
    relevant_passage: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.NEUTRAL_UNSUPPORTED
    confidence_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 1.0
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification_status"] = self.verification_status.value
        return d


CONVERSATIONAL_FILLER_PATTERNS = [
    r"^(certainly|sure|here is|here are|as requested|in summary|based on the context|hope this helps)",
    r"^(let me know|feel free|i hope this helps|to summarize|in conclusion|if you need anything else)",
    r"^(thank you|please note that|according to the provided text|as mentioned above)",
]


def extract_atomic_claims(answer_text: str) -> List[AtomicClaim]:
    """
    Splits text into fine-grained atomic factual claims.
    Extracts citation references (e.g., [1], [Passage 2], [chunk-q1]).
    """
    if not answer_text or not answer_text.strip():
        return []

    # Strip code blocks and markdown headers
    cleaned = re.sub(r"```[\s\S]*?```", "", answer_text)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*[-*•\d+.]\s*", "", cleaned, flags=re.MULTILINE)

    # Sentence boundary split
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    claims: List[AtomicClaim] = []
    claim_counter = 1

    for sent in raw_sentences:
        sent = sent.strip()
        if len(sent) < 8:
            continue

        # Extract citation references like [1], [Passage 2], [doc-123]
        citations = re.findall(r"\[(Passage\s*\d+|\d+|[a-zA-Z0-9_\-]+)\]", sent, flags=re.IGNORECASE)

        # Remove citations for cleaner propositional representation
        clean_claim_text = re.sub(r"\[(?:Passage\s*\d+|\d+|[a-zA-Z0-9_\-]+)\]", "", sent).strip()

        # Check conversational filler
        is_filler = any(re.search(pat, clean_claim_text.lower()) for pat in CONVERSATIONAL_FILLER_PATTERNS)
        if is_filler and len(clean_claim_text.split()) < 10:
            continue

        if len(clean_claim_text) < 10:
            continue

        claims.append(
            AtomicClaim(
                claim_id=f"CLM-{claim_counter:03d}",
                claim_text=clean_claim_text,
                evidence_references=citations,
            )
        )
        claim_counter += 1

    return claims

