"""
Verified RAG — Citation Integrity & Grounding Verifier.

Asserts that:
1. Every generated citation points to a real, retrieved document passage (no hallucinated citations).
2. The cited passage actually supports the specific claim it is attributed to (Citation Precision).
3. Evaluates Citation Coverage, Alignment, and Unsupported Claim Rate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.claim_extractor import AtomicClaim


@dataclass
class CitationAuditRecord:
    citation_tag: str
    is_valid_reference: bool
    mapped_passage_index: Optional[int]
    actually_supports_claim: bool
    hallucinated_reference: bool
    notes: str


@dataclass
class CitationIntegrityReport:
    total_citations: int
    valid_citations: int
    invented_citations: int
    citation_precision_pct: float  # Supported citations / Valid citations
    citation_coverage_pct: float   # Claims with citations / Total claims
    unsupported_claim_rate_pct: float
    is_valid: bool
    citation_audits: List[CitationAuditRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invented_citations": self.invented_citations,
            "citation_precision_pct": round(self.citation_precision_pct, 2),
            "citation_coverage_pct": round(self.citation_coverage_pct, 2),
            "unsupported_claim_rate_pct": round(self.unsupported_claim_rate_pct, 2),
            "is_valid": self.is_valid,
            "audits": [
                {
                    "tag": a.citation_tag,
                    "valid": a.is_valid_reference,
                    "supports_claim": a.actually_supports_claim,
                    "hallucinated": a.hallucinated_reference,
                    "notes": a.notes,
                }
                for a in self.citation_audits
            ],
        }


class CitationVerifier:
    """Verifies that citations map strictly to actual retrieved evidence."""

    def __init__(self, retrieved_passages: List[str]):
        self.retrieved_passages = retrieved_passages
        self.total_passages = len(retrieved_passages)

    def parse_passage_index(self, tag: str) -> Optional[int]:
        """Maps tags like 'Passage 1', '1', '[2]' to 0-based index."""
        match = re.search(r"\b(\d+)\b", tag)
        if match:
            idx = int(match.group(1)) - 1
            return idx if 0 <= idx < self.total_passages else None
        return None

    def audit_citations(self, claims: List[AtomicClaim]) -> CitationIntegrityReport:
        audits: List[CitationAuditRecord] = []
        total_citations = 0
        valid_citations = 0
        invented_citations = 0
        supported_citations = 0
        claims_with_citations = 0

        for claim in claims:
            if claim.evidence_references:
                claims_with_citations += 1

            for tag in claim.evidence_references:
                total_citations += 1
                p_idx = self.parse_passage_index(tag)

                if p_idx is not None:
                    valid_citations += 1
                    passage_text = self.retrieved_passages[p_idx].lower()
                    # Check if claim keywords appear in cited passage
                    claim_words = [w for w in re.findall(r"\b\w+\b", claim.claim_text.lower()) if len(w) > 3]
                    overlap = sum(1 for w in claim_words if w in passage_text)
                    supports = (overlap / len(claim_words) >= 0.5) if claim_words else True

                    if supports:
                        supported_citations += 1
                        note = f"Citation '{tag}' successfully verified against Passage {p_idx + 1}."
                    else:
                        note = f"Citation '{tag}' points to Passage {p_idx + 1}, but passage does not contain necessary facts for claim."

                    audits.append(
                        CitationAuditRecord(
                            citation_tag=tag,
                            is_valid_reference=True,
                            mapped_passage_index=p_idx,
                            actually_supports_claim=supports,
                            hallucinated_reference=False,
                            notes=note,
                        )
                    )
                else:
                    invented_citations += 1
                    audits.append(
                        CitationAuditRecord(
                            citation_tag=tag,
                            is_valid_reference=False,
                            mapped_passage_index=None,
                            actually_supports_claim=False,
                            hallucinated_reference=True,
                            notes=f"Invented reference '{tag}' does not exist in the {self.total_passages} retrieved passages.",
                        )
                    )

        total_claims = len(claims)
        precision = (supported_citations / valid_citations * 100.0) if valid_citations > 0 else 100.0
        coverage = (claims_with_citations / total_claims * 100.0) if total_claims > 0 else 100.0
        unsupported_rate = 100.0 - precision if total_citations > 0 else 0.0
        is_valid = invented_citations == 0 and precision >= 80.0

        return CitationIntegrityReport(
            total_citations=total_citations,
            valid_citations=valid_citations,
            invented_citations=invented_citations,
            citation_precision_pct=precision,
            citation_coverage_pct=coverage,
            unsupported_claim_rate_pct=unsupported_rate,
            is_valid=is_valid,
            citation_audits=audits,
        )

