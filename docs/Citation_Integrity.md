# Verified RAG — Citation Integrity & Provenance Auditing

## 1. Grounding Principles
LLMs frequently hallucinate source references by fabricating non-existent passage indices (e.g., citing `[Passage 8]` when only 3 passages were retrieved) or attaching citations to unrelated claims.

---

## 2. Citation Integrity Metrics

### Citation Precision
$$\text{Citation Precision} = \left(\frac{\text{Citations that Actually Support Attributed Claim}}{\text{Total Valid Retrieved Citations}}\right) \times 100$$

### Citation Coverage
$$\text{Citation Coverage} = \left(\frac{\text{Atomic Claims with Valid Citations}}{\text{Total Atomic Factual Claims}}\right) \times 100$$

### Invented Citation Detection
Any citation referring to a passage index $j \ge K_{\text{retrieved}}$ or non-existent document ID is automatically intercepted, flagged as an **Invented Reference**, and penalized in the final integrity audit.

