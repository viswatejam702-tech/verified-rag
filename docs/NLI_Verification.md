# Verified RAG — Calibrated NLI Verification & Claim Decomposition

## 1. Atomic Propositional Decomposition
Rather than scoring an entire paragraph as a monolithic block, Verified RAG deconstructs responses into individual atomic assertions:
- **Filtering**: Strips conversational preambles ("Certainly! Here is the summary...").
- **Citation Extraction**: Isolates bracketed passage pointers (`[1]`, `[2]`).
- **Proposition Isolation**: Breaks complex compound sentences into independent atomic assertions.

---

## 2. 5-Way Categorical Verification States

| Category | Definition | Action |
|---|---|---|
| **`VERIFIED_ENTAILMENT`** | Evidence directly entails the claim with probability $P_e \ge 0.65$. | Marked green; passed to user. |
| **`CONTRADICTION`** | Evidence contradicts claim facts or numbers with $P_c \ge 0.50$. | Marked red; flagged as severe hallucination. |
| **`NEUTRAL_UNSUPPORTED`** | Evidence is silent or incomplete regarding the assertion. | Marked yellow; flagged as unverified speculation. |
| **`EVIDENCE_INSUFFICIENT`** | Context passages are empty, fragmented, or corrupt. | Triggers model abstention. |
| **`UNVERIFIABLE`** | Subjective opinion, future projection, or rhetorical remark. | Excluded from factual score. |

---

## 3. Faithfulness Score & Hallucination Rate Formulations

$$\text{Faithfulness Score} = \frac{\sum_{i=1}^N \mathbb{I}(\text{status}_i = \text{VERIFIED})}{\text{Total Non-Trivial Claims}}$$

$$\text{Hallucination Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{status}_i \in \{\text{CONTRADICTION}, \text{NEUTRAL}\})}{\text{Total Non-Trivial Claims}}$$

---

## 4. Configurable Abstention Policy
When $\text{Faithfulness Score} < 0.50$ or $\text{Contradiction Count} > 0$, Verified RAG suppresses ungrounded generations and returns a calibrated abstention:
*"The retrieved evidence is insufficient to verify this assertion without risk of hallucination."*

