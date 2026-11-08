# 🛡️ Verified RAG — System Architecture Specification

## 1. Executive Overview
Verified RAG is a research-grade Retrieval-Augmented Generation system designed to eliminate ungrounded hallucinations through deterministic claim extraction, cross-encoder reranking, and 5-way Natural Language Inference (NLI) verification.

---

## 2. End-to-End System Topology

```
+---------------------------------------------------------------------------------------+
|                                VERIFIED RAG PIPELINE                                  |
+---------------------------------------------------------------------------------------+

 1. MULTI-FORMAT INGESTION ENGINE
    [PDF / DOCX / HTML / CSV / TXT / Markdown]
         │
         ▼
    [Content Hashing & Deduplication (SHA-256)]
         │
         ▼
    [Semantic Sentence Chunking with Page Provenance]
         │
         ├───▶ [Dense Embedding Service (:8001)] (384-dim all-MiniLM-L6-v2)
         └───▶ [BM25 Inverted Indexer] (Robertson-Spärck Jones Okapi BM25)
         │
         ▼
 2. COMPARATIVE DUAL RETRIEVAL & FUSION
    [Dense Vector Search]             [BM25 Sparse Lexical Search]
            │                                     │
            └───────────────┬─────────────────────┘
                            ▼
         [Reciprocal Rank Fusion (RRF: k=60)]
                            │
                            ▼
 3. CROSS-ENCODER RERANKING (:8002)
    [Joint Attention Reranker (ms-marco-MiniLM-L-6-v2)]
         │
         ▼ (Top-K Filtered Evidence Passages)
 4. ADVERSARIAL SECURITY & RBAC GUARD
    [Tenant Isolation & Indirect Prompt Injection Neutralizer]
         │
         ▼ (Sanitized Evidence Context)
 5. GROUNDED LLM GENERATION
    [LLM Answer Generation with Explicit Bracketed Citations [1], [2]]
         │
         ▼
 6. CLAIM-LEVEL DECOMPOSITION & VERIFICATION (:8003)
    [Atomic Propositional Claim Extractor]
         │
         ▼
    [5-Way Calibrated NLI Verifier (nli-deberta-v3-base)]
         │
         ├── 🟢 ENTAILMENT (Verified Evidence)
         ├── 🔴 CONTRADICTION (Hallucinated - Flagged)
         ├── 🟡 NEUTRAL_UNSUPPORTED (Unsubstantiated)
         ├── ⚪ EVIDENCE_INSUFFICIENT (Abstention Triggered)
         └── 🟣 UNVERIFIABLE (Subjective Modal)
         │
         ▼
 7. CITATION INTEGRITY & OBSERVABILITY
    [Citation Precision & Coverage Audit] ──▶ [Telemetry Tracing Engine]
```

---

## 3. Microservice Interfaces

| Service | Port | Primary Model / Engine | Role |
|---|---|---|---|
| **Next.js Dashboard** | `:3000` | React 18, Tailwind, Lucide | User Cockpit, Document Upload, Claim Drawer |
| **Embed Service** | `:8001` | `all-MiniLM-L6-v2` | Dense embedding vector computation |
| **Rerank Service** | `:8002` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder joint relevance scoring |
| **Verification Service** | `:8003` | `nli-deberta-v3-base` | Atomic claim extraction & 5-way NLI verification |

