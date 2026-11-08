# Verified RAG — System Limitations & Operational Boundaries

## 1. NLI Model Context Window
Cross-encoder models (`nli-deberta-v3-base`) have a standard sequence limit of 512 tokens. For long compound assertions, claims must be pre-chunked to avoid silent token truncation during pair scoring.

## 2. Low-Resource Language Grounding
While BM25 handles arbitrary character sets, dense embedding and cross-encoder models trained primarily on English may exhibit reduced recall on vernacular or low-resource languages unless fine-tuned.

## 3. Computational Latency Trade-Off
Executing joint cross-attention across candidate pairs and claim-level NLI introduces additional inference latency (~10–25ms per claim on GPU; ~50–150ms on CPU). High-throughput endpoints should configure batch verification and caching.

