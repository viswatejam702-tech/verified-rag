# 🛡️ Verified RAG: Retrieval-Augmented Generation with Claim-Level NLI Verification

A production-grade, state-of-the-art **Verified RAG** system designed to eliminate LLM hallucinations. It retrieves knowledge chunks via vector search, reranks them using Cross-Encoders, generates grounded answers with citations, and executes atomic claim extraction paired with **Natural Language Inference (NLI)** verification to flag unsupported assertions sentence-by-sentence in real time.

---

## 📐 Architecture Pipeline

```
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │  1. Ingestion   │  ──▶  │  2. Retrieval   │  ──▶  │   3. Rerank     │
  │  Text Chunking  │       │ Dense Embedding │       │  Cross-Encoder  │
  │  & pgvector DB  │       │ (MiniLM / BGE)  │       │ (MS-MARCO Mini) │
  └─────────────────┘       └─────────────────┘       └─────────────────┘
                                                               │
                                                               ▼
  ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
  │ 6. Hallucination│  ◀──  │ 5. Verification │  ◀──  │ 4. Generation   │
  │ Metric & Report │       │  NLI Entailment │       │ LLM Grounded    │
  │ Sentence Colors │       │  DeBERTa / NLI  │       │ Answer + Source │
  └─────────────────┘       └─────────────────┘       └─────────────────┘
```

---

## 🌟 Key Features

1. **Precision Vector Embedding Service (`:8001`)**:
   - Computes dense vector representations using `sentence-transformers/all-MiniLM-L6-v2` (384 dims).
   - High-throughput batch inference with cosine similarity matrix support.
2. **Cross-Encoder Reranking Service (`:8002`)**:
   - Deep joint attention over (query, document) pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Filters out irrelevant top-K vector search noise and orders passages with relevance logits.
3. **Claim-Level NLI Verification Service (`:8003`)**:
   - Deconstructs LLM responses into atomic propositional claims.
   - Evaluates each claim against retrieved evidence using `cross-encoder/nli-deberta-v3-base`.
   - Produces fine-grained verdicts:
     - 🟢 **`VERIFIED` (Entailment)**: Directly supported by context with high confidence.
     - 🔴 **`CONTRADICTION` (Hallucinated)**: Contradicts or falsifies source material.
     - 🟡 **`UNSUPPORTED` (Neutral)**: Unsupported speculation without context evidence.
   - Calculates **Faithfulness Score** ($0.0 \to 1.0$) and **Hallucination Rate**.
4. **Interactive Next.js Dashboard (`:3000`)**:
   - Modern glassmorphic chat interface with inline color-coded claim inspection.
   - Interactive evidence drawer linking each sentence to its exact citation passage.
   - Real-time pipeline visualizer showing intermediate logits and chunk scores.
   - Knowledge Base manager to drag & drop / ingest documents.
   - Standalone Evaluation benchmark runner.

---

## 🚀 Quick Start

### 1. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Set your Supabase credentials or use the built-in standalone mock mode for local offline testing.

### 2. Database Migration

Run the provided SQL schema once in your Supabase SQL Editor:
```bash
psql $DATABASE_URL -f schema.sql
```

### 3. Running with Docker Compose (All Services)

```bash
docker-compose up --build
```
This boots up:
- Next.js Web App on `http://localhost:3000`
- Embeddings Service on `http://localhost:8001`
- Rerank Service on `http://localhost:8002`
- Verification Service on `http://localhost:8003`

### 4. Running Services Locally (Standalone Development)

#### Terminal 1: Embed Service
```bash
cd embed-service
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8001 --reload
```

#### Terminal 2: Rerank Service
```bash
cd rerank-service
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8002 --reload
```

#### Terminal 3: Verification Service
```bash
cd verification-service
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8003 --reload
```

#### Terminal 4: Next.js Frontend
```bash
cd next-app
npm install
npm run dev
```

---

## 📚 Ingesting Seed Data

To populate your vector database with 10 research documents:
```bash
cd seed-data
python load_seed.py
```

---

## 📊 Running Evaluation Benchmark

To run automated faithfulness, precision, and hallucination-detection benchmarks:
```bash
cd eval
python benchmark.py
```

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/ingest` | `POST` | Chunks & embeds documents into Supabase / vector store |
| `/api/retrieve` | `POST` | Vector similarity retrieval + Cross-Encoder reranking |
| `/api/generate` | `POST` | Grounded LLM generation with citation tags |
| `/api/verify` | `POST` | Sentence claim extraction & NLI hallucination scoring |
| `/api/eval` | `POST` | Full evaluation metrics calculation |

---

## 📜 License
MIT License. Built for rigorous, zero-hallucination AI research & enterprise workflows.
