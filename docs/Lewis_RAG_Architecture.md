# Lewis et al. (NeurIPS 2020) RAG Architecture Specification

**Reference Paper:** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela — NeurIPS 2020)

---

## 1. Architectural Overview & Workflow

The Lewis et al. RAG architecture models conditional sequence generation by combining a **non-parametric dense retriever** $p_\eta(z|x)$ with a **parametric sequence-to-sequence generator** $p_\theta(y|x, z)$. The system marginalizes over retrieved latent documents $z$ to produce the target output $y$.

```
                       ┌─────────────────────────────────────────────────────────────┐
                       │                   End-to-End Backprop                       │
                       │               through q and p_theta                         │
                       └─────────────────────────────────────────────────────────────┘
                               ▲                                           │
                               │                                           ▼
┌──────────────────┐    ┌───────────────────────────────────┐    ┌─────────────────────────────────┐    ┌──────────────────────────────┐
│  Input Query (x) │───▶│       Retriever p_η(z|x)          │───▶│        Generator p_θ            │───▶│     Generated Output (y)     │
│                  │    │  Non-Parametric Dense Index       │    │   Parametric Transformer        │    │                              │
│  QA:             │    │                                   │    │                                 │    │  QA:                         │
│  Define "middle  │    │  q(x) ──▶ [ MIPS ] ──▶ d(z_1)     │    │  Marginalize:                   │    │  The middle ear includes the │
│  ear"            │    │                         d(z_2)    │    │  RAG-Sequence:                  │    │  tympanic cavity and the     │
│                  │    │                         d(z_3)    │    │  p(y|x) = ∑ p(z|x) ∏ p(y_i|...) │    │  three ossicles.             │
│  Fact Verif:     │    │                         d(z_4)    │    │                                 │    │                              │
│  Barack Obama    │    │                                   │    │  RAG-Token:                     │    │  Fact Verif:                 │
│  was born in HI. │    │                                   │    │  p(y|x) = ∏ ∑ p(z|x) p(y_i|...) │    │  supports (y)                │
│                  │    │                                   │    │                                 │    │                              │
│  Jeopardy:       │    │                                   │    │                                 │    │  Jeopardy:                   │
│  The Divine      │    │                                   │    │                                 │    │  This 14th century work is   │
│  Comedy          │    │                                   │    │                                 │    │  divided into 3 sections...  │
└──────────────────┘    └───────────────────────────────────┘    └─────────────────────────────────┘    └──────────────────────────────┘
```

---

## 2. Mathematical Formulation

### 2.1 Non-Parametric MIPS Retriever $p_\eta(z|x)$
The retriever computes the dense representation of the input prompt $x$ using a neural encoder $q(x) \in \mathbb{R}^d$ and compares it to pre-indexed document passages $d(z) \in \mathbb{R}^d$ stored in a dense vector index (e.g., FAISS / MIPS):

$$\text{MIPS}(x, z) = q(x)^\top d(z)$$

The retrieval probability distribution over candidate passages $z \in \text{top-}k$ is given by temperature-scaled softmax:

$$p_\eta(z \mid x) = \frac{\exp\left(\frac{q(x)^\top d(z)}{\tau}\right)}{\sum_{z' \in \text{top-}k} \exp\left(\frac{q(x)^\top d(z')}{\tau}\right)}$$

---

### 2.2 Parametric Generator $p_\theta(y|x, z)$ with Marginalization

#### A. RAG-Sequence Formulation
In RAG-Sequence, the model assumes that the entire target sequence $y = (y_1, y_2, \dots, y_N)$ is generated conditioned on the same retrieved document $z$. The marginal likelihood is computed by summing the complete joint probabilities over the top-$k$ latent documents:

$$p_{\text{RAG-Seq}}(y \mid x) \approx \sum_{z \in \text{top-}k} p_\eta(z \mid x) \prod_{i=1}^N p_\theta(y_i \mid x, z, y_{1:i-1})$$

*Decoding Strategy:* Fast decoding evaluates top candidate sequences generated under each $z$, followed by marginalized likelihood reranking.

#### B. RAG-Token Formulation
In RAG-Token, the model marginalizes over documents at each token step, allowing different passages to explain different factual clauses in the generated sequence:

$$p_{\text{RAG-Tok}}(y \mid x) \approx \prod_{i=1}^N \left[ \sum_{z \in \text{top-}k} p_\eta(z \mid x) p_\theta(y_i \mid x, z, y_{1:i-1}) \right]$$

*Per-Token Attribution:* At each step $i$, the active document weight is:

$$w_i(z) = p_\eta(z \mid x) \cdot p_\theta(y_i \mid x, z, y_{1:i-1})$$

---

## 3. The Three Downstream Task Heads (Figure 1)

The Lewis et al. paper demonstrated that this single architecture natively supports three distinct downstream NLP tasks without task-specific modifications to the retriever:

### 3.1 Task Head A: Open-Domain Question Answering (QA)
- **Input Query ($x$):** Direct natural language question, e.g.:
  $$\text{Define "middle ear"}$$
- **Retrieved Document ($z$):** Anatomical passage describing the middle ear.
- **Generated Answer ($y$):**
  $$\text{"The middle ear includes the tympanic cavity and the three ossicles."}$$

### 3.2 Task Head B: Fact Verification (FEVER Benchmark)
- **Input Query ($x$):** Factual claim to verify, e.g.:
  $$\text{"Barack Obama was born in Hawaii."}$$
- **Retrieved Document ($z$):** Biographical excerpt verifying birth location in Honolulu, Hawaii.
- **Generated Label ($y$):**
  $$y \in \{\texttt{supports}, \texttt{refutes}, \texttt{not\_enough\_info}\}$$
  Bayesian marginalization evaluates evidence consistency across top-$k$ passages to emit the classification label.

### 3.3 Task Head C: Jeopardy Question Generation
- **Input Query ($x$):** Target entity / answer, e.g.:
  $$\text{"The Divine Comedy"}$$
- **Retrieved Document ($z$):** Encyclopedic article on Dante's masterwork.
- **Generated Clue ($y$):**
  $$\text{"This 14th century work is divided into 3 sections: 'Inferno', 'Purgatorio' \& 'Paradiso'"}$$

---

## 4. End-to-End Training & Backpropagation

In the end-to-end framework, the gradient with respect to retriever parameters $\eta$ is propagated directly through the marginal log-likelihood:

$$\nabla_\eta \log p(y \mid x) = \sum_{z \in \text{top-}k} \left( \frac{p_\eta(z \mid x) p_\theta(y \mid x, z)}{\sum_{z'} p_\eta(z' \mid x) p_\theta(y \mid x, z')} - p_\eta(z \mid x) \right) \nabla_\eta (q(x)^\top d(z))$$

This backpropagation automatically upweights passages that make the ground truth target $y$ more likely under generator $p_\theta$, training the non-parametric retriever without requiring gold passage labels.

---

## 5. Implementation in Verified RAG

In this platform, the Lewis RAG architecture operates alongside the **Cross-Encoder Claim-Level NLI Verification Pipeline**:

| Component | Lewis et al. (NeurIPS 2020) | Verified RAG Platform Extension |
|---|---|---|
| **Retriever** | Dense MIPS ($q(x)^\top d(z)$) | Hybrid Dense MIPS + BM25 Sparse + Cross-Encoder Reranking |
| **Generator** | BART-large with marginalization | Multi-provider LLM (GPT-4o, Gemini, Claude, Local) + Marginalization |
| **Verification** | FEVER 3-way token generation | Atomic Claim Extraction + DeBERTa-v3 Cross-Encoder NLI |
| **Citations** | Implicit latent document weights | Explicit passage citation verification & hallucination rate audit |

### Executable Python Module
- File: [`core/lewis_rag.py`](file:///c:/Users/VISWA/OneDrive/Pictures/Screenshots/apps%20play%20and%20app%20store/VTR/verified-rag/core/lewis_rag.py)
- Unit & Integration Test Suite: [`eval/test_lewis_rag.py`](file:///c:/Users/VISWA/OneDrive/Pictures/Screenshots/apps%20play%20and%20app%20store/VTR/verified-rag/eval/test_lewis_rag.py)
- REST API Endpoint: [`next-app/app/api/lewis-rag/route.ts`](file:///c:/Users/VISWA/OneDrive/Pictures/Screenshots/apps%20play%20and%20app%20store/VTR/verified-rag/next-app/app/api/lewis-rag/route.ts)
