# Verified RAG — Advanced Retrieval Methodology & Formulations

## 1. Dense Vector Retrieval
Dense retrieval models project queries and document chunks into a joint continuous metric space:

$$\text{Sim}(q, d) = \frac{\mathbf{v}_q \cdot \mathbf{v}_d}{\|\mathbf{v}_q\| \|\mathbf{v}_d\|}$$

Where $\mathbf{v}_q, \mathbf{v}_d \in \mathbb{R}^{384}$ are generated using `sentence-transformers/all-MiniLM-L6-v2`.

---

## 2. Sparse Lexical Retrieval (Okapi BM25)
BM25 scores documents based on term saturation and length normalization:

$$\text{BM25}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

Where:
- $\text{IDF}(t) = \ln\left(1 + \frac{N - n(t) + 0.5}{n(t) + 0.5}\right)$
- $k_1 = 1.5$ (term frequency saturation parameter)
- $b = 0.75$ (document length penalty ratio)
- $\text{avgdl}$: average document length across indexed corpus

---

## 3. Hybrid Reciprocal Rank Fusion (RRF)
Combines dense and sparse ranked retrieval lists without requiring score normalization:

$$RRF(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{k + \text{rank}_m(d)}$$

Where $k = 60$ acts as a ranking smoothing constant.

---

## 4. Cross-Encoder Joint Reranking
While bi-encoders generate separate embeddings $\mathbf{v}_q, \mathbf{v}_d$, cross-encoders concatenate query and passage into a single transformer input $[CLS] \circ q \circ [SEP] \circ d \circ [SEP]$, allowing full cross-attention across all token pairs:

$$s(q, d) = \sigma(\mathbf{W} \cdot \text{Transformer}([CLS] \circ q \circ [SEP] \circ d))$$

Yields superior relevance filtering at the expense of $O(K)$ computational complexity.

