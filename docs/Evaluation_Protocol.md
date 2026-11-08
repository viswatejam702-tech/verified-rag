# Verified RAG — Retrieval Evaluation Protocol & Metrics

## 1. Information Retrieval Metrics
Evaluates retrieval performance across ranked passages:

### Recall@K
$$\text{Recall}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

### Precision@K
$$\text{Precision}@K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

### Mean Reciprocal Rank (MRR)
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Where $\text{rank}_i$ is the rank position of the first relevant passage.

### Normalized Discounted Cumulative Gain (nDCG@K)
$$\text{DCG}@K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}, \quad \text{nDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$

---

## 2. RAG-Specific Context Quality Metrics
- **Context Precision**: The proportion of retrieved context chunks that contain actual signal relevant to answering the query, weighted by their rank.
- **Context Recall**: The extent to which the retrieved passages span all ground-truth facts necessary to compose the complete reference answer.

