# Retrieval Analysis Report

**Analysis Date:** 2026-01-06
**Dataset:** Golden Dataset v1.0.0 (5 easy questions analyzed)

---

## Executive Summary

Analysis of the Florida Tax RAG retrieval system reveals that **keyword-weighted hybrid search (alpha=0.25)** significantly outperforms pure vector or pure keyword approaches. The optimal alpha of 0.25 indicates that BM25 keyword matching is more effective than vector similarity for legal document retrieval, but semantic understanding from vectors still provides meaningful improvement.

---

## Key Findings

### 1. Optimal Alpha: 0.25 (Keyword-Heavy Hybrid)

The alpha parameter controls the balance between vector similarity (alpha=1.0) and keyword/BM25 matching (alpha=0.0).

| Alpha | MRR | Recall@5 | Recall@10 | NDCG@10 | Interpretation |
|-------|-----|----------|-----------|---------|----------------|
| **0.25** | **0.6125** | **0.5000** | **0.6000** | **0.7843** | **Best overall** |
| 0.00 | 0.4154 | 0.5000 | 0.5000 | 0.6475 | Pure keyword |
| 0.50 | 0.5133 | 0.5000 | 0.6000 | 0.7471 | Balanced |
| 0.75 | 0.3192 | 0.5000 | 0.5000 | 0.5251 | Vector-heavy |
| 1.00 | 0.2533 | 0.2000 | 0.5000 | 0.3660 | Pure vector |

**Key Observation:** Pure vector search (alpha=1.0) performs worst, with MRR of only 0.2533 vs 0.6125 for alpha=0.25. This suggests legal queries benefit significantly from exact term matching.

### 2. Vector vs Keyword Performance

| Metric | Pure Keyword (α=0.0) | Pure Vector (α=1.0) | Improvement |
|--------|---------------------|---------------------|-------------|
| MRR | 0.4154 | 0.2533 | +64% (keyword wins) |
| Recall@5 | 0.5000 | 0.2000 | +150% (keyword wins) |
| Recall@10 | 0.5000 | 0.5000 | Equal |

**Why Keyword Works Better for Legal Queries:**
- Legal citations (e.g., "212.05", "12A-1.001") are exact terms
- Statutory language uses precise terminology
- BM25 captures these exact matches effectively

**Why Some Vector Search Still Helps:**
- Captures semantic similarity for paraphrased concepts
- Handles synonyms and related terminology
- The 25% vector boost in alpha=0.25 captures this benefit

### 3. Retrieval Quality Metrics (5 Questions, Alpha=0.25)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean Reciprocal Rank | 0.6125 | Expected docs found at rank ~1.6 on average |
| Recall@5 | 0.5000 | 50% of expected docs in top 5 |
| Recall@10 | 0.6000 | 60% of expected docs in top 10 |
| NDCG@10 | 0.7843 | Good ranking quality |

---

## Sample Query Analysis

### Query: "What is the Florida state sales tax rate?"

- **Expected:** 212.05
- **Keyword (α=0.0):** Found at rank 2 (MRR=0.5)
- **Hybrid (α=0.25):** Found at rank 2 (MRR=0.5)
- **Vector (α=1.0):** Found at rank 12 (MRR=0.083)

The vector search struggles because "sales tax rate" semantically matches many tax-related chunks, while keyword search directly matches statute 212.05 which explicitly mentions "sales tax."

---

## Recommendations

### Immediate Actions

1. **Set default alpha to 0.25** in production
   ```python
   # In src/retrieval/hybrid.py
   def retrieve(self, query: str, alpha: float = 0.25, ...):
   ```

2. **Consider query-type adaptive alpha:**
   - Citation-heavy queries (e.g., "§ 212.05"): Use alpha=0.1 (more keyword)
   - Conceptual queries (e.g., "how is software taxed"): Use alpha=0.5 (more balanced)

### Future Improvements

3. **Expand retrieval window:** Current Recall@10 of 60% means 40% of relevant docs are missed. Consider:
   - Increasing top_k for retrieval
   - Adding query expansion for legal terminology

4. **Citation extraction boost:** When queries contain statute numbers, boost exact matches for those citations.

5. **Enable graph expansion:** Once Neo4j is available, analyze whether graph-based document expansion improves recall for related statutes and implementing rules.

---

## Technical Notes

### Environment Issues

- **Neo4j:** Authentication failed during analysis. Graph expansion comparison was skipped.
- **Weaviate:** Connected and functional
- **Voyage AI Embeddings:** Using `voyage-law-2` (1024-dim)

### Dataset Coverage

- 5 questions analyzed (easy difficulty)
- Categories: sales_tax, exemptions, corporate_tax, procedures
- Expected citations: Statutes (212.05, 212.08, 212.06, 220.11, 212.11) + Rules

---

## Appendix: CLI Usage

```bash
# Run alpha tuning
python scripts/analyze_retrieval.py --tune-alpha --alphas "0.0,0.25,0.5,0.75,1.0"

# Analyze single question
python scripts/analyze_retrieval.py --question eval_001

# Debug a specific query
python scripts/analyze_retrieval.py --debug "What is the Florida sales tax rate?"

# Full analysis (all 20 questions)
python scripts/analyze_retrieval.py --output RETRIEVAL_ANALYSIS.md
```
