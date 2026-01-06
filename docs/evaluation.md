# Evaluation Guide

This guide covers the evaluation framework for measuring the Florida Tax RAG system's quality.

## Table of Contents

- [Overview](#overview)
- [Golden Dataset](#golden-dataset)
- [Metrics](#metrics)
- [LLM Judge](#llm-judge)
- [Retrieval Analysis](#retrieval-analysis)
- [Running Evaluations](#running-evaluations)
- [Interpreting Results](#interpreting-results)
- [Current Performance](#current-performance)
- [Improving Performance](#improving-performance)

---

## Overview

The evaluation framework measures RAG system quality across multiple dimensions:

| Dimension | Description | Metrics |
|-----------|-------------|---------|
| **Citation Accuracy** | Are the correct sources cited? | Precision, Recall, F1 |
| **Answer Completeness** | Are all key concepts covered? | Contains Score |
| **Factual Correctness** | Are the facts accurate? | LLM Judge Score |
| **Retrieval Quality** | Are relevant docs retrieved? | MRR, NDCG, Recall@k |

---

## Golden Dataset

### Location

```
data/evaluation/golden_dataset.json
```

### Structure

The dataset contains 20 expert-curated evaluation questions:

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 5 | Direct statutory answers, single source |
| Medium | 10 | Requires connecting statute + rule |
| Hard | 5 | Requires case law/TAA interpretation |

### Categories

- **sales_tax** - General sales tax questions
- **property_tax** - Property tax questions
- **corporate_tax** - Corporate income tax questions
- **exemptions** - Tax exemption questions
- **procedures** - Filing and administrative procedures

### Question Schema

```json
{
  "id": "eval_001",
  "question": "What is the Florida state sales tax rate?",
  "category": "sales_tax",
  "difficulty": "easy",
  "expected_statutes": ["212.05"],
  "expected_rules": ["12A-1.001"],
  "expected_answer_contains": ["6%", "six percent"],
  "expected_answer_type": "numeric",
  "notes": "Standard rate question - should cite 212.05"
}
```

### Question Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., `eval_001`) |
| `question` | string | The question to ask the system |
| `category` | string | Question category |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `expected_statutes` | list | Statute citations that should be referenced |
| `expected_rules` | list | F.A.C. rules that should be referenced |
| `expected_answer_contains` | list | Key phrases the answer should include |
| `expected_answer_type` | string | Answer type for evaluation |
| `notes` | string | Context for evaluators |

### Answer Types

| Type | Description | Example |
|------|-------------|---------|
| `yes/no` | Binary answer | "Are groceries exempt?" |
| `numeric` | Specific number/rate | "What is the tax rate?" |
| `it_depends` | Conditional answer | "Is SaaS taxable?" |
| `explanation` | Detailed explanation | "What are the requirements?" |

---

## Metrics

### Citation Metrics

**Precision** - Are we citing relevant sources?

```
Precision = Correct Citations / All Generated Citations
```

- High precision = Few irrelevant citations
- A citation is "correct" if it matches any expected citation

**Recall** - Are we finding all relevant sources?

```
Recall = Correct Citations / Expected Citations
```

- High recall = Not missing important citations

**F1 Score** - Harmonic mean of precision and recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Citation Matching

Citations are normalized before comparison:

```python
# These all match "212.05":
"Fla. Stat. § 212.05"
"§ 212.05(1)(a)"
"212.05"

# These all match "12A-1.001":
"Rule 12A-1.001"
"F.A.C. 12A-1.001"
"12a-1.001"
```

Subsection specificity is handled:

- Expected `212.05`, Generated `212.05(1)(a)` → Match
- Expected `212.08(1)`, Generated `212.08(1)(a)` → Match
- Expected `212.05`, Generated `212.06` → No match

### Answer Contains Score

```
Contains Score = Expected Phrases Found / Total Expected Phrases
```

Measures whether the answer includes key concepts from `expected_answer_contains`.

### Retrieval Metrics

**Mean Reciprocal Rank (MRR)**

```
MRR = 1 / rank of first relevant document
```

- MRR = 1.0 means relevant doc is first
- MRR = 0.5 means relevant doc is second
- MRR = 0 means no relevant doc found

**Normalized Discounted Cumulative Gain (NDCG@k)**

```
NDCG@k = DCG@k / IDCG@k
```

Measures ranking quality, penalizing relevant docs at lower ranks.

**Recall@k**

```
Recall@k = |relevant docs in top-k| / |total relevant docs|
```

Measures what fraction of relevant documents appear in top-k results.

**Precision@k**

```
Precision@k = |relevant docs in top-k| / k
```

Measures what fraction of top-k results are relevant.

---

## LLM Judge

The system uses GPT-4 as an LLM judge to evaluate answer quality on a 0-10 scale.

### Judge Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Factual accuracy, no errors |
| Completeness | 30% | Covers all aspects of question |
| Citation Accuracy | 20% | Cites correct sources |
| Clarity | 10% | Well-organized, readable |

### Scoring Rubric

| Score | Correctness | Completeness | Clarity | Citation Accuracy |
|-------|-------------|--------------|---------|-------------------|
| 0-3 | Major errors | Missing critical info | Confusing | Wrong citations |
| 4-6 | Some errors | Partial coverage | Acceptable | Some correct |
| 7-8 | Mostly correct | Good coverage | Clear | Mostly correct |
| 9-10 | Fully correct | Comprehensive | Excellent | All correct |

### Pass Criteria

An answer **passes** if:
- Overall score ≥ 7
- No hallucinations detected

### Configuration

```bash
# Required environment variable
export OPENAI_API_KEY="your-openai-api-key"
```

To skip the LLM judge (faster, metrics only):

```bash
python scripts/run_evaluation.py --no-judge
```

---

## Retrieval Analysis

### Alpha Parameter Tuning

The hybrid search alpha controls the balance between vector and keyword search:

- `alpha = 0.0` → Pure keyword (BM25)
- `alpha = 0.5` → Balanced hybrid
- `alpha = 1.0` → Pure vector

### Optimal Alpha: 0.25

Analysis shows **alpha=0.25 (keyword-heavy)** performs best for legal queries:

| Alpha | MRR | Recall@5 | Recall@10 | Interpretation |
|-------|-----|----------|-----------|----------------|
| **0.25** | **0.6125** | **0.5000** | **0.6000** | **Best overall** |
| 0.00 | 0.4154 | 0.5000 | 0.5000 | Pure keyword |
| 0.50 | 0.5133 | 0.5000 | 0.6000 | Balanced |
| 1.00 | 0.2533 | 0.2000 | 0.5000 | Pure vector |

### Why Keyword-Heavy Works

1. **Legal citations are exact terms** - "212.05" must match exactly
2. **Statutory language is precise** - BM25 captures exact terminology
3. **Vector search is too fuzzy** - Matches many tax-related chunks
4. **25% vector helps** - Captures semantic similarity for paraphrases

### Running Retrieval Analysis

```bash
# Full analysis with default settings
python scripts/analyze_retrieval.py

# Alpha tuning with custom values
python scripts/analyze_retrieval.py --tune-alpha --alphas "0.0,0.25,0.5,0.75,1.0"

# Debug a specific query
python scripts/analyze_retrieval.py --debug "What is the Florida sales tax rate?"

# Analyze single question
python scripts/analyze_retrieval.py --question eval_001

# Show only failed queries
python scripts/analyze_retrieval.py --failed-only

# Output JSON
python scripts/analyze_retrieval.py --json --output analysis.json
```

---

## Running Evaluations

### Full Evaluation

```bash
# Full evaluation with GPT-4 judge
python scripts/run_evaluation.py
```

**Output:**
- JSON report: `data/evaluation/reports/eval_{run_id}.json`
- Markdown report: `data/evaluation/reports/eval_{run_id}.md`

### Quick Check

```bash
# Test with 5 questions
python scripts/run_evaluation.py --limit 5

# Skip LLM judge (faster)
python scripts/run_evaluation.py --no-judge

# Both
python scripts/run_evaluation.py --limit 5 --no-judge
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `data/evaluation/golden_dataset.json` | Path to evaluation dataset |
| `--output` | `data/evaluation/reports` | Output directory for reports |
| `--limit` | All | Limit to N questions |
| `--no-judge` | False | Skip LLM judge (faster) |
| `--timeout` | 180 | Timeout per question in seconds |

### Programmatic Usage

```python
import asyncio
from src.agent.graph import create_tax_agent_graph
from src.evaluation import EvaluationRunner, LLMJudge

async def run_evaluation():
    # Initialize components
    graph = create_tax_agent_graph()
    judge = LLMJudge(api_key="your-openai-api-key")

    # Create runner
    runner = EvaluationRunner(
        agent=graph,
        judge=judge,
        dataset_path="data/evaluation/golden_dataset.json",
    )

    # Run evaluation
    report = await runner.run_all(limit=5, timeout_per_question=180)

    # Access results
    print(f"Pass Rate: {report.pass_rate:.1%}")
    print(f"Avg Score: {report.avg_overall_score:.1f}/10")
    print(f"Citation F1: {report.avg_citation_f1:.1%}")

asyncio.run(run_evaluation())
```

### Using Metrics Directly

```python
from src.evaluation import (
    citation_precision,
    citation_recall,
    f1_score,
    extract_citations_from_answer,
)

# Extract citations from generated answer
answer = "Under § 212.05, the Florida sales tax rate is 6%..."
citations = extract_citations_from_answer(answer)
# ['212.05']

# Calculate metrics
precision = citation_precision(
    generated=citations,
    expected_statutes=["212.05"],
    expected_rules=["12A-1.001"],
)
# 1.0 (212.05 is correct)

recall = citation_recall(
    generated=citations,
    expected_statutes=["212.05"],
    expected_rules=["12A-1.001"],
)
# 0.5 (only 1 of 2 expected citations found)

f1 = f1_score(precision, recall)
# 0.667
```

---

## Interpreting Results

### Example Output

```
============================================================
EVALUATION SUMMARY
============================================================
  Run ID: 2026-01-05_14-30-00
  Questions: 20/20
  Failed: 0

  Avg Overall Score: 7.5/10
  Pass Rate: 75%

  Citation Precision: 82.5%
  Citation Recall: 68.0%
  Citation F1: 74.5%
  Answer Contains: 71.0%

  Avg Latency: 3,250ms
  Total Hallucinations: 2

By Difficulty:
  easy    : 5 questions, score=8.5, pass=100%
  medium  : 10 questions, score=7.5, pass=80%
  hard    : 5 questions, score=6.0, pass=40%

By Category:
  sales_tax      : 8 questions, score=7.8, pass=88%
  exemptions     : 5 questions, score=7.2, pass=60%
  procedures     : 4 questions, score=7.5, pass=75%
  corporate_tax  : 3 questions, score=6.5, pass=67%
============================================================

Result: PASS (>=70% pass rate)
```

### Target Metrics

| Metric | Target | Minimum |
|--------|--------|---------|
| Citation Precision | ≥ 80% | 60% |
| Citation Recall | ≥ 70% | 50% |
| Citation F1 | ≥ 75% | 55% |
| Average Overall Score | ≥ 7.0 | 6.0 |
| Pass Rate | ≥ 70% | 50% |

### Common Issues

| Issue | Indicator | Likely Cause |
|-------|-----------|--------------|
| Low Precision | Many irrelevant citations | Retrieval returning too many docs |
| Low Recall | Missing key citations | Retrieval not finding relevant docs |
| Hallucinations | Judge flags fabricated facts | Generation inventing information |
| Low Completeness | Answer missing aspects | Retrieval or generation incomplete |

---

## Current Performance

### Baseline Results (v1.0.0)

| Metric | Value |
|--------|-------|
| Overall Score | 7.5/10 |
| Pass Rate | 75% |
| Citation Precision | 82.5% |
| Citation Recall | 68.0% |
| Citation F1 | 74.5% |
| MRR (alpha=0.25) | 0.6125 |
| Recall@10 | 60% |
| Avg Latency | 3,250ms |

### Performance by Difficulty

| Difficulty | Score | Pass Rate | Notes |
|------------|-------|-----------|-------|
| Easy | 8.5 | 100% | Single-source questions |
| Medium | 7.5 | 80% | Multi-source questions |
| Hard | 6.0 | 40% | Interpretation required |

---

## Improving Performance

### Retrieval Improvements

1. **Use optimal alpha (0.25)** - Already configured as default
2. **Increase top_k** - Retrieve more candidates for reranking
3. **Enable graph expansion** - Find related statutes via citations
4. **Add query expansion** - Expand legal terminology

### Generation Improvements

1. **Better hallucination detection** - Catch more fabricated facts
2. **Citation injection** - Ensure all retrieved docs are cited
3. **Self-correction loop** - Fix errors before final answer

### Dataset Improvements

1. **Add more hard questions** - Improve handling of edge cases
2. **Add case law questions** - Test TAA/court case retrieval
3. **Add multi-step questions** - Test complex reasoning

---

## Extending the Dataset

### Adding New Questions

1. Assign unique ID: `eval_021`, `eval_022`, etc.
2. Select appropriate category and difficulty
3. Research and list expected citations
4. Include key phrases for answer validation
5. Specify answer type
6. Add evaluator notes

### Example New Question

```json
{
  "id": "eval_021",
  "question": "How is cloud computing infrastructure taxed in Florida?",
  "category": "sales_tax",
  "difficulty": "hard",
  "expected_statutes": ["212.05(1)(i)"],
  "expected_rules": ["12A-1.032"],
  "expected_answer_contains": [
    "information services",
    "data processing",
    "communication services"
  ],
  "expected_answer_type": "it_depends",
  "notes": "Complex - depends on specific service type (IaaS vs SaaS vs PaaS)"
}
```

---

## See Also

- [Architecture](./architecture.md) - System overview
- [Configuration](./configuration.md) - Environment settings
- [API Reference](./api.md) - Query endpoints
- [data/evaluation/README.md](../data/evaluation/README.md) - Dataset details
- [RETRIEVAL_ANALYSIS.md](../RETRIEVAL_ANALYSIS.md) - Retrieval findings
