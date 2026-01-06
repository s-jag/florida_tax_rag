# Evaluation Dataset and Methodology

This directory contains the golden evaluation dataset and documentation for measuring the Florida Tax RAG system's accuracy.

## Overview

The evaluation framework measures RAG system quality across multiple dimensions:

1. **Citation Accuracy** - Are the correct statutes and rules cited?
2. **Answer Completeness** - Are all key concepts covered?
3. **Factual Correctness** - Are the facts accurate (no hallucinations)?
4. **Clarity** - Is the answer well-organized and readable?

## Dataset Structure

### `golden_dataset.json`

Contains 20 seed evaluation questions with the following distribution:

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 5 | Direct statutory answers, single source |
| Medium | 10 | Requires connecting statute + rule |
| Hard | 5 | Requires case law/TAA interpretation, nuanced answers |

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
  "question": "The question to ask the RAG system",
  "category": "sales_tax",
  "difficulty": "easy|medium|hard",
  "expected_statutes": ["212.05", "212.08(1)"],
  "expected_rules": ["12A-1.001"],
  "expected_answer_contains": ["key", "phrases"],
  "expected_answer_type": "yes/no|numeric|it_depends|explanation",
  "notes": "Context for evaluators"
}
```

## Evaluation Metrics

### Citation Metrics

**Precision** = Correct Citations / All Generated Citations
- Measures: Are we citing relevant sources?
- High precision = Few irrelevant citations

**Recall** = Correct Citations / Expected Citations
- Measures: Are we finding all relevant sources?
- High recall = Not missing important citations

**F1 Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean of precision and recall

### Answer Quality Metrics

**Answer Contains Score** = Expected Phrases Found / Total Expected Phrases
- Measures: Does the answer include key concepts?

### LLM Judge Scores (0-10 scale)

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

## Running Evaluations

### Using the Evaluation Module

```python
from src.evaluation import (
    EvalDataset,
    EvalResult,
    LLMJudge,
    citation_precision,
    citation_recall,
    extract_citations_from_answer,
)

# Load dataset
import json
with open("data/evaluation/golden_dataset.json") as f:
    data = json.load(f)
dataset = EvalDataset(**data)

# Initialize judge
judge = LLMJudge(api_key="your-openai-key")

# Evaluate a single answer
result = await judge.judge_answer(
    question=dataset.questions[0],
    generated_answer="The Florida sales tax rate is 6%...",
)

print(f"Overall Score: {result.overall_score}/10")
print(f"Passed: {result.passed}")
```

### Calculating Metrics

```python
from src.evaluation import (
    citation_precision,
    citation_recall,
    extract_citations_from_answer,
    f1_score,
)

# Extract citations from generated answer
generated = extract_citations_from_answer(answer_text)

# Calculate metrics
precision = citation_precision(generated, expected_statutes, expected_rules)
recall = citation_recall(generated, expected_statutes, expected_rules)
f1 = f1_score(precision, recall)
```

## Extending the Dataset

When adding new questions:

1. Assign a unique ID (e.g., `eval_021`)
2. Select appropriate category and difficulty
3. List all expected statute and rule citations
4. Include key phrases that should appear in answer
5. Specify answer type for proper evaluation
6. Add evaluator notes explaining the expected answer

### Answer Types

| Type | Description | Example |
|------|-------------|---------|
| `yes/no` | Binary answer | "Are groceries exempt?" |
| `numeric` | Specific number/rate | "What is the tax rate?" |
| `it_depends` | Conditional answer | "Is SaaS taxable?" |
| `explanation` | Detailed explanation | "What are the requirements?" |

## Interpreting Results

### Target Metrics

| Metric | Target | Minimum |
|--------|--------|---------|
| Citation Precision | ≥ 0.80 | 0.60 |
| Citation Recall | ≥ 0.70 | 0.50 |
| F1 Score | ≥ 0.75 | 0.55 |
| Average Overall Score | ≥ 7.0 | 6.0 |
| Pass Rate | ≥ 70% | 50% |

### Common Issues

1. **Low Precision** - System citing too many irrelevant sources
2. **Low Recall** - System missing key citations
3. **Hallucinations** - System inventing facts or citations
4. **Low Completeness** - System not addressing all aspects

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-05 | Initial 20-question seed dataset |
