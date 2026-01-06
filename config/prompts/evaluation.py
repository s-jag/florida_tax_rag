"""Prompts for LLM-based evaluation.

These prompts are used by:
- src/evaluation/llm_judge.py
"""

from __future__ import annotations

JUDGE_PROMPT = """You are evaluating a Florida tax law Q&A system's response.

## Question
{question}

## Expected Answer Type
{expected_type}

## Key Concepts That Should Appear
{expected_contains}

## Expected Citations
- Statutes: {expected_statutes}
- Rules: {expected_rules}

## Evaluator Notes
{notes}

## Generated Answer to Evaluate
{generated_answer}

---

Evaluate the answer and return a JSON object with these fields:

{{
    "correctness": <0-10 score for factual accuracy>,
    "completeness": <0-10 score for coverage of relevant points>,
    "clarity": <0-10 score for organization and readability>,
    "citation_accuracy": <0-10 score for proper citation usage>,
    "hallucinations": [<list any invented or incorrect facts>],
    "missing_concepts": [<list expected concepts not covered>],
    "overall_score": <0-10 overall quality score>,
    "reasoning": "<brief explanation of your evaluation>"
}}

Scoring Guidelines:
- 0-3: Poor/Incorrect - Major errors or missing critical information
- 4-6: Partial - Some correct information but significant gaps or minor errors
- 7-8: Good - Mostly correct with minor issues
- 9-10: Excellent - Comprehensive, accurate, well-cited

Be strict about hallucinations - any invented statute numbers, incorrect rates, or fabricated rules should be flagged.
"""
