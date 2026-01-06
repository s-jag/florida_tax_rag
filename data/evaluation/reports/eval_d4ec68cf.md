# Florida Tax RAG Evaluation Report

**Run ID:** d4ec68cf
**Date:** 2026-01-06 02:53:05 UTC
**Dataset Version:** 1.0.0

## Summary

| Metric | Value |
|--------|-------|
| Questions Evaluated | 3/3 |
| Pass Rate | 0.0% |
| Avg Overall Score | 0.0/10 |
| Citation Precision | 76.7% |
| Citation Recall | 50.0% |
| Citation F1 | 60.5% |
| Answer Contains Score | 38.9% |
| Avg Latency | 71,550ms |
| Total Hallucinations | 0 |

## Results by Category

| Category | Count | Avg Score | Pass Rate | Precision | Recall | Hallucinations |
|----------|-------|-----------|-----------|-----------|--------|----------------|
| exemptions | 1 | 0.0 | 0% | 100% | 0% | 0 |
| sales_tax | 2 | 0.0 | 0% | 65% | 75% | 0 |

## Results by Difficulty

| Difficulty | Count | Avg Score | Pass Rate | Precision | Recall | Avg Latency |
|------------|-------|-----------|-----------|-----------|--------|-------------|
| easy | 3 | 0.0 | 0% | 77% | 50% | 71,550ms |

## Worst Performing Questions

### 1. eval_001 (easy, sales_tax) - Score: N/A

**Question:** What is the Florida state sales tax rate?

### 2. eval_002 (easy, exemptions) - Score: N/A

**Question:** Are groceries exempt from Florida sales tax?

### 3. eval_003 (easy, sales_tax) - Score: N/A

**Question:** Who is required to collect and remit Florida sales tax?


## Individual Results

| ID | Category | Difficulty | Score | Passed | Precision | Recall | Latency |
|----|----------|------------|-------|--------|-----------|--------|---------|
| eval_001 | sales_tax | easy | N/A | No | 80% | 100% | 100,950ms |
| eval_002 | exemptions | easy | N/A | No | 100% | 0% | 11,082ms |
| eval_003 | sales_tax | easy | N/A | No | 50% | 50% | 102,617ms |
