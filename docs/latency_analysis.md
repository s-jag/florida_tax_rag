# Latency Analysis: Score Relevance & Synthesis Stages

This document analyzes the high latency observed in the `score_relevance` and `synthesis` pipeline stages, explains the root causes, and provides actionable optimization recommendations.

## Executive Summary

### Current Latency Profile

| Stage | Documented | Actual Observed | Root Cause |
|-------|------------|-----------------|------------|
| `score_relevance` | 500-1500ms | **2000-5000ms+** | LLM call per chunk, low parallelism |
| `synthesis` pipeline | 1000-3000ms | **2000-10000ms** | Multiple LLM calls, regeneration loops |

### Key Bottlenecks Identified

1. **Score Relevance:** Makes 20-35 individual LLM calls (one per chunk) with only 5 concurrent
2. **Synthesis:** Up to 3 sequential LLM calls, with regeneration loops that can triple latency

### Quick Wins (Implementable Today)

| Optimization | Expected Improvement | Effort |
|--------------|---------------------|--------|
| Increase parallelism to 20 | 50-70% faster scoring | 1 line change |
| Use Haiku for scoring | 60% faster per call | 1 line change |
| Reduce validator max_tokens | 10-20% faster validation | 1 line change |
| Skip validation for high confidence | 30-50% faster synthesis | ~20 lines |

---

## Score Relevance Deep Dive

### How It Works

The `score_relevance` node evaluates each retrieved chunk's relevance to the query using an LLM.

**Location:** `src/agent/nodes.py:321-423`

```
Query Flow:
retrieve (20 chunks) → expand_graph (+10-15 chunks) → SCORE_RELEVANCE → filter
```

**Current Implementation:**
```python
# For EACH chunk (20-35 total):
response = await asyncio.to_thread(
    client.messages.create,
    model="claude-sonnet-4-20250514",  # Expensive model
    max_tokens=150,
    messages=[{"role": "user", "content": prompt}],
)
# Parse JSON: {"score": 0-10, "reasoning": "..."}
```

### Why It's Slow

#### Problem 1: One LLM Call Per Chunk

| Metric | Value |
|--------|-------|
| Chunks from retrieval | 20 (default `top_k`) |
| Chunks from graph expansion | 10-15 additional |
| **Total LLM calls** | **25-35 calls** |
| Time per call | 200-500ms |

#### Problem 2: Low Parallelism (Semaphore=5)

```python
# src/agent/nodes.py:389
semaphore = asyncio.Semaphore(5)  # Only 5 concurrent calls!
```

**Batching Effect:**
```
30 chunks ÷ 5 concurrent = 6 sequential batches
6 batches × ~800ms per batch = 4.8 seconds
```

#### Problem 3: Heavyweight Model

Using `claude-sonnet-4-20250514` for simple relevance scoring is overkill:

| Model | Latency | Cost | Quality for Scoring |
|-------|---------|------|---------------------|
| Sonnet 4 | 200-500ms | $$$ | Overkill |
| **Haiku** | **50-150ms** | **$** | **Sufficient** |

#### Problem 4: Full Context in Prompt

Each scoring prompt includes up to 2000 characters of chunk text:

```python
# src/agent/nodes.py:360
text = chunk.get("text", "")[:2000]  # 2000 chars per call
```

### Code Locations for Fixes

| File | Line | Current | Recommended |
|------|------|---------|-------------|
| `src/agent/nodes.py` | 389 | `Semaphore(5)` | `Semaphore(20)` |
| `src/agent/nodes.py` | 365 | `claude-sonnet-4-20250514` | `claude-haiku-3-5-20241022` |
| `src/agent/nodes.py` | 360 | `[:2000]` | `[:1000]` |

### Optimization Options

#### Option 1: Increase Parallelism (Quick Win)

```python
# Change from:
semaphore = asyncio.Semaphore(5)
# To:
semaphore = asyncio.Semaphore(20)
```

**Impact:** 50-70% faster (6 batches → 2 batches)

#### Option 2: Use Haiku for Scoring (Quick Win)

```python
# Change from:
model="claude-sonnet-4-20250514"
# To:
model="claude-haiku-3-5-20241022"
```

**Impact:** 60% faster per call, 70% cost reduction

#### Option 3: Batch Multiple Chunks Per Call (Medium Effort)

Score 5 chunks in a single LLM call instead of 1:

```python
BATCH_RELEVANCE_PROMPT = """Score the relevance of these 5 documents to the query.
Query: {query}

Documents:
1. [{doc_type}] {citation_1}: {text_1}
2. [{doc_type}] {citation_2}: {text_2}
...

Return JSON: [{"chunk_id": "...", "score": 0-10}, ...]
"""
```

**Impact:** 80% fewer API calls (35 calls → 7 calls)

#### Option 4: Embedding-Based Pre-Filter (Long-term)

Use embedding similarity as a fast first pass:

```python
# Fast embedding similarity (already computed during retrieval)
embedding_scores = cosine_similarity(query_embedding, chunk_embeddings)

# Only LLM-score chunks with embedding_score > 0.3
chunks_to_score = [c for c in chunks if embedding_scores[c.id] > 0.3]
```

**Impact:** 50-80% fewer chunks to LLM-score

---

## Synthesis Pipeline Deep Dive

### How It Works

The synthesis pipeline generates an answer, validates it for hallucinations, and optionally corrects issues.

**Pipeline Flow:**
```
synthesize_answer → validate_response → [regenerate loop] → correct_response
       │                   │                    │                │
    Sonnet 4           Haiku              Back to start      Sonnet 4
   (1-2 sec)         (200-500ms)           (if failed)      (500-1000ms)
```

**Location:** `src/agent/nodes.py:578-685` (synthesize), `src/agent/edges.py:103-139` (routing)

### Why It's Slow

#### Problem 1: Multiple Sequential LLM Calls

| Stage | Model | Max Tokens | Typical Latency |
|-------|-------|------------|-----------------|
| `synthesize_answer` | Sonnet 4 | 2048 | 1000-2000ms |
| `validate_response` | Haiku | **4096** | 200-500ms |
| `correct_response` | Sonnet 4 | 2048 | 500-1000ms |

**Minimum total:** 1200ms (no regeneration)
**With correction:** 1700-3500ms

#### Problem 2: Regeneration Loops

From `src/agent/edges.py:103-139`:

```python
def route_after_validation(state):
    needs_regeneration = validation_data.get("needs_regeneration", False)
    regeneration_count = state.get("regeneration_count", 0)
    max_regenerations = state.get("max_regenerations", 2)  # Can loop 2x!

    if needs_regeneration and regeneration_count < max_regenerations:
        return "regenerate"  # Back to synthesize_answer!
```

**Worst Case Timeline:**
```
1st synthesis:    1500ms
1st validation:    400ms
→ needs_regeneration=True, loop back
2nd synthesis:    1500ms
2nd validation:    400ms
→ still failing, loop back
3rd synthesis:    1500ms
3rd validation:    400ms
→ max reached, force correction
correction:        800ms
─────────────────────────
TOTAL:           7000ms+ (7+ seconds)
```

#### Problem 3: Excessive max_tokens in Validator

```python
# src/generation/validator.py:38-53
response = await asyncio.to_thread(
    client.messages.create,
    model="claude-haiku-3-5-20241022",
    max_tokens=self.max_tokens,  # 4096 tokens!
    messages=[...],
)
```

The validator returns JSON with hallucination details. **4096 tokens is excessive** - even detailed validation output rarely exceeds 1000 tokens.

#### Problem 4: No Confidence-Based Skipping

From `src/agent/nodes.py:717-722`:

```python
# Always validates, no matter how confident the synthesis
validator = ResponseValidator()
result = await validator.validate_response(
    response=final_answer,
    context=context_text,
    query=state["original_query"],
)
```

High-confidence synthesis outputs (e.g., simple factual queries) still go through full validation.

#### Problem 5: Full Context Re-sent to Validator

The hallucination detection prompt re-sends ALL 10 chunks plus the response:

```python
HALLUCINATION_DETECTION_PROMPT = """
SOURCE DOCUMENTS:
{context}  # 10 chunks × 500-2000 tokens each = 5000-20000 tokens!

RESPONSE TO VERIFY:
{response}  # 500-2000 tokens
"""
```

**Token consumption per validation:** 6000-22000 input tokens

### Code Locations for Fixes

| File | Line | Current | Recommended |
|------|------|---------|-------------|
| `src/generation/validator.py` | ~45 | `max_tokens=4096` | `max_tokens=1500` |
| `src/generation/corrector.py` | ~44 | `claude-sonnet-4-20250514` | `claude-haiku-3-5-20241022` |
| `src/agent/state.py` | 74 | `max_regenerations=2` | `max_regenerations=1` |
| `src/agent/nodes.py` | 617 | `valid_chunks[:10]` | `valid_chunks[:5]` |

### Optimization Options

#### Option 1: Reduce Validator max_tokens (Quick Win)

```python
# Change from:
max_tokens=4096
# To:
max_tokens=1500
```

**Impact:** 10-20% faster validation, lower cost

#### Option 2: Use Haiku for Corrections (Quick Win)

```python
# src/generation/corrector.py
# Change from:
model="claude-sonnet-4-20250514"
# To:
model="claude-haiku-3-5-20241022"
```

**Impact:** 50-70% faster corrections

#### Option 3: Add Confidence-Based Validation Skip (Medium Effort)

```python
# src/agent/nodes.py - in synthesize_answer
if confidence_score > 0.85 and not state.get("force_validation", False):
    # Skip validation for high-confidence, simple queries
    return {
        "final_answer": answer,
        "validation_passed": True,  # Assumed
        "validation_skipped": True,
    }
```

**Impact:** 30-50% faster for high-confidence queries

#### Option 4: Reduce Regeneration Limit (Quick Win)

```python
# src/agent/state.py
# Change from:
max_regenerations: int = 2
# To:
max_regenerations: int = 1
```

**Impact:** Caps worst-case at 1 regeneration instead of 2

#### Option 5: Reduce Context Chunk Count (Medium Effort)

```python
# src/agent/nodes.py:617
# Change from:
context_chunks = valid_chunks[:10]
# To:
context_chunks = valid_chunks[:5]
```

**Impact:** 30-50% fewer tokens, faster LLM calls

---

## Recommended Optimization Roadmap

### Phase 1: Quick Wins (Day 1)

| Change | File | Line | Before | After |
|--------|------|------|--------|-------|
| Increase scoring parallelism | `nodes.py` | 389 | `Semaphore(5)` | `Semaphore(20)` |
| Use Haiku for scoring | `nodes.py` | 365 | `sonnet-4` | `haiku-3-5` |
| Reduce validator tokens | `validator.py` | ~45 | `4096` | `1500` |
| Reduce regenerations | `state.py` | 74 | `2` | `1` |

**Expected Result:** 40-60% latency reduction

### Phase 2: Medium Effort (Week 1)

| Change | Effort | Impact |
|--------|--------|--------|
| Use Haiku for corrections | 1 line | 50% faster corrections |
| Batch relevance scoring (5/call) | ~50 lines | 80% fewer API calls |
| Reduce context chunks to 5 | 1 line | 30% faster synthesis |
| Add confidence-based skip | ~20 lines | Skip validation 30% of queries |

**Expected Result:** Additional 30-40% latency reduction

### Phase 3: Long-term (Month 1+)

| Change | Effort | Impact |
|--------|--------|--------|
| Embedding-based relevance pre-filter | ~100 lines | 60% fewer scoring calls |
| Cache relevance scores | ~150 lines | Skip scoring for repeated queries |
| Streaming synthesis | ~200 lines | Perceived latency improvement |
| Parallel validation + correction | ~100 lines | 30% faster when both needed |

---

## Expected Impact Summary

### Before Optimization

```
Typical Query Latency Breakdown:
├── decompose:        300ms
├── retrieve:         800ms
├── expand_graph:     150ms
├── score_relevance: 4000ms  ← BOTTLENECK
├── filter:            10ms
├── temporal_check:    10ms
├── synthesize:      1500ms
├── validate:         400ms
├── correct:          800ms  (30% of queries)
└── TOTAL:          8000ms+ (8+ seconds)
```

### After Phase 1 Optimizations

```
Optimized Query Latency Breakdown:
├── decompose:        300ms
├── retrieve:         800ms
├── expand_graph:     150ms
├── score_relevance: 1200ms  ← 70% faster (Haiku + parallel)
├── filter:            10ms
├── temporal_check:    10ms
├── synthesize:      1500ms
├── validate:         350ms  ← 15% faster (reduced tokens)
├── correct:          800ms  (fewer due to 1 regen max)
└── TOTAL:          5000ms  (5 seconds, 37% improvement)
```

### After All Optimizations

```
Fully Optimized Latency:
├── decompose:        300ms
├── retrieve:         800ms
├── expand_graph:     150ms
├── score_relevance:  500ms  ← Batched + cached
├── filter:            10ms
├── temporal_check:    10ms
├── synthesize:      1200ms  ← 5 chunks instead of 10
├── validate:            0ms  ← Skipped (high confidence)
├── correct:             0ms  ← Not needed
└── TOTAL:          3000ms  (3 seconds, 62% improvement)
```

---

## Monitoring Latency

### Add Stage Timing Metrics

The API already returns `stage_timings` in the response. Monitor these fields:

```json
{
  "stage_timings": {
    "decompose": 312.5,
    "retrieve": 845.2,
    "expand_graph": 156.8,
    "score_relevance": 3892.1,  // Watch this
    "filter": 8.2,
    "temporal_check": 5.1,
    "synthesize": 1567.3,       // Watch this
    "validate": 423.7,
    "correct": 812.4
  }
}
```

### Key Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| `score_relevance` | <1500ms | >3000ms |
| `synthesize` | <2000ms | >4000ms |
| `validate` | <500ms | >1000ms |
| Total latency | <5000ms | >10000ms |
| Regeneration rate | <20% | >40% |

---

## Appendix: Token Cost Analysis

### Current Cost Per Query

| Stage | Model | Input Tokens | Output Tokens | Cost |
|-------|-------|--------------|---------------|------|
| decompose | Sonnet | ~500 | ~200 | $0.002 |
| score_relevance (×30) | Sonnet | ~600 each | ~50 each | $0.06 |
| synthesize | Sonnet | ~8000 | ~1500 | $0.03 |
| validate | Haiku | ~10000 | ~500 | $0.003 |
| correct (30%) | Sonnet | ~12000 | ~1500 | $0.01 |
| **TOTAL** | | | | **~$0.10/query** |

### Optimized Cost Per Query

| Stage | Model | Input Tokens | Output Tokens | Cost |
|-------|-------|--------------|---------------|------|
| decompose | Sonnet | ~500 | ~200 | $0.002 |
| score_relevance (×6 batched) | Haiku | ~3000 each | ~200 each | $0.005 |
| synthesize | Sonnet | ~4000 | ~1500 | $0.02 |
| validate (skipped 50%) | Haiku | ~5000 | ~500 | $0.001 |
| correct (10%) | Haiku | ~6000 | ~1000 | $0.002 |
| **TOTAL** | | | | **~$0.03/query** |

**Cost reduction: 70%**
