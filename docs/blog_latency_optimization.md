# Taming the Latency Beast: How We Cut RAG Response Times by 60%

*A practical guide to optimizing Retrieval-Augmented Generation systems for production*

---

When we first deployed our legal research RAG system, the results were impressive—accurate answers with proper citations, hallucination detection, and source verification. There was just one problem: users were waiting 8-10 seconds for each response. In a world of instant search results, that felt like an eternity.

This is the story of how we diagnosed the bottlenecks and reduced our p95 latency from 10 seconds to under 4 seconds—without sacrificing quality.

## The Hidden Cost of "Doing It Right"

RAG systems are deceptively complex. What looks like a simple question-answer flow actually involves multiple stages:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG Pipeline Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   User Query                                                         │
│       │                                                              │
│       ▼                                                              │
│   ┌─────────────┐                                                   │
│   │  Decompose  │ ─── Break complex queries into sub-queries        │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │  Retrieve   │ ─── Vector search + keyword search                │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │   Score     │ ─── Evaluate relevance of each chunk      ← SLOW │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │ Synthesize  │ ─── Generate answer with citations        ← SLOW │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│   ┌─────────────┐                                                   │
│   │  Validate   │ ─── Check for hallucinations              ← SLOW │
│   └──────┬──────┘                                                   │
│          │                                                           │
│          ▼                                                           │
│      Response                                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

Each of these stages serves a purpose. But when you're making multiple LLM calls with large context windows, the latency adds up fast.

## Profiling: Finding Where Time Goes

Before optimizing, we needed to understand where our time was actually going. We instrumented every stage of our pipeline:

```
┌────────────────────────────────────────────────────────────────────┐
│                    Original Latency Breakdown                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Decompose      ████                               300ms (4%)      │
│  Retrieve       ████████                           800ms (10%)     │
│  Score          ████████████████████████████████   4000ms (50%)  ← │
│  Synthesize     ████████████████                   1500ms (19%)    │
│  Validate       ████████                           800ms (10%)     │
│  Correct        ██████                             600ms (7%)      │
│                                                                     │
│  TOTAL: ~8000ms                                                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

Two stages dominated: **relevance scoring** (50%) and the **synthesis pipeline** (36% including validation and correction).

## The Relevance Scoring Trap

Our relevance scoring seemed straightforward: for each retrieved chunk, ask an LLM "how relevant is this to the query?" and get a 0-10 score. Clean, accurate, interpretable.

The problem? We were scoring 25-35 chunks per query, and each scoring call was a full LLM inference.

### What We Found

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Relevance Scoring: Before                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Retrieved Chunks: 30                                               │
│   Concurrent Calls: 5  (semaphore limit)                            │
│   Batches Required: 6                                                │
│   Time per Batch: ~700ms                                            │
│                                                                      │
│   Timeline:                                                          │
│   ═══════════════════════════════════════════════════════════       │
│   Batch 1   Batch 2   Batch 3   Batch 4   Batch 5   Batch 6         │
│   [=====]   [=====]   [=====]   [=====]   [=====]   [=====]         │
│   700ms     700ms     700ms     700ms     700ms     700ms           │
│                                                                      │
│   Total: 4200ms                                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

We were using a powerful (expensive, slow) model for a simple classification task, with conservative parallelism that serialized most of the work.

### The Fix: Right-Size the Model and Parallelize

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Relevance Scoring: After                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Retrieved Chunks: 30                                               │
│   Concurrent Calls: 20 (increased parallelism)                      │
│   Batches Required: 2                                                │
│   Time per Batch: ~400ms (faster model)                             │
│                                                                      │
│   Timeline:                                                          │
│   ═══════════════════════════════════════════════════════════       │
│   Batch 1              Batch 2                                       │
│   [=================]  [=================]                           │
│   400ms                400ms                                         │
│                                                                      │
│   Total: 800ms (5x faster)                                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight:** Relevance scoring is a classification task, not a generation task. A smaller, faster model performs nearly as well at a fraction of the latency and cost.

## The Synthesis Pipeline: Death by a Thousand Cuts

Our synthesis pipeline was designed for quality: generate an answer, validate it for hallucinations, and correct any issues. Noble goals, but the implementation had compounding latency problems.

### The Regeneration Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Synthesis Pipeline: Worst Case                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                    ┌─────────────┐                                  │
│                    │  Synthesize │ 1500ms                           │
│                    └──────┬──────┘                                  │
│                           │                                          │
│                           ▼                                          │
│                    ┌─────────────┐                                  │
│             ┌──────│  Validate   │ 500ms                            │
│             │      └──────┬──────┘                                  │
│             │             │                                          │
│             │             ▼                                          │
│   Regenerate│      ┌─────────────┐                                  │
│   (if bad)  │      │  Decision   │                                  │
│             │      └──────┬──────┘                                  │
│             │             │                                          │
│             │    ┌────────┴────────┐                                │
│             │    │                 │                                 │
│             │    ▼                 ▼                                 │
│             └──► Regenerate      Correct                            │
│                  (loop back)      │                                  │
│                                   ▼                                  │
│                               ┌─────────────┐                       │
│                               │   Correct   │ 800ms                 │
│                               └─────────────┘                       │
│                                                                      │
│   Worst case: 2 regeneration loops + correction                     │
│   = 1500 + 500 + 1500 + 500 + 1500 + 500 + 800 = 6800ms            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

When validation detected potential issues, it could trigger regeneration—sending us back through the entire synthesis loop. With a maximum of 2 regenerations allowed, worst-case queries took nearly 7 seconds just in this stage.

### Over-Provisioned Token Limits

We also discovered our validation step was configured with a 4096 token limit for responses—even though validation output (a JSON object with detected issues) rarely exceeded 500 tokens. This seemingly minor configuration was adding measurable overhead to every request.

### The Fixes

1. **Cap regeneration loops**: Reduced maximum regenerations from 2 to 1
2. **Right-size token limits**: Reduced validation output tokens from 4096 to 1500
3. **Use faster models for corrections**: Switched correction step to a smaller model
4. **Smart validation skipping**: Skip validation entirely for high-confidence simple queries

```
┌─────────────────────────────────────────────────────────────────────┐
│              Confidence-Based Validation Skip                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   if (confidence >= 0.85 AND is_simple_query):                      │
│       skip_validation()  # Save 500-800ms                           │
│                                                                      │
│   Simple queries: Single-fact questions with direct source matches  │
│   High confidence: Strong citation verification, consistent sources │
│                                                                      │
│   Impact: ~30% of queries skip validation entirely                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## The Results

After implementing these optimizations:

```
┌────────────────────────────────────────────────────────────────────┐
│                    Optimized Latency Breakdown                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Decompose      ████                               300ms (9%)      │
│  Retrieve       ████████                           800ms (24%)     │
│  Score          ████████                           800ms (24%)   ✓ │
│  Synthesize     ████████████                       1200ms (36%)    │
│  Validate       ███                                250ms (7%)    ✓ │
│  Correct        (rarely triggered)                 0ms            ✓ │
│                                                                     │
│  TOTAL: ~3350ms (58% reduction)                                    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| p50 Latency | 6.5s | 2.8s | 57% faster |
| p95 Latency | 10.2s | 4.1s | 60% faster |
| API Cost | $0.10/query | $0.03/query | 70% cheaper |

## Lessons Learned

### 1. Profile Before You Optimize

We assumed synthesis was the bottleneck—it's the most complex stage. But profiling revealed relevance scoring consumed half our time. Without measurement, we would have optimized the wrong thing.

### 2. Match Model Size to Task Complexity

Not every task needs your most powerful model. Classification, scoring, and simple corrections can use smaller, faster models without quality loss. Save the big models for tasks that need them.

### 3. Sequential Calls Are Silent Killers

Even fast operations become slow when serialized. Our 5-request semaphore seemed reasonable until we calculated that it forced 6 sequential batches for 30 chunks. Increasing parallelism had outsized impact.

### 4. Design for the Common Case

Most of our queries were simple fact lookups that didn't need extensive validation. By identifying these cases and fast-pathing them, we improved average latency without compromising quality for complex queries.

### 5. Configuration Defaults Matter

A 4096 token limit on a response that never exceeds 500 tokens seems harmless. But these small inefficiencies compound across every request. Audit your defaults.

## What's Next

These quick wins got us to acceptable latency, but there's more to explore:

- **Batched scoring**: Score multiple chunks in a single LLM call
- **Embedding-based pre-filtering**: Use vector similarity as a fast first pass before LLM scoring
- **Streaming responses**: Return partial results while synthesis completes
- **Caching**: Cache relevance scores for repeated query patterns

## Conclusion

RAG systems optimize for quality by default—multiple retrieval passes, relevance filtering, hallucination detection. These are valuable features. But production systems need to balance quality with latency.

The good news: there's usually low-hanging fruit. By profiling our pipeline, right-sizing our models, increasing parallelism, and adding smart fast-paths, we cut latency by 60% and costs by 70%—while maintaining the quality that made our system valuable in the first place.

The key is treating latency as a first-class concern, not an afterthought. Instrument your stages. Question your defaults. And remember: the fastest code is the code that doesn't run.

---

*Have questions about RAG optimization? Found different bottlenecks in your system? We'd love to hear about it.*
