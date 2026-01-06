# How Margen Solves the Hallucination Problem in Regulatory AI

## Abstract

One major pain point of building AI systems for regulatory documents is that they hallucinate. Confidently. With citations that don't exist.

Ask a standard RAG system about tax exemptions, and it might cite "§ 212.08(7)(a)" with perfect confidence—except that citation doesn't say what the AI claims, or worse, doesn't exist at all. In regulated industries where accuracy isn't optional, this is a dealbreaker.

Here, we present Margen's approach to regulatory AI: a hybrid system that combines semantic search, exact term matching, knowledge graph relationships, and—critically—self-correction. The system doesn't just find relevant documents; it reasons about what it found, traces authority chains, and verifies every citation before responding.

Our main contributions:

- **Hybrid retrieval** that captures both conceptual meaning and exact terminology
- **Query decomposition** that breaks complex questions into focused sub-queries
- **Knowledge graph enhancement** that understands document relationships (statutes → rules → cases)
- **Self-correcting validation** that catches hallucinations before they reach the user

The result: high citation precision, zero fabricated references in our baseline testing, and answers that legal professionals can actually trust.

---

## The Problem with Traditional RAG

Before we explain our approach, let's look at why standard Retrieval-Augmented Generation fails on regulatory documents.

### Failure Mode 1: Chunking Destroys Legal Meaning

Every RAG system starts by splitting documents into chunks. The problem? Fixed-size chunking tears legal text apart mid-sentence.

Consider this statute:

```
§ 212.05 Sales Tax Rates

(1) For the exercise of such privilege, a tax is levied on each taxable
transaction at the rate of 6 percent of the sales price of each item
or article of tangible personal property.

(2) Exemptions to subsection (1) are provided in § 212.08.
```

A naive 200-character chunker produces:

#### Chunk 1
```
§ 212.05 Sales Tax Rates

(1) For the exercise of such privilege, a tax is levied on each taxable
transaction at the rate of 6 percent of the sales price of each item
or article of tan
```

#### Chunk 2
```
gible personal property.

(2) Exemptions to subsection (1) are provided in § 212.08.
```

Chunk 1 now says "tan" instead of "tangible personal property." Chunk 2 lost the context that it's part of § 212.05. The cross-reference to exemptions is severed from the tax rate it modifies.

Legal documents have structure. Chunking ignores it.

### Failure Mode 2: Vector Search Misses Exact Terms

Semantic embeddings are powerful—they understand that "levy" and "impose" mean similar things. But in regulatory contexts, exact terminology matters.

```
Query: "What does § 212.05(1)(a) say about sales tax?"

Vector search returns:
1. "Sales taxes are generally imposed on retail transactions..."
2. "The taxation of goods follows standard commercial principles..."
3. "Revenue collection mechanisms include various levies..."

What the user needed: The actual text of § 212.05(1)(a)
```

The embeddings found conceptually similar content, but missed the specific section the user asked for. A keyword search would have found it instantly—but keyword search alone misses conceptual connections.

Neither approach works in isolation.

### Failure Mode 3: No Understanding of Authority

Legal documents aren't created equal. A statute is binding law. An administrative rule interprets the statute. A court case applies the rule to specific facts. An advisory opinion is just guidance.

Traditional RAG treats them all the same:

```
Query: "Is software consulting taxable?"

Standard RAG returns (ranked by vector similarity):
1. Technical Advisory (2019): "Software services may be..." [0.89 similarity]
2. Blog post: "Many businesses wonder about..." [0.87 similarity]
3. § 212.05 Statute: "Taxable transactions include..." [0.82 similarity]
```

The advisory opinion—which isn't binding—ranks higher than the actual statute. A human researcher would never make this mistake. They know to start with the statute, then check implementing rules, then look at how courts have interpreted it.

Standard RAG has no concept of authority hierarchy.

---

## Our Approach: Hybrid Agentic GraphRAG

What if we combined semantic understanding, exact matching, and relationship awareness into a single system that could reason about what it found?

That's Margen.

### The Big Picture

```
┌─────────────────┐
│   User Query    │
│ "Is software    │
│  consulting     │
│  taxable?"      │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                     MARGEN PIPELINE                         │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Decompose│──▶│ Retrieve │──▶│  Expand  │──▶│ Validate │ │
│  │  Query   │   │ (Hybrid) │   │  (Graph) │   │ + Correct│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Verified Answer │
│ with Citations  │
│ [Source: §212.05]│
└─────────────────┘
```

Let's walk through each component.

---

### Hybrid Search: Best of Both Worlds

The first insight is simple: don't choose between semantic and keyword search. Use both.

```
Query: "What is the 6% rate applied to?"

┌─────────────────────────────────────────────────────┐
│              HYBRID SEARCH                          │
│                                                     │
│   Semantic Search          Keyword Search           │
│   ──────────────          ──────────────           │
│   "rate" → taxation       "6%" → exact match       │
│   "applied" → levied      "rate" → exact match     │
│   concepts related        in § 212.05(1)           │
│   to tax rates                                      │
│                                                     │
│              ↓ FUSION ↓                             │
│                                                     │
│   Results ranked by BOTH conceptual relevance      │
│   AND exact term matches                           │
└─────────────────────────────────────────────────────┘
```

When a user asks about "§ 212.05," the keyword component finds exact matches. When they ask about "sales tax exemptions for nonprofits," the semantic component understands the concept even if those exact words don't appear in the statute.

The fusion is tunable. For citation lookups, we weight keywords heavily. For conceptual questions, we weight semantics. The system adapts to what the query needs.

---

### Query Decomposition: Breaking Down Complexity

Real questions aren't simple. "Is software consulting taxable, and if so, what exemptions might apply for a nonprofit providing educational services?" contains at least four distinct sub-questions.

Traditional RAG throws the whole thing at a search engine and hopes for the best. Margen breaks it apart:

```
Original Query:
"Is software consulting taxable, and if so, what exemptions
might apply for a nonprofit providing educational services?"

                    ↓ DECOMPOSITION ↓

┌─────────────────────────────────────────────────────────────┐
│ Sub-query 1: "software consulting sales tax taxability"     │
│ Type: DEFINITION | Priority: 1                              │
├─────────────────────────────────────────────────────────────┤
│ Sub-query 2: "professional services tax exemptions"         │
│ Type: EXEMPTION | Priority: 1                               │
├─────────────────────────────────────────────────────────────┤
│ Sub-query 3: "nonprofit organization tax exemptions"        │
│ Type: EXEMPTION | Priority: 2                               │
├─────────────────────────────────────────────────────────────┤
│ Sub-query 4: "educational services tax treatment"           │
│ Type: EXEMPTION | Priority: 2                               │
└─────────────────────────────────────────────────────────────┘
```

Each sub-query is focused. Each retrieves highly relevant chunks for that specific aspect. The results are then merged, deduplicated, and ranked.

And just like that, we retrieve exactly what a human researcher would find—but in parallel, in seconds.

---

### Knowledge Graph Enhancement: Following the Authority Chain

Here's something standard RAG completely misses: legal documents cite each other. A lot.

A statute might be implemented by five administrative rules. Those rules might be interpreted by a dozen court cases. The cases might reference technical advisories for guidance.

Margen builds this into a knowledge graph:

```
                    ┌─────────────────┐
                    │   § 212.05      │
                    │   (Statute)     │
                    │   PRIMARY LAW   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │ IMPLEMENTS   │ IMPLEMENTS   │
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │Rule 12A-1│   │Rule 12A-2│   │Rule 12A-3│
        │(Admin)   │   │(Admin)   │   │(Admin)   │
        └────┬─────┘   └────┬─────┘   └──────────┘
             │              │
             │ INTERPRETS   │ INTERPRETS
             ▼              ▼
        ┌──────────┐   ┌──────────┐
        │ Case A   │   │ Case B   │
        │(Court)   │   │(Court)   │
        └──────────┘   └──────────┘
```

When you retrieve § 212.05, Margen automatically expands to find:
- Rules that implement this statute
- Cases that interpret it
- Advisory opinions that discuss it

This is how human researchers work. They don't stop at the first document they find—they trace the authority chain to build a complete picture.

The graph also enables smart ranking. Statutes outrank rules. Rules outrank advisories. Recent interpretations are weighted higher than old ones. The authority hierarchy is built into the retrieval.

---

### The Reasoning Loop: Thinking, Not Just Searching

Here's where Margen diverges most dramatically from traditional RAG.

Standard RAG is essentially: search → generate → done. One shot. No verification.

Margen implements a reasoning loop:

```
┌──────────────────────────────────────────────────────────────────┐
│                      REASONING LOOP                              │
│                                                                  │
│   ┌─────────┐                                                   │
│   │Decompose│ ─── Break query into focused sub-queries          │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │Retrieve │ ─── Hybrid search for each sub-query (parallel)   │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ Expand  │ ─── Follow knowledge graph relationships          │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │  Score  │ ─── LLM evaluates relevance of each chunk         │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │ Filter  │ ─── Remove low-quality results                    │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐                                                   │
│   │Synthesize│ ─── Generate answer with inline citations        │
│   └────┬────┘                                                   │
│        ▼                                                        │
│   ┌─────────┐      ┌─────────┐                                  │
│   │Validate │ ───▶ │ Correct │ ─── If issues found              │
│   └────┬────┘      └────┬────┘                                  │
│        │                │                                        │
│        ▼                ▼                                        │
│   ┌──────────────────────┐                                      │
│   │    Final Answer      │                                      │
│   │  (Verified, Cited)   │                                      │
│   └──────────────────────┘                                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

The system doesn't just search and respond. It evaluates what it found. It checks if it needs more information. It validates its own output before delivering it.

This is reasoning, not just retrieval.

---

### Hallucination Detection & Self-Correction

This is the heart of what makes Margen different.

After generating an answer, the system doesn't just ship it. It validates every claim against the source documents:

```
Generated claim: "The standard sales tax rate is 6% per § 212.05(1)"

                    ↓ VALIDATION ↓

┌─────────────────────────────────────────────────────────────┐
│ 1. Extract citation: § 212.05(1)                            │
│                                                             │
│ 2. Find source chunk containing § 212.05(1)                 │
│                                                             │
│ 3. Verify claim against source text:                        │
│    Source says: "...tax is levied...at the rate of 6        │
│    percent of the sales price..."                           │
│    Claim says: "standard sales tax rate is 6%"              │
│                                                             │
│ 4. Result: ✓ VERIFIED - claim matches source                │
└─────────────────────────────────────────────────────────────┘
```

But what happens when validation finds a problem?

The system classifies hallucinations by type and severity:

| Type | Description | What Margen Does |
|------|-------------|------------------|
| **Unsupported Claim** | Statement not backed by sources | Removes or qualifies |
| **Fabricated Citation** | Citation doesn't exist | Removes entirely |
| **Misquoted Text** | Quote doesn't match source | Corrects the quote |
| **Misattributed** | Right info, wrong source | Fixes the attribution |
| **Overgeneralization** | Claim too broad for source | Adds qualifiers |

Based on severity, the system routes to one of three paths:

1. **Accept**: Minor issues or none—response is good
2. **Correct**: Moderate issues—patch the specific problems
3. **Regenerate**: Severe issues—start over with better guidance

The key insight: hallucination detection isn't a post-hoc filter. It's integrated into the generation loop. The system catches its own mistakes before they reach the user.

---

## Caveats & Limitations

Of course, there are caveats.

**Latency**: Complex queries that trigger decomposition, graph expansion, and validation take longer than a simple vector search. For intricate multi-part questions, response times can reach tens of seconds. We're optimizing this, but accuracy comes first.

**Coverage**: The system is only as good as its source documents. If a statute was recently amended and we haven't ingested the update, the answer will be based on outdated law. We maintain regular update cycles, but there's inherent lag.

**Not Legal Advice**: Margen is a research tool, not a lawyer. It can find and synthesize regulatory information with high accuracy, but it cannot replace professional legal counsel for consequential decisions.

---

## Results

We've evaluated Margen against a curated test set of regulatory questions spanning different difficulty levels and document types.

**Citation Accuracy**:
- High precision: when Margen cites a source, that source overwhelmingly supports the claim
- Improving recall: we're continuously expanding coverage to find more relevant sources

**Hallucination Rate**:
- Zero fabricated citations in baseline testing
- Self-correction catches and fixes the vast majority of unsupported claims before they reach users

**Authority Awareness**:
- Statutes consistently ranked above advisory opinions
- Implementation chains (statute → rule → case) correctly traced
- Recent interpretations appropriately weighted

We're not publishing exact numbers here—competitive reasons—but we're confident enough to put this in production.

---

## Conclusion

The future of regulatory AI isn't "find similar text." It's about understanding relationships, verifying claims, and reasoning about what was found.

Traditional RAG fails on regulatory documents because it treats legal research like web search. It doesn't understand that statutes outrank advisories. It doesn't trace citation networks. It doesn't verify its own outputs.

Margen does.

By combining hybrid search, query decomposition, knowledge graphs, and self-correcting validation, we've built a system that legal professionals can actually trust. Not because it's perfect—no AI system is—but because it knows when it doesn't know, it shows its sources, and it catches its own mistakes.

And we're just getting started.

---

*For more information about Margen, contact us at [contact information].*
