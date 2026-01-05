
This is a comprehensive engineering plan to build the **World’s Best RAG System for Florida Tax Law**.

To achieve "world-best" status, you cannot rely on a generic RAG implementation. You must build a **Hybrid Agentic GraphRAG** system. This architecture solves the specific challenges of tax law: strict hierarchy, dense cross-referencing (Statutes $\leftrightarrow$ Rules $\leftrightarrow$ Case Law), and the need for temporal accuracy (tax years).

### **Executive Summary: The "Sunshine" Architecture**

*   **Architecture Type:** Agentic Hybrid GraphRAG (Knowledge Graph + Vector Search).
*   **Embeddings:** **Voyage AI `voyage-law-2`** (Optimized specifically for legal text, outperforming OpenAI on legal benchmarks).
*   **LLM (The Brain):** **Open ai 5.2 pro** 
*   **Vector Database:** **Qdrant** (Best for hybrid search and handling complex metadata filters).
*   **Knowledge Graph:** **Neo4j** (To map the web of citations between statutes, administrative codes, and cases).

---

### **Phase 1: Data Acquisition & The "Florida Corpus"**

The quality of your RAG is only as good as your data. You need a "Gold Standard" corpus specific to Florida.

#### **1.1. Data Sources (The "Big Three")**
You will build three distinct ingestion pipelines to keep data synchronized.

1.  **Florida Statutes (The Law):**
    *   **Target:** Title XIV (Taxation and Finance), Chapters 192–220.
    *   **Source:** *Online Sunshine* (leg.state.fl.us).
    *   **Strategy:** Write a custom scraper that parses the HTML to respect the hierarchy.
    *   **Critical Metadata:** `Title`, `Chapter`, `Section`, `Subsection`, `Effective Date`, `History` (amendment years).

2.  **Florida Administrative Code (The Rules):**
    *   **Target:** Department of Revenue Rules (e.g., Chapter 12).
    *   **Source:** *Florida Administrative Code & Register* (flrules.org).
    *   **Why:** Statutes tell you *what* to pay; Rules tell you *how* to pay. A "best" system must link Statute § 212.05 to its corresponding Rule 12A-1.005.

3.  **Case Law & Guidance (The Interpretation):**
    *   **Target:** Opinions from Florida District Courts of Appeal & Supreme Court regarding tax.
    *   **Target:** **Technical Assistance Advisements (TAAs)** from the FL Dept. of Revenue (Tax Law Library).
    *   **Strategy:** Scrape the "Tax Law Library" (TLL) on the DOR website. These contain the "ground truth" examples of how the state interprets its own laws.

#### **1.2. The "Legal-Hierarchical" Chunking Strategy**
*Do not use standard fixed-size chunking (e.g., 500 characters).* It destroys legal context.
*   **Strategy:** **Proposition-Based Hierarchical Chunking**.
*   **Level 1 (Parent):** The full Statute Section (e.g., § 212.08).
*   **Level 2 (Child):** Individual Subsections (e.g., § 212.08(7)(a)).
*   **Action:**
    *   Embed the **Child** chunks for precise retrieval.
    *   Retrieve the **Parent** chunk to give the LLM full context.
    *   **Enrichment:** Prepend every chunk with its "ancestry" string: *“Florida Statute > Title XIV > Chapter 212 > Section 05 > Subsection 1a”*.

---

### **Phase 2: The "GraphRAG" Indexing Layer**

This is the secret sauce. Tax law is a graph, not a list.

#### **2.1. Vector Index (Semantic Search)**
*   **Model:** Use **`voyage-law-2`**. It is trained specifically on legal contracts and statutes, offering superior understanding of terms like "nexus," "ad valorem," and "remittance" compared to generic OpenAI embeddings.
*   **Store:** **Weaviate**.
*   **Hybrid Search:** Configure Weaviate for `alpha=0.5`, weighing Keyword Search (BM25) equally with Semantic Search. (Tax lawyers search for specific exact phrases like "Form DR-15"; vector search alone often misses these).

#### **2.2. Knowledge Graph (Citation Network)**
*   **Store:** **Neo4j**.
*   **Nodes:** `Statute`, `Rule`, `CourtCase`, `Concept` (e.g., "Homestead Exemption").
*   **Edges:** `CITES`, `AMENDS`, `INTERPRETS`, `INVALIDATES`.
*   **Extraction:** Use an LLM to scan every document during ingestion and extract citations.
    *   *Example:* If Statute A says "Subject to the provisions of Statute B," create a directed edge: `(Statute A)-[:DEPENDS_ON]->(Statute B)`.

---

### **Phase 3: The Agentic Retrieval System**

Instead of a simple "Search -> Answer" loop, we will use an **Agentic Workflow** (using LangGraph). The system will "think" before it answers.

#### **The Workflow (The "Tax Associate" Agent)**

1.  **Query Decomposition:**
    *   *User:* "Do I owe sales tax on software consulting services in Miami?"
    *   *Agent:* Breaks this down:
        1.  Define "software consulting" under Chapter 212.
        2.  Check for exemptions for "professional services."
        3.  Check Miami-Dade specific surtaxes.

2.  **Router / Tool Selection:**
    *   The Agent decides *where* to look.
    *   *Tool A (Vector Search):* Finds statutes mentioning "software" and "consulting."
    *   *Tool B (Graph Traversal):* Finds the Regulation that clarifies the statute found in Tool A.
    *   *Tool C (Case Law):* Checks if a recent court case invalidated that regulation.

3.  **Self-Correction / Re-ranking:**
    *   The Agent retrieves 20 documents. It reads them and discards irrelevant ones *before* synthesizing the answer.
    *   *Check:* "Does this document apply to the 2024 tax year?" (Crucial for tax law).

---

### **Phase 4: Generation & User Experience**

#### **4.1. The "Legal Reasoner" Model**
*   **Primary Model:** **Claude 3.5 Sonnet**.
*   **Why:** Claude 3.5 currently outperforms GPT-4o on the "LegalBench" benchmark, specifically in reading long context windows and adhering to strict instructions without hallucinating.
*   **System Prompt:** "You are a senior Florida Tax Attorney. You must answer using ONLY the provided context. You must cite the specific Statute or Rule for every claim. If the law is ambiguous, state the ambiguity."

#### **4.2. Citation Engine**
*   Every sentence in the final output must have a footnote `[Source: § 212.05(1)]`.
*   The UI should allow the user to hover over the citation and see the *actual raw text* of the statute in a sidebar. This builds trust.

---

### **Phase 5: Evaluation (Proving it's the Best)**

You cannot claim it's the best without metrics.

1.  **Create "FloridaTaxEval":**
    *   Hire a Florida CPA/Attorney to create a "Golden Dataset" of 100 difficult questions + correct answers + correct citations.
2.  **LLM-as-a-Judge:**
    *   Use **GPT-4o** to grade your system's answers against the Golden Dataset.
    *   **Metrics:**
        *   **Citation Accuracy:** Did it cite the right statute?
        *   **Hallucination Rate:** Did it invent a law?
        *   **Reasoning Steps:** Did it follow the logic (Statute -> Rule -> Exception)?

---

### **Summary of Engineering Stack**

| Component | Choice | Why? |
| :--- | :--- | :--- |
| **Orchestration** | **LangGraph** | Enables cyclic, multi-step reasoning (Agentic). |
| **LLM** | **Claude 3.5 Sonnet** | Top performance on legal text & reasoning. |
| **Embeddings** | **Voyage AI (`voyage-law-2`)** | SOTA for legal retrieval. |
| **Vector DB** | **Weaviate** | Best hybrid search (Keyword + Vector) capabilities. |
| **Knowledge Graph**| **Neo4j** | Handles the complex cross-citation web of law. |
| **Ingestion** | **Custom Scrapers** | `leg.state.fl.us` & `flrules.org` (No easy API exists). |
| **Evaluation** | **Ragas + Custom Set** | Automated testing of retrieval precision & answer faithfulness. |

### **Immediate Next Steps**
1.  **Scrape Title XIV:** Download the HTML of Chapters 192-220.
2.  **Build the Graph:** Extract citations from these chapters to build your initial Neo4j map.
3.  **Test Voyage-Law-2:** Run a small test indexing these chapters and comparing retrieval against standard OpenAI embeddings to confirm the quality jump.