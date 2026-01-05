# Processed Corpus

This directory contains the consolidated Florida Tax Law corpus, ready for downstream processing (chunking, embedding, graph construction).

## Files

| File | Description |
|------|-------------|
| `corpus.json` | Unified corpus containing all legal documents |
| `statistics.json` | Consolidation statistics and metrics |

## Corpus Schema

### Top-Level Structure

```json
{
  "metadata": {
    "created_at": "ISO timestamp",
    "total_documents": 1152,
    "by_type": {
      "statute": 742,
      "rule": 101,
      "taa": 1,
      "case": 308
    },
    "version": "1.0"
  },
  "documents": [...]
}
```

### LegalDocument Schema

Each document in the `documents` array follows the unified `LegalDocument` schema:

```json
{
  "id": "string",           // Unique identifier (e.g., "statute:212.05", "case:1093614")
  "doc_type": "string",     // One of: "statute", "rule", "taa", "case"
  "title": "string",        // Document title or name
  "full_citation": "string", // Canonical legal citation

  "text": "string",         // Plain text content of the document

  "effective_date": "date|null", // Effective/filing date (YYYY-MM-DD or null)

  "source_url": "string",   // Original source URL

  "parent_id": "string|null",    // Parent document ID (e.g., chapter for section)
  "children_ids": ["string"],    // Child document IDs (e.g., subsections)

  "cites_statutes": ["string"],  // Statute citations (e.g., ["212.05", "212.02(10)(i)"])
  "cites_rules": ["string"],     // Rule citations (e.g., ["12A-1.073"])
  "cites_cases": ["string"],     // Case document IDs (e.g., ["case:1093614"])

  "scraped_at": "datetime",      // When the document was scraped
  "metadata": {}                 // Type-specific extra fields
}
```

### ID Generation

Document IDs use a `{type}:{identifier}` format:

| Type | ID Format | Example |
|------|-----------|---------|
| Statute | `statute:{section}` | `statute:212.05` |
| Rule | `rule:{rule_number}` | `rule:12A-1.001` |
| TAA | `taa:{taa_number}` | `taa:TAA_25A-009` |
| Case | `case:{cluster_id}` | `case:1093614` |

### Type-Specific Metadata

Each document type stores additional fields in the `metadata` object:

#### Statutes
```json
{
  "title_number": 14,
  "title_name": "TAXATION AND FINANCE",
  "chapter": 212,
  "chapter_name": "TAX ON SALES...",
  "section_name": "Tax on sales...",
  "subsection": null,
  "history": ["1949", "1961", "2024"]
}
```

#### Rules
```json
{
  "chapter": "12A-1",
  "rulemaking_authority": ["212.18(2)", "213.06(1)"],
  "law_implemented": ["212.05", "212.08(7)(f)"],
  "references_statutes": ["213.255(2)", "213.37"]
}
```

#### TAAs
```json
{
  "tax_type": "Sales and Use Tax",
  "tax_type_code": "A",
  "topics": [],
  "question": "Whether the violation fee...",
  "answer": "Sales tax was collected on...",
  "pdf_path": "data/raw/taa/pdfs/25A-009.pdf"
}
```

#### Cases
```json
{
  "case_name_full": "Full case name with parties",
  "citations": ["215 So. 3d 46"],
  "court": "Supreme Court of Florida",
  "court_id": "fla",
  "docket_number": "SC17-1947",
  "judges": "Per Curiam",
  "pdf_url": "http://www.floridasupremecourt.org/...",
  "has_opinion_html": false
}
```

## Cross-Reference Fields

The corpus preserves cross-references for knowledge graph construction:

| Field | Description | Source |
|-------|-------------|--------|
| `cites_statutes` | Raw statute citations | Rules (rulemaking_authority, law_implemented), TAAs (statutes_cited), Cases (statutes_cited) |
| `cites_rules` | Raw rule citations | TAAs (rules_cited) |
| `cites_cases` | Case document IDs | Cases (cases_cited converted to "case:{id}" format) |

## Statistics

The `statistics.json` file contains:

- Processing timestamp and duration
- Document counts by type
- Cross-reference totals
- Text length statistics (count, avg, min, max) per document type

## Regenerating the Corpus

To regenerate the corpus from raw data:

```bash
python scripts/consolidate_corpus.py
```

To run a data quality audit first:

```bash
python scripts/audit_raw_data.py
```

## Data Quality Notes

From the audit (see `data/raw/audit_report.json`):

- **Statutes**: All 742 sections missing `effective_date` (scraper doesn't parse this)
- **Cases**: 285 of 308 cases have truncated opinions (<500 chars) - this is a CourtListener API limitation
- **Cases**: Only 13 cases have `statutes_cited` populated
- **Cross-references**: 2,469 case-to-case citations preserved
