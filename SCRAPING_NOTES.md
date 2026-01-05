# Florida Tax Law Scraping Notes

This document details the data sources, scraping strategies, and challenges encountered when building the Florida Tax RAG knowledge base.

## Data Sources Overview

| Source | Type | Count | Status |
|--------|------|-------|--------|
| Florida Statutes | Chapters 198-220 | 744 sections | Complete |
| Florida Admin Code | Chapter 12A | 101 rules | Complete |
| Technical Assistance Advisements | PDFs | ~100+ | Partial (recent only) |
| Florida Case Law | CourtListener API | 308+ cases | In Progress |

## 1. Florida Statutes

**Source:** `leg.state.fl.us/Statutes/`

**Strategy:**
- Navigate through Title XIV (Taxation and Finance)
- Scrape chapters 198-220 (tax-related)
- Parse HTML to extract section text and metadata

**Output:** `data/raw/statutes/` - 744 JSON files across 17 chapters

**Challenges:**
- Complex HTML structure with nested tables
- Cross-reference links between sections
- History/amendment notes embedded in text

## 2. Florida Administrative Code

**Source:** `flrules.org`

**Strategy:**
- Scrape Chapter 12A (Department of Revenue) rules
- Division 12A contains ~100+ rules
- Parse rule text, effective dates, and statutory references

**Output:** `data/raw/admin_code/` - 101 JSON files

**Key Fields Extracted:**
- `rulemaking_authority`: Statutes granting rulemaking power
- `law_implemented`: Statutes the rule implements
- Links enable building statute-to-rule relationships

## 3. Technical Assistance Advisements (TAAs)

**Source:** `floridarevenue.com/TaxLaw/`

**Challenges:**
- SharePoint-based document library
- Dynamic JavaScript content loading
- httpx blocked; required curl fallback
- Only recent TAAs visible on results page
- Historical TAAs may require direct URL enumeration

**Workaround:**
```python
# SharePoint serves different content to httpx vs curl
if "TaxLaw/Documents" not in html:
    html = await self._fetch_search_page_with_curl()
```

**Output:** `data/raw/taa/` - PDFs and extracted JSON

**TAA Naming Convention:**
- Format: `YY[A-Z]NN-NNN.pdf`
- Year: 2-digit (25 = 2025)
- Tax Type: Single letter code
  - A = Sales and Use Tax
  - B = Corporate Income Tax
  - C = Documentary Stamp Tax
  - D = Property Tax
  - etc.
- Number: Sequential within year/tax type

**Parsed Metadata:**
- Question/issue posed
- Answer/response from DOR
- Statutes cited
- Rules cited (12A-1.xxx)

## 4. Florida Case Law

**Source:** CourtListener REST API (`courtlistener.com/api/rest/v4/`)

**Advantages:**
- Free, no authentication required
- Well-structured JSON responses
- Includes PDF download URLs
- Case citation graph (cases_cited field)

**Query Strategy:**
```
q="department of revenue"
court=fla (Florida Supreme Court)
type=o (opinions only)
```

**Coverage:**
- 308+ Florida Supreme Court cases
- Could expand to District Courts of Appeal (flaapp)

**Key Fields:**
- `cluster_id`: Unique case identifier
- `opinions[].snippet`: Text excerpt
- `opinions[].cites`: Case citation IDs
- `opinions[].download_url`: Direct PDF link

**Output:** `data/raw/case_law/` - JSON per case

## Citation Extraction

All scrapers extract cross-references for knowledge graph construction:

### Statute Citations
Pattern: `§ 212.05`, `Section 212.02(10)(i), F.S.`

### Rule Citations
Pattern: `Rule 12A-1.073, F.A.C.`, `12A-1.007(13)(e)`

### Case Citations
CourtListener provides `cites` array with cluster IDs, enabling direct graph edges.

## Data Quality Notes

1. **Statutes**: High quality, complete coverage of tax chapters
2. **Admin Code**: Complete for 12A, includes statutory cross-references
3. **TAAs**: Only recent documents accessible via web; historical may need direct URLs
4. **Cases**: Good coverage, though some older cases may lack full text

## Rate Limiting

All scrapers implement polite delays:
- Default: 0.5-1.0 seconds between requests
- Configurable via `--delay` CLI flag
- Caching prevents redundant fetches

## Usage

```bash
# Statutes (already complete)
python scripts/scrape_statutes.py --chapters 212

# Admin Code
python scripts/scrape_admin_code.py --division 12A --delay 0.5

# TAAs
python scripts/scrape_taa.py --max 20

# Case Law
python scripts/scrape_case_law.py --court fla --delay 0.5
```

## Future Improvements

1. **TAA Historical Access**: Enumerate URLs by year/type pattern
2. **District Courts**: Add `flaapp` court to case scraper
3. **Full Opinion Text**: Download PDFs for complete case text
4. **Incremental Updates**: Track last scrape date, fetch only new content
