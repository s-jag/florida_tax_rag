#!/usr/bin/env python3
"""Data quality audit script for the Florida Tax RAG corpus.

This script analyzes all raw scraped data, identifies quality issues,
and generates a comprehensive audit report.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_json_file(path: Path) -> dict | None:
    """Load a JSON file, returning None on error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return {"_error": str(e), "_path": str(path)}


def check_encoding_issues(text: str) -> list[str]:
    """Check for encoding problems in text."""
    issues = []
    # Check for replacement characters
    if "\ufffd" in text:
        issues.append("contains_replacement_chars")
    # Check for null bytes
    if "\x00" in text:
        issues.append("contains_null_bytes")
    # Check for other control characters (except newline, tab, carriage return)
    control_chars = set(chr(i) for i in range(32) if i not in (9, 10, 13))
    if any(c in text for c in control_chars):
        issues.append("contains_control_chars")
    return issues


def validate_date(date_str: str | None) -> tuple[bool, str | None]:
    """Validate a date string. Returns (is_valid, error_message)."""
    if date_str is None:
        return True, None  # None is valid (optional field)

    # Try common date formats
    formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z"]
    for fmt in formats:
        try:
            parsed = datetime.strptime(
                date_str.split("+")[0].split("Z")[0], fmt.split("%z")[0].rstrip()
            )
            # Check reasonable date range (1800-2100)
            if parsed.year < 1800 or parsed.year > 2100:
                return False, f"date_out_of_range: {date_str}"
            return True, None
        except ValueError:
            continue
    return False, f"invalid_date_format: {date_str}"


def audit_statutes(statutes_dir: Path) -> dict:
    """Audit all statute files."""
    issues = []
    stats = {
        "count": 0,
        "by_chapter": Counter(),
        "text_lengths": [],
        "missing_fields": Counter(),
        "empty_text": 0,
        "missing_effective_date": 0,
    }
    ids_seen = set()

    required_fields = ["section", "chapter", "title_number", "title"]

    for chapter_dir in sorted(statutes_dir.glob("chapter_*")):
        for json_file in sorted(chapter_dir.glob("*.json")):
            data = load_json_file(json_file)
            if data is None:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "statute",
                        "issue": "file_read_error",
                        "severity": "error",
                    }
                )
                continue

            if "_error" in data:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "statute",
                        "issue": "json_parse_error",
                        "details": data["_error"],
                        "severity": "error",
                    }
                )
                continue

            stats["count"] += 1
            metadata = data.get("metadata", {})
            text = data.get("text", "")

            # Generate ID
            section = metadata.get("section", "")
            doc_id = f"statute:{section}"

            # Check for duplicates
            if doc_id in ids_seen:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "statute",
                        "issue": "duplicate_id",
                        "id": doc_id,
                        "severity": "warning",
                    }
                )
            ids_seen.add(doc_id)

            # Track by chapter
            chapter = metadata.get("chapter", "unknown")
            stats["by_chapter"][str(chapter)] += 1

            # Check required fields
            for field in required_fields:
                if not metadata.get(field):
                    stats["missing_fields"][field] += 1
                    issues.append(
                        {
                            "file": str(json_file.relative_to(DATA_DIR)),
                            "type": "statute",
                            "issue": "missing_field",
                            "field": field,
                            "severity": "warning",
                        }
                    )

            # Check text
            stats["text_lengths"].append(len(text))
            if len(text) < 50:
                stats["empty_text"] += 1
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "statute",
                        "issue": "empty_or_short_text",
                        "length": len(text),
                        "severity": "warning",
                    }
                )

            # Check encoding
            encoding_issues = check_encoding_issues(text)
            for issue in encoding_issues:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "statute",
                        "issue": issue,
                        "severity": "warning",
                    }
                )

            # Check effective_date
            if metadata.get("effective_date") is None:
                stats["missing_effective_date"] += 1

    # Compute statistics
    if stats["text_lengths"]:
        stats["avg_text_length"] = sum(stats["text_lengths"]) // len(stats["text_lengths"])
        stats["min_text_length"] = min(stats["text_lengths"])
        stats["max_text_length"] = max(stats["text_lengths"])
    del stats["text_lengths"]  # Don't include raw list in output
    stats["by_chapter"] = dict(stats["by_chapter"])
    stats["missing_fields"] = dict(stats["missing_fields"])

    return {"issues": issues, "stats": stats, "unique_ids": len(ids_seen)}


def audit_rules(rules_dir: Path) -> dict:
    """Audit all administrative code rule files."""
    issues = []
    stats = {
        "count": 0,
        "text_lengths": [],
        "missing_fields": Counter(),
        "empty_text": 0,
        "with_rulemaking_authority": 0,
        "with_law_implemented": 0,
    }
    ids_seen = set()

    required_fields = ["rule_number", "chapter", "title"]

    for chapter_dir in sorted(rules_dir.glob("chapter_*")):
        for json_file in sorted(chapter_dir.glob("*.json")):
            data = load_json_file(json_file)
            if data is None or "_error" in data:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "rule",
                        "issue": "file_read_error",
                        "severity": "error",
                    }
                )
                continue

            stats["count"] += 1
            metadata = data.get("metadata", {})
            text = data.get("text", "")

            # Generate ID
            rule_number = metadata.get("rule_number", "")
            doc_id = f"rule:{rule_number}"

            # Check for duplicates
            if doc_id in ids_seen:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "rule",
                        "issue": "duplicate_id",
                        "id": doc_id,
                        "severity": "warning",
                    }
                )
            ids_seen.add(doc_id)

            # Check required fields
            for field in required_fields:
                if not metadata.get(field):
                    stats["missing_fields"][field] += 1
                    issues.append(
                        {
                            "file": str(json_file.relative_to(DATA_DIR)),
                            "type": "rule",
                            "issue": "missing_field",
                            "field": field,
                            "severity": "warning",
                        }
                    )

            # Check cross-references
            if metadata.get("rulemaking_authority"):
                stats["with_rulemaking_authority"] += 1
            if metadata.get("law_implemented"):
                stats["with_law_implemented"] += 1

            # Check text
            stats["text_lengths"].append(len(text))
            if len(text) < 50:
                stats["empty_text"] += 1
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "rule",
                        "issue": "empty_or_short_text",
                        "length": len(text),
                        "severity": "warning",
                    }
                )

            # Check encoding
            encoding_issues = check_encoding_issues(text)
            for issue in encoding_issues:
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "rule",
                        "issue": issue,
                        "severity": "warning",
                    }
                )

    # Compute statistics
    if stats["text_lengths"]:
        stats["avg_text_length"] = sum(stats["text_lengths"]) // len(stats["text_lengths"])
        stats["min_text_length"] = min(stats["text_lengths"])
        stats["max_text_length"] = max(stats["text_lengths"])
    del stats["text_lengths"]
    stats["missing_fields"] = dict(stats["missing_fields"])

    return {"issues": issues, "stats": stats, "unique_ids": len(ids_seen)}


def audit_taas(taa_dir: Path) -> dict:
    """Audit all TAA files."""
    issues = []
    stats = {
        "count": 0,
        "text_lengths": [],
        "missing_fields": Counter(),
        "empty_text": 0,
        "with_statutes_cited": 0,
        "with_rules_cited": 0,
        "avg_statutes_cited": 0,
        "avg_rules_cited": 0,
    }
    ids_seen = set()
    statutes_cited_counts = []
    rules_cited_counts = []

    required_fields = ["taa_number", "title"]

    for json_file in sorted(taa_dir.glob("TAA_*.json")):
        data = load_json_file(json_file)
        if data is None or "_error" in data:
            issues.append(
                {
                    "file": str(json_file.relative_to(DATA_DIR)),
                    "type": "taa",
                    "issue": "file_read_error",
                    "severity": "error",
                }
            )
            continue

        stats["count"] += 1
        metadata = data.get("metadata", {})
        text = data.get("text", "")

        # Generate ID
        taa_number = metadata.get("taa_number", "").replace(" ", "_")
        doc_id = f"taa:{taa_number}"

        # Check for duplicates
        if doc_id in ids_seen:
            issues.append(
                {
                    "file": str(json_file.relative_to(DATA_DIR)),
                    "type": "taa",
                    "issue": "duplicate_id",
                    "id": doc_id,
                    "severity": "warning",
                }
            )
        ids_seen.add(doc_id)

        # Check required fields
        for field in required_fields:
            if not metadata.get(field):
                stats["missing_fields"][field] += 1
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "taa",
                        "issue": "missing_field",
                        "field": field,
                        "severity": "warning",
                    }
                )

        # Check cross-references
        statutes = metadata.get("statutes_cited", [])
        rules = metadata.get("rules_cited", [])
        if statutes:
            stats["with_statutes_cited"] += 1
            statutes_cited_counts.append(len(statutes))
        if rules:
            stats["with_rules_cited"] += 1
            rules_cited_counts.append(len(rules))

        # Check text
        stats["text_lengths"].append(len(text))
        if len(text) < 50:
            stats["empty_text"] += 1
            issues.append(
                {
                    "file": str(json_file.relative_to(DATA_DIR)),
                    "type": "taa",
                    "issue": "empty_or_short_text",
                    "length": len(text),
                    "severity": "warning",
                }
            )

    # Compute statistics
    if stats["text_lengths"]:
        stats["avg_text_length"] = sum(stats["text_lengths"]) // len(stats["text_lengths"])
    del stats["text_lengths"]
    stats["missing_fields"] = dict(stats["missing_fields"])
    if statutes_cited_counts:
        stats["avg_statutes_cited"] = sum(statutes_cited_counts) // len(statutes_cited_counts)
    if rules_cited_counts:
        stats["avg_rules_cited"] = sum(rules_cited_counts) // len(rules_cited_counts)

    return {"issues": issues, "stats": stats, "unique_ids": len(ids_seen)}


def audit_cases(case_dir: Path) -> dict:
    """Audit all case law files."""
    issues = []
    stats = {
        "count": 0,
        "opinion_lengths": [],
        "missing_fields": Counter(),
        "empty_opinion": 0,
        "truncated_opinion": 0,
        "with_citations": 0,
        "with_statutes_cited": 0,
        "with_cases_cited": 0,
        "total_cases_cited": 0,
        "missing_date_filed": 0,
        "null_pdf_url": 0,
        "null_opinion_html": 0,
    }
    ids_seen = set()

    required_fields = ["cluster_id", "case_name", "court", "court_id"]

    for json_file in sorted(case_dir.glob("case_*.json")):
        data = load_json_file(json_file)
        if data is None or "_error" in data:
            issues.append(
                {
                    "file": str(json_file.relative_to(DATA_DIR)),
                    "type": "case",
                    "issue": "file_read_error",
                    "severity": "error",
                }
            )
            continue

        stats["count"] += 1
        metadata = data.get("metadata", {})
        opinion_text = data.get("opinion_text", "")

        # Generate ID
        cluster_id = metadata.get("cluster_id", "")
        doc_id = f"case:{cluster_id}"

        # Check for duplicates
        if doc_id in ids_seen:
            issues.append(
                {
                    "file": str(json_file.relative_to(DATA_DIR)),
                    "type": "case",
                    "issue": "duplicate_id",
                    "id": doc_id,
                    "severity": "warning",
                }
            )
        ids_seen.add(doc_id)

        # Check required fields
        for field in required_fields:
            if not metadata.get(field):
                stats["missing_fields"][field] += 1
                issues.append(
                    {
                        "file": str(json_file.relative_to(DATA_DIR)),
                        "type": "case",
                        "issue": "missing_field",
                        "field": field,
                        "severity": "warning",
                    }
                )

        # Check citations
        citations = metadata.get("citations", [])
        if citations:
            stats["with_citations"] += 1

        # Check cross-references
        statutes_cited = metadata.get("statutes_cited", [])
        cases_cited = metadata.get("cases_cited", [])
        if statutes_cited:
            stats["with_statutes_cited"] += 1
        if cases_cited:
            stats["with_cases_cited"] += 1
            stats["total_cases_cited"] += len(cases_cited)

        # Check date
        if metadata.get("date_filed") is None:
            stats["missing_date_filed"] += 1

        # Check optional URLs
        if data.get("pdf_url") is None:
            stats["null_pdf_url"] += 1
        if data.get("opinion_html") is None:
            stats["null_opinion_html"] += 1

        # Check opinion text
        opinion_len = len(opinion_text)
        stats["opinion_lengths"].append(opinion_len)

        if opinion_len < 50:
            stats["empty_opinion"] += 1
            issues.append(
                {
                    "file": str(json_file.relative_to(DATA_DIR)),
                    "type": "case",
                    "issue": "empty_or_short_opinion",
                    "length": opinion_len,
                    "severity": "warning",
                }
            )
        elif opinion_len < 500:
            # Likely truncated (full opinions are typically much longer)
            stats["truncated_opinion"] += 1

    # Compute statistics
    if stats["opinion_lengths"]:
        stats["avg_opinion_length"] = sum(stats["opinion_lengths"]) // len(stats["opinion_lengths"])
        stats["min_opinion_length"] = min(stats["opinion_lengths"])
        stats["max_opinion_length"] = max(stats["opinion_lengths"])
    del stats["opinion_lengths"]
    stats["missing_fields"] = dict(stats["missing_fields"])

    return {"issues": issues, "stats": stats, "unique_ids": len(ids_seen)}


def run_audit() -> dict:
    """Run the complete data quality audit."""
    print("Starting data quality audit...")
    print(f"Data directory: {DATA_DIR}")
    print()

    all_issues = []
    all_stats = {}

    # Audit statutes
    print("Auditing statutes...")
    statutes_dir = DATA_DIR / "statutes"
    if statutes_dir.exists():
        result = audit_statutes(statutes_dir)
        all_issues.extend(result["issues"])
        all_stats["statutes"] = result["stats"]
        print(f"  Found {result['stats']['count']} statutes, {result['unique_ids']} unique IDs")
        print(f"  Issues: {len(result['issues'])}")
    else:
        print("  Directory not found")

    # Audit rules
    print("Auditing administrative rules...")
    rules_dir = DATA_DIR / "admin_code"
    if rules_dir.exists():
        result = audit_rules(rules_dir)
        all_issues.extend(result["issues"])
        all_stats["rules"] = result["stats"]
        print(f"  Found {result['stats']['count']} rules, {result['unique_ids']} unique IDs")
        print(f"  Issues: {len(result['issues'])}")
    else:
        print("  Directory not found")

    # Audit TAAs
    print("Auditing TAAs...")
    taa_dir = DATA_DIR / "taa"
    if taa_dir.exists():
        result = audit_taas(taa_dir)
        all_issues.extend(result["issues"])
        all_stats["taas"] = result["stats"]
        print(f"  Found {result['stats']['count']} TAAs, {result['unique_ids']} unique IDs")
        print(f"  Issues: {len(result['issues'])}")
    else:
        print("  Directory not found")

    # Audit cases
    print("Auditing case law...")
    case_dir = DATA_DIR / "case_law"
    if case_dir.exists():
        result = audit_cases(case_dir)
        all_issues.extend(result["issues"])
        all_stats["cases"] = result["stats"]
        print(f"  Found {result['stats']['count']} cases, {result['unique_ids']} unique IDs")
        print(f"  Issues: {len(result['issues'])}")
    else:
        print("  Directory not found")

    # Build summary
    total_docs = sum(
        all_stats.get(key, {}).get("count", 0) for key in ["statutes", "rules", "taas", "cases"]
    )

    by_type = {
        "statute": all_stats.get("statutes", {}).get("count", 0),
        "rule": all_stats.get("rules", {}).get("count", 0),
        "taa": all_stats.get("taas", {}).get("count", 0),
        "case": all_stats.get("cases", {}).get("count", 0),
    }

    # Count issues by type and severity
    issue_counts = Counter()
    for issue in all_issues:
        issue_counts[issue["issue"]] += 1

    report = {
        "summary": {
            "total_documents": total_docs,
            "by_type": by_type,
            "issues_found": len(all_issues),
            "issue_types": dict(issue_counts),
            "audit_timestamp": datetime.now().isoformat(),
        },
        "issues": all_issues,
        "statistics": all_stats,
    }

    return report


def main():
    """Main entry point."""
    report = run_audit()

    # Print summary
    print()
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total documents: {report['summary']['total_documents']}")
    print(f"  - Statutes: {report['summary']['by_type']['statute']}")
    print(f"  - Rules: {report['summary']['by_type']['rule']}")
    print(f"  - TAAs: {report['summary']['by_type']['taa']}")
    print(f"  - Cases: {report['summary']['by_type']['case']}")
    print()
    print(f"Total issues found: {report['summary']['issues_found']}")
    if report["summary"]["issue_types"]:
        print("Issue breakdown:")
        for issue_type, count in sorted(
            report["summary"]["issue_types"].items(), key=lambda x: -x[1]
        ):
            print(f"  - {issue_type}: {count}")
    print()

    # Print key statistics
    print("KEY STATISTICS")
    print("-" * 40)
    if "statutes" in report["statistics"]:
        s = report["statistics"]["statutes"]
        print("Statutes:")
        print(f"  Avg text length: {s.get('avg_text_length', 'N/A')} chars")
        print(f"  Missing effective_date: {s.get('missing_effective_date', 0)}")
        print(f"  Empty/short text: {s.get('empty_text', 0)}")

    if "rules" in report["statistics"]:
        r = report["statistics"]["rules"]
        print("Rules:")
        print(f"  Avg text length: {r.get('avg_text_length', 'N/A')} chars")
        print(f"  With rulemaking_authority: {r.get('with_rulemaking_authority', 0)}")
        print(f"  With law_implemented: {r.get('with_law_implemented', 0)}")

    if "taas" in report["statistics"]:
        t = report["statistics"]["taas"]
        print("TAAs:")
        print(f"  Avg text length: {t.get('avg_text_length', 'N/A')} chars")
        print(f"  With statutes cited: {t.get('with_statutes_cited', 0)}")
        print(f"  With rules cited: {t.get('with_rules_cited', 0)}")

    if "cases" in report["statistics"]:
        c = report["statistics"]["cases"]
        print("Cases:")
        print(f"  Avg opinion length: {c.get('avg_opinion_length', 'N/A')} chars")
        print(f"  With statutes_cited: {c.get('with_statutes_cited', 0)}")
        print(f"  Empty statutes_cited: {c.get('count', 0) - c.get('with_statutes_cited', 0)}")
        print(f"  Truncated opinions (<500 chars): {c.get('truncated_opinion', 0)}")
        print(f"  Total case citations: {c.get('total_cases_cited', 0)}")

    # Save report
    output_path = DATA_DIR / "audit_report.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
