"""Unit tests for scraper utility functions."""

from datetime import date

from src.scrapers.utils import (
    clean_html_text,
    extract_chapter_number,
    extract_dates,
    extract_section_number,
    normalize_whitespace,
    parse_rule_citation,
    parse_statute_citation,
)


class TestParseStatuteCitation:
    """Tests for parse_statute_citation function."""

    def test_basic_section_symbol(self):
        """Test parsing with § symbol."""
        text = "See § 212.05 for details."
        assert parse_statute_citation(text) == ["212.05"]

    def test_section_symbol_no_space(self):
        """Test parsing with § symbol and no space."""
        text = "See §212.05 for details."
        assert parse_statute_citation(text) == ["212.05"]

    def test_section_word(self):
        """Test parsing with 'section' word."""
        text = "See Section 212.05 for details."
        assert parse_statute_citation(text) == ["212.05"]

    def test_section_abbreviation(self):
        """Test parsing with 's.' abbreviation."""
        text = "See s. 212.05 for details."
        assert parse_statute_citation(text) == ["212.05"]

    def test_fla_stat_prefix(self):
        """Test parsing with 'Fla. Stat.' prefix."""
        text = "Pursuant to Fla. Stat. § 212.05."
        assert parse_statute_citation(text) == ["212.05"]

    def test_fs_suffix(self):
        """Test parsing with 'F.S.' suffix."""
        text = "See 212.05, F.S. for details."
        assert parse_statute_citation(text) == ["212.05"]

    def test_with_subsection(self):
        """Test parsing with subsection."""
        text = "See § 212.05(1)(a) for details."
        assert parse_statute_citation(text) == ["212.05(1)(a)"]

    def test_with_complex_subsection(self):
        """Test parsing with complex subsection."""
        text = "See § 212.08(7)(a)2. for details."
        assert parse_statute_citation(text) == ["212.08(7)(a)2."]

    def test_multiple_citations(self):
        """Test parsing multiple citations."""
        text = "See § 212.05 and § 212.08(1). Also see 196.012, F.S."
        result = parse_statute_citation(text)
        assert "212.05" in result
        assert "212.08(1)" in result
        assert "196.012" in result

    def test_no_citations(self):
        """Test text with no citations."""
        text = "This is just regular text without any citations."
        assert parse_statute_citation(text) == []

    def test_deduplication(self):
        """Test that duplicate citations are deduplicated."""
        text = "See § 212.05 and also see § 212.05 again."
        assert parse_statute_citation(text) == ["212.05"]

    def test_three_digit_chapter(self):
        """Test parsing with three-digit chapter."""
        text = "See § 196.012 for exemptions."
        assert parse_statute_citation(text) == ["196.012"]

    def test_four_digit_section(self):
        """Test parsing with four-digit section part."""
        text = "See § 212.0606 for details."
        assert parse_statute_citation(text) == ["212.0606"]


class TestParseRuleCitation:
    """Tests for parse_rule_citation function."""

    def test_basic_rule(self):
        """Test parsing basic rule citation."""
        text = "See Rule 12A-1.005 for details."
        assert parse_rule_citation(text) == ["12A-1.005"]

    def test_rule_abbreviation(self):
        """Test parsing with 'R.' abbreviation."""
        text = "See R. 12A-1.005 for details."
        assert parse_rule_citation(text) == ["12A-1.005"]

    def test_fac_suffix(self):
        """Test parsing with 'F.A.C.' suffix."""
        text = "See 12A-1.005, F.A.C. for details."
        assert parse_rule_citation(text) == ["12A-1.005"]

    def test_fla_admin_code_prefix(self):
        """Test parsing with 'Fla. Admin. Code' prefix."""
        text = "Pursuant to Fla. Admin. Code R. 12A-1.005."
        assert parse_rule_citation(text) == ["12A-1.005"]

    def test_with_subsection(self):
        """Test parsing with subsection."""
        text = "See Rule 12A-1.005(1)(a) for details."
        assert parse_rule_citation(text) == ["12A-1.005(1)(a)"]

    def test_multiple_rules(self):
        """Test parsing multiple rule citations."""
        text = "See Rule 12A-1.005 and Rule 12A-1.006(2)."
        result = parse_rule_citation(text)
        assert "12A-1.005" in result
        assert "12A-1.006(2)" in result

    def test_lowercase_letter_normalized(self):
        """Test that lowercase letter is normalized to uppercase."""
        text = "See Rule 12a-1.005 for details."
        assert parse_rule_citation(text) == ["12A-1.005"]

    def test_no_citations(self):
        """Test text with no rule citations."""
        text = "This is just regular text."
        assert parse_rule_citation(text) == []

    def test_rule_without_letter(self):
        """Test parsing rule without letter suffix."""
        text = "See Rule 12-1.005 for details."
        assert parse_rule_citation(text) == ["12-1.005"]


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_collapse_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        text = "Hello    world"
        assert normalize_whitespace(text) == "Hello world"

    def test_collapse_tabs(self):
        """Test collapsing tabs."""
        text = "Hello\t\tworld"
        assert normalize_whitespace(text) == "Hello world"

    def test_normalize_line_endings(self):
        """Test normalizing different line endings."""
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result = normalize_whitespace(text)
        assert "\r" not in result
        assert "Line 1\nLine 2\nLine 3\nLine 4" == result

    def test_strip_line_whitespace(self):
        """Test stripping whitespace from lines."""
        text = "  Hello  \n  World  "
        assert normalize_whitespace(text) == "Hello\nWorld"

    def test_limit_blank_lines(self):
        """Test limiting consecutive blank lines."""
        text = "Paragraph 1\n\n\n\n\nParagraph 2"
        result = normalize_whitespace(text)
        # Should have at most 2 blank lines (3 newlines) between paragraphs
        # 4+ consecutive newlines should not exist
        assert "\n\n\n\n" not in result
        assert "Paragraph 1" in result
        assert "Paragraph 2" in result

    def test_strip_leading_trailing(self):
        """Test stripping leading and trailing whitespace."""
        text = "\n\n  Hello  \n\n"
        assert normalize_whitespace(text) == "Hello"


class TestExtractDates:
    """Tests for extract_dates function."""

    def test_month_day_year(self):
        """Test extracting 'Month DD, YYYY' format."""
        text = "Effective January 1, 2024"
        result = extract_dates(text)
        assert date(2024, 1, 1) in result

    def test_month_day_year_no_comma(self):
        """Test extracting 'Month DD YYYY' format (no comma)."""
        text = "Effective January 1 2024"
        result = extract_dates(text)
        assert date(2024, 1, 1) in result

    def test_numeric_date_slash(self):
        """Test extracting 'MM/DD/YYYY' format."""
        text = "Date: 01/15/2024"
        result = extract_dates(text)
        assert date(2024, 1, 15) in result

    def test_numeric_date_dash(self):
        """Test extracting 'MM-DD-YYYY' format."""
        text = "Date: 01-15-2024"
        result = extract_dates(text)
        assert date(2024, 1, 15) in result

    def test_iso_date(self):
        """Test extracting 'YYYY-MM-DD' format."""
        text = "Date: 2024-01-15"
        result = extract_dates(text)
        assert date(2024, 1, 15) in result

    def test_multiple_dates(self):
        """Test extracting multiple dates."""
        text = "From January 1, 2024 to December 31, 2024"
        result = extract_dates(text)
        assert date(2024, 1, 1) in result
        assert date(2024, 12, 31) in result

    def test_no_dates(self):
        """Test text with no dates."""
        text = "This text has no dates."
        assert extract_dates(text) == []

    def test_sorted_output(self):
        """Test that dates are sorted chronologically."""
        text = "December 31, 2024 and January 1, 2024"
        result = extract_dates(text)
        assert result == [date(2024, 1, 1), date(2024, 12, 31)]


class TestExtractSectionNumber:
    """Tests for extract_section_number function."""

    def test_basic_extraction(self):
        """Test basic section number extraction."""
        text = "212.05 Sales tax on services"
        assert extract_section_number(text) == "212.05"

    def test_with_prefix_text(self):
        """Test extraction with prefix text."""
        text = "Section 212.05 - Sales tax"
        assert extract_section_number(text) == "212.05"

    def test_no_section_number(self):
        """Test text without section number."""
        text = "General provisions"
        assert extract_section_number(text) is None


class TestExtractChapterNumber:
    """Tests for extract_chapter_number function."""

    def test_chapter_keyword(self):
        """Test extraction with 'Chapter' keyword."""
        text = "Chapter 212 - Tax on Sales"
        assert extract_chapter_number(text) == 212

    def test_chapter_keyword_lowercase(self):
        """Test extraction with lowercase 'chapter'."""
        text = "chapter 212"
        assert extract_chapter_number(text) == 212

    def test_no_chapter(self):
        """Test text without chapter number."""
        text = "General provisions"
        assert extract_chapter_number(text) is None


class TestCleanHtmlText:
    """Tests for clean_html_text function."""

    def test_html_entities(self):
        """Test replacing HTML entities."""
        text = "Section&nbsp;212.05 &amp; 212.06"
        result = clean_html_text(text)
        assert "Section 212.05 & 212.06" == result

    def test_section_entity(self):
        """Test replacing &sect; entity."""
        text = "&sect; 212.05"
        assert clean_html_text(text) == "§ 212.05"

    def test_remaining_tags(self):
        """Test removing remaining HTML tags."""
        text = "Hello <b>world</b>"
        assert clean_html_text(text) == "Hello world"

    def test_combined_cleanup(self):
        """Test combined HTML cleanup."""
        text = "<p>Section&nbsp;212.05</p>\n\n\n\n<p>Next paragraph</p>"
        result = clean_html_text(text)
        assert "Section 212.05" in result
        assert "Next paragraph" in result
        assert "<p>" not in result
