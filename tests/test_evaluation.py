"""Tests for the evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluation import (
    AnswerType,
    Category,
    Difficulty,
    EvalDataset,
    EvalQuestion,
    EvalReport,
    EvalResult,
    JudgmentResult,
    LLMJudge,
    answer_contains_expected,
    citation_precision,
    citation_recall,
    extract_citations_from_answer,
    f1_score,
    normalize_citation,
)

# =============================================================================
# Model Tests
# =============================================================================


class TestEvalQuestion:
    """Tests for EvalQuestion model."""

    def test_create_question(self):
        """Test creating an evaluation question."""
        question = EvalQuestion(
            id="eval_001",
            question="What is the Florida sales tax rate?",
            category=Category.SALES_TAX,
            difficulty=Difficulty.EASY,
            expected_statutes=["212.05"],
            expected_rules=[],
            expected_answer_contains=["6%"],
            expected_answer_type=AnswerType.NUMERIC,
        )

        assert question.id == "eval_001"
        assert question.category == Category.SALES_TAX
        assert question.difficulty == Difficulty.EASY
        assert "212.05" in question.expected_statutes

    def test_question_with_notes(self):
        """Test question with evaluator notes."""
        question = EvalQuestion(
            id="eval_002",
            question="Are groceries exempt?",
            category=Category.EXEMPTIONS,
            difficulty=Difficulty.EASY,
            expected_answer_type=AnswerType.YES_NO,
            notes="Food for human consumption is exempt.",
        )

        assert question.notes == "Food for human consumption is exempt."


class TestJudgmentResult:
    """Tests for JudgmentResult model."""

    def test_judgment_passed(self):
        """Test judgment passes with high score and no hallucinations."""
        judgment = JudgmentResult(
            correctness=8,
            completeness=7,
            clarity=8,
            citation_accuracy=7,
            hallucinations=[],
            missing_concepts=[],
            overall_score=8,
            reasoning="Good answer.",
        )

        assert judgment.passed is True

    def test_judgment_failed_low_score(self):
        """Test judgment fails with low overall score."""
        judgment = JudgmentResult(
            correctness=5,
            completeness=5,
            clarity=6,
            citation_accuracy=5,
            hallucinations=[],
            missing_concepts=["key concept"],
            overall_score=5,
            reasoning="Incomplete answer.",
        )

        assert judgment.passed is False

    def test_judgment_failed_hallucinations(self):
        """Test judgment fails when hallucinations present."""
        judgment = JudgmentResult(
            correctness=6,
            completeness=7,
            clarity=8,
            citation_accuracy=6,
            hallucinations=["Invented statute § 999.99"],
            missing_concepts=[],
            overall_score=7,
            reasoning="Contains fabricated citation.",
        )

        assert judgment.passed is False


class TestEvalResult:
    """Tests for EvalResult model."""

    def test_create_result(self):
        """Test creating an evaluation result."""
        result = EvalResult(
            question_id="eval_001",
            generated_answer="The Florida sales tax rate is 6%.",
            generated_citations=["212.05"],
            citation_precision=1.0,
            citation_recall=1.0,
            answer_contains_score=1.0,
            latency_ms=150,
        )

        assert result.question_id == "eval_001"
        assert result.citation_precision == 1.0
        assert result.latency_ms == 150

    def test_result_with_judgment(self):
        """Test result with LLM judgment attached."""
        judgment = JudgmentResult(
            correctness=8,
            completeness=8,
            clarity=8,
            citation_accuracy=8,
            overall_score=8,
            reasoning="Excellent.",
        )

        result = EvalResult(
            question_id="eval_001",
            generated_answer="Answer text",
            citation_precision=0.9,
            citation_recall=0.8,
            answer_contains_score=1.0,
            latency_ms=100,
            judgment=judgment,
        )

        assert result.judgment is not None
        assert result.judgment.overall_score == 8


class TestEvalReport:
    """Tests for EvalReport model."""

    def test_empty_report(self):
        """Test empty report returns zero metrics."""
        report = EvalReport(results=[])

        assert report.avg_citation_precision == 0.0
        assert report.avg_citation_recall == 0.0
        assert report.avg_overall_score == 0.0
        assert report.pass_rate == 0.0

    def test_report_aggregates_metrics(self):
        """Test report correctly aggregates metrics."""
        results = [
            EvalResult(
                question_id="eval_001",
                generated_answer="Answer 1",
                citation_precision=0.8,
                citation_recall=0.6,
                answer_contains_score=1.0,
                latency_ms=100,
                judgment=JudgmentResult(
                    correctness=8,
                    completeness=7,
                    clarity=8,
                    citation_accuracy=7,
                    overall_score=8,
                    reasoning="Good.",
                ),
            ),
            EvalResult(
                question_id="eval_002",
                generated_answer="Answer 2",
                citation_precision=1.0,
                citation_recall=1.0,
                answer_contains_score=0.5,
                latency_ms=200,
                judgment=JudgmentResult(
                    correctness=6,
                    completeness=5,
                    clarity=6,
                    citation_accuracy=6,
                    hallucinations=["fake fact"],
                    overall_score=6,
                    reasoning="Partial.",
                ),
            ),
        ]

        report = EvalReport(results=results)

        assert report.avg_citation_precision == 0.9
        assert report.avg_citation_recall == 0.8
        assert report.avg_overall_score == 7.0
        assert report.pass_rate == 0.5  # 1 of 2 passed


# =============================================================================
# Metrics Tests
# =============================================================================


class TestNormalizeCitation:
    """Tests for citation normalization."""

    def test_normalize_statute_with_prefix(self):
        """Test normalizing statute with Fla. Stat. prefix."""
        assert normalize_citation("Fla. Stat. § 212.05") == "212.05"
        assert normalize_citation("Fla Stat § 212.05") == "212.05"

    def test_normalize_statute_section_only(self):
        """Test normalizing bare section number."""
        assert normalize_citation("§ 212.05") == "212.05"
        assert normalize_citation("212.05") == "212.05"

    def test_normalize_statute_with_subsections(self):
        """Test normalizing statute with subsections."""
        assert normalize_citation("§ 212.08(1)(a)") == "212.08(1)(a)"

    def test_normalize_rule(self):
        """Test normalizing rule citations."""
        assert normalize_citation("Rule 12A-1.001") == "12a-1.001"
        assert normalize_citation("F.A.C. 12A-1.001") == "12a-1.001"

    def test_normalize_case_insensitive(self):
        """Test normalization is case insensitive."""
        assert normalize_citation("FLA. STAT. § 212.05") == "212.05"


class TestCitationPrecision:
    """Tests for citation precision calculation."""

    def test_perfect_precision(self):
        """Test precision is 1.0 when all generated citations are correct."""
        precision = citation_precision(
            generated=["212.05"],
            expected_statutes=["212.05", "212.08"],
            expected_rules=[],
        )
        assert precision == 1.0

    def test_zero_precision(self):
        """Test precision is 0.0 when no generated citations are correct."""
        precision = citation_precision(
            generated=["999.99"],
            expected_statutes=["212.05"],
            expected_rules=[],
        )
        assert precision == 0.0

    def test_partial_precision(self):
        """Test precision with some correct citations."""
        precision = citation_precision(
            generated=["212.05", "999.99"],
            expected_statutes=["212.05"],
            expected_rules=[],
        )
        assert precision == 0.5

    def test_empty_generated(self):
        """Test precision is 1.0 when no citations generated (no false positives)."""
        precision = citation_precision(
            generated=[],
            expected_statutes=["212.05"],
            expected_rules=[],
        )
        assert precision == 1.0


class TestCitationRecall:
    """Tests for citation recall calculation."""

    def test_perfect_recall(self):
        """Test recall is 1.0 when all expected citations are found."""
        recall = citation_recall(
            generated=["212.05", "212.08"],
            expected_statutes=["212.05", "212.08"],
            expected_rules=[],
        )
        assert recall == 1.0

    def test_zero_recall(self):
        """Test recall is 0.0 when no expected citations are found."""
        recall = citation_recall(
            generated=["999.99"],
            expected_statutes=["212.05"],
            expected_rules=[],
        )
        assert recall == 0.0

    def test_partial_recall(self):
        """Test recall with some expected citations found."""
        recall = citation_recall(
            generated=["212.05"],
            expected_statutes=["212.05", "212.08"],
            expected_rules=[],
        )
        assert recall == 0.5

    def test_empty_expected(self):
        """Test recall is 1.0 when no expected citations (nothing to miss)."""
        recall = citation_recall(
            generated=["212.05"],
            expected_statutes=[],
            expected_rules=[],
        )
        assert recall == 1.0


class TestAnswerContainsExpected:
    """Tests for answer contains score."""

    def test_all_phrases_found(self):
        """Test score is 1.0 when all expected phrases found."""
        score = answer_contains_expected(
            answer="The rate is 6 percent and applies to sales.",
            expected_phrases=["6 percent", "sales"],
        )
        assert score == 1.0

    def test_no_phrases_found(self):
        """Test score is 0.0 when no expected phrases found."""
        score = answer_contains_expected(
            answer="This is an unrelated answer.",
            expected_phrases=["6%", "tax rate"],
        )
        assert score == 0.0

    def test_partial_phrases_found(self):
        """Test partial score when some phrases found."""
        score = answer_contains_expected(
            answer="The rate is 6%.",
            expected_phrases=["6%", "sales tax", "Florida"],
        )
        assert abs(score - 1 / 3) < 0.01

    def test_empty_expected(self):
        """Test score is 1.0 when no expected phrases."""
        score = answer_contains_expected(
            answer="Any answer.",
            expected_phrases=[],
        )
        assert score == 1.0


class TestExtractCitations:
    """Tests for citation extraction from answer text."""

    def test_extract_statute_section_symbol(self):
        """Test extracting statute with § symbol."""
        citations = extract_citations_from_answer("See § 212.05 for details.")
        assert "212.05" in citations

    def test_extract_statute_fla_stat(self):
        """Test extracting statute with Fla. Stat. prefix."""
        citations = extract_citations_from_answer("Per Fla. Stat. § 212.05(1).")
        assert any("212.05" in c for c in citations)

    def test_extract_rule(self):
        """Test extracting rule citation."""
        citations = extract_citations_from_answer("Rule 12A-1.001 provides...")
        assert "12A-1.001" in citations

    def test_extract_multiple(self):
        """Test extracting multiple citations."""
        text = "See § 212.05 and Rule 12A-1.001 for requirements."
        citations = extract_citations_from_answer(text)
        assert len(citations) >= 2

    def test_deduplicate(self):
        """Test citations are deduplicated."""
        text = "See § 212.05. Also refer to § 212.05 again."
        citations = extract_citations_from_answer(text)
        assert citations.count("212.05") == 1


class TestF1Score:
    """Tests for F1 score calculation."""

    def test_perfect_f1(self):
        """Test F1 is 1.0 with perfect precision and recall."""
        assert f1_score(1.0, 1.0) == 1.0

    def test_zero_f1(self):
        """Test F1 is 0.0 when either metric is 0."""
        assert f1_score(0.0, 0.0) == 0.0
        assert f1_score(1.0, 0.0) == 0.0
        assert f1_score(0.0, 1.0) == 0.0

    def test_balanced_f1(self):
        """Test F1 with balanced precision and recall."""
        f1 = f1_score(0.8, 0.8)
        assert abs(f1 - 0.8) < 0.01


# =============================================================================
# Dataset Tests
# =============================================================================


class TestGoldenDataset:
    """Tests for the golden evaluation dataset."""

    @pytest.fixture
    def dataset_path(self):
        """Path to golden dataset."""
        return Path(__file__).parent.parent / "data" / "evaluation" / "golden_dataset.json"

    def test_dataset_exists(self, dataset_path):
        """Test dataset file exists."""
        assert dataset_path.exists(), f"Dataset not found at {dataset_path}"

    def test_dataset_valid_json(self, dataset_path):
        """Test dataset is valid JSON."""
        with open(dataset_path) as f:
            data = json.load(f)
        assert "questions" in data
        assert "metadata" in data

    def test_dataset_has_20_questions(self, dataset_path):
        """Test dataset contains 20 questions."""
        with open(dataset_path) as f:
            data = json.load(f)
        assert len(data["questions"]) == 20

    def test_dataset_difficulty_distribution(self, dataset_path):
        """Test dataset has correct difficulty distribution."""
        with open(dataset_path) as f:
            data = json.load(f)

        difficulties = [q["difficulty"] for q in data["questions"]]
        assert difficulties.count("easy") == 5
        assert difficulties.count("medium") == 10
        assert difficulties.count("hard") == 5

    def test_dataset_parses_to_model(self, dataset_path):
        """Test dataset parses to EvalDataset model."""
        with open(dataset_path) as f:
            data = json.load(f)

        dataset = EvalDataset(**data)
        assert len(dataset.questions) == 20
        assert all(isinstance(q, EvalQuestion) for q in dataset.questions)

    def test_all_questions_have_required_fields(self, dataset_path):
        """Test all questions have required fields."""
        with open(dataset_path) as f:
            data = json.load(f)

        for q in data["questions"]:
            assert "id" in q
            assert "question" in q
            assert "category" in q
            assert "difficulty" in q
            assert "expected_answer_type" in q


# =============================================================================
# LLM Judge Tests
# =============================================================================


class TestLLMJudge:
    """Tests for LLM judge (mocked)."""

    @pytest.fixture
    def mock_openai(self):
        """Create mock OpenAI client."""
        with patch("src.evaluation.llm_judge.AsyncOpenAI") as mock:
            mock_response = MagicMock()
            mock_response.choices = [
                MagicMock(
                    message=MagicMock(
                        content=json.dumps(
                            {
                                "correctness": 8,
                                "completeness": 7,
                                "clarity": 8,
                                "citation_accuracy": 7,
                                "hallucinations": [],
                                "missing_concepts": [],
                                "overall_score": 8,
                                "reasoning": "Good answer with correct citations.",
                            }
                        )
                    )
                )
            ]

            mock_client = MagicMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock.return_value = mock_client

            yield mock

    @pytest.mark.asyncio
    async def test_judge_answer(self, mock_openai):
        """Test judging a single answer."""
        judge = LLMJudge(api_key="test-key")

        question = EvalQuestion(
            id="eval_001",
            question="What is the Florida sales tax rate?",
            category=Category.SALES_TAX,
            difficulty=Difficulty.EASY,
            expected_answer_type=AnswerType.NUMERIC,
        )

        result = await judge.judge_answer(
            question=question,
            generated_answer="The Florida sales tax rate is 6%.",
        )

        assert isinstance(result, JudgmentResult)
        assert result.overall_score == 8
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_judge_uses_correct_model(self, mock_openai):
        """Test judge uses specified model."""
        judge = LLMJudge(api_key="test-key", model="gpt-4o")

        question = EvalQuestion(
            id="eval_001",
            question="Test question",
            category=Category.SALES_TAX,
            difficulty=Difficulty.EASY,
            expected_answer_type=AnswerType.EXPLANATION,
        )

        await judge.judge_answer(question=question, generated_answer="Test answer")

        # Verify model was passed to API
        call_args = judge.client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"
