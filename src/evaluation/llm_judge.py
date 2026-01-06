"""LLM-based answer evaluation using GPT-4."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from openai import AsyncOpenAI

from .models import EvalQuestion, JudgmentResult
from config.prompts import JUDGE_PROMPT


class LLMJudge:
    """Uses GPT-4 to evaluate answer quality."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
    ):
        """Initialize the LLM judge.

        Args:
            api_key: OpenAI API key
            model: Model to use for evaluation (default: gpt-4-turbo-preview)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def judge_answer(
        self,
        question: EvalQuestion,
        generated_answer: str,
    ) -> JudgmentResult:
        """Evaluate answer quality using GPT-4.

        Args:
            question: The evaluation question with expected answers
            generated_answer: The RAG system's generated answer

        Returns:
            JudgmentResult with scores and feedback
        """
        prompt = JUDGE_PROMPT.format(
            question=question.question,
            expected_type=question.expected_answer_type.value,
            expected_contains=", ".join(question.expected_answer_contains) or "N/A",
            expected_statutes=", ".join(question.expected_statutes) or "N/A",
            expected_rules=", ".join(question.expected_rules) or "N/A",
            generated_answer=generated_answer,
            notes=question.notes or "N/A",
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Florida tax law evaluator. "
                    "Evaluate answers strictly and objectively.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result_json = json.loads(response.choices[0].message.content)

        return JudgmentResult(
            correctness=result_json.get("correctness", 0),
            completeness=result_json.get("completeness", 0),
            clarity=result_json.get("clarity", 0),
            citation_accuracy=result_json.get("citation_accuracy", 0),
            hallucinations=result_json.get("hallucinations", []),
            missing_concepts=result_json.get("missing_concepts", []),
            overall_score=result_json.get("overall_score", 0),
            reasoning=result_json.get("reasoning", ""),
        )

    async def judge_batch(
        self,
        questions: list[EvalQuestion],
        answers: list[str],
        max_concurrent: int = 5,
    ) -> list[JudgmentResult]:
        """Evaluate multiple answers with concurrency control.

        Args:
            questions: List of evaluation questions
            answers: List of generated answers (same order as questions)
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of JudgmentResult in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def judge_with_limit(q: EvalQuestion, a: str) -> JudgmentResult:
            async with semaphore:
                return await self.judge_answer(q, a)

        tasks = [judge_with_limit(q, a) for q, a in zip(questions, answers)]
        return await asyncio.gather(*tasks)
