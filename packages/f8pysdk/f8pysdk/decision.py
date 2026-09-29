"""Typed decision primitives, independent of conversational agent protocols."""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

import msgspec

from .specs import F8JsonValue


Description: TypeAlias = str | dict[str, F8JsonValue] | list[F8JsonValue]
Probability: TypeAlias = Annotated[float, msgspec.Meta(ge=0, le=1)]


class ChoiceQuestion(msgspec.Struct, frozen=True, kw_only=True, tag="choice", tag_field="type", forbid_unknown_fields=True):
    instructions: Description
    criteria: dict[str, Description | None]


class ScoreQuestion(msgspec.Struct, frozen=True, kw_only=True, tag="score", tag_field="type", forbid_unknown_fields=True):
    instructions: Description
    criteria: tuple[Description, ...]


class NoulQuestion(msgspec.Struct, frozen=True, kw_only=True, tag="noul", tag_field="type", forbid_unknown_fields=True):
    instructions: Description
    criteria: dict[Literal["true", "false"], Description] = msgspec.field(default_factory=dict)


Question: TypeAlias = ChoiceQuestion | ScoreQuestion | NoulQuestion


class ChoiceAnswer(msgspec.Struct, frozen=True, kw_only=True, tag="choice", tag_field="type"):
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(msgspec.Struct, frozen=True, kw_only=True, tag="score", tag_field="type"):
    score: float
    legend: dict[str, Description]
    probabilities: dict[str, Probability]
    confidence: Probability


class NoulAnswer(msgspec.Struct, frozen=True, kw_only=True, tag="noul", tag_field="type"):
    noul: Probability


Answer: TypeAlias = ChoiceAnswer | ScoreAnswer | NoulAnswer


class DecisionUsage(msgspec.Struct, frozen=True, kw_only=True):
    input_tokens: Annotated[int, msgspec.Meta(ge=0)]
    output_tokens: Annotated[int, msgspec.Meta(ge=0)]


class DecisionResult(msgspec.Struct, frozen=True, kw_only=True):
    model: str
    answers: dict[str, Answer]
    usage: DecisionUsage


class DecisionRequest(msgspec.Struct, frozen=True, kw_only=True, rename="camel", forbid_unknown_fields=True):
    state: Description
    questions: dict[str, Question]
    provider_id: str = "typesafe"
    image_data_url: str | None = None


def validate_questions(questions: dict[str, Question]) -> None:
    if not 1 <= len(questions) <= 256:
        raise ValueError("A decision request must contain 1 to 256 questions")
    for question_id, question in questions.items():
        if not question_id.strip() or not question.instructions:
            raise ValueError("Question IDs and instructions must be non-empty")
        if isinstance(question, ChoiceQuestion):
            if not 2 <= len(question.criteria) <= 255 or any(not key.strip() for key in question.criteria):
                raise ValueError("Choice requires 2 to 255 named options")
        elif isinstance(question, ScoreQuestion):
            if not 2 <= len(question.criteria) <= 10:
                raise ValueError("Score requires 2 to 10 ordered levels")


def validate_result(result: DecisionResult, questions: dict[str, Question]) -> None:
    if set(result.answers) != set(questions):
        raise ValueError("Decision response question IDs do not match the request")
    for question_id, question in questions.items():
        answer = result.answers[question_id]
        if isinstance(question, NoulQuestion) and isinstance(answer, NoulAnswer):
            if not math.isfinite(answer.noul) or not 0 <= answer.noul <= 1:
                raise ValueError("Noul probability is outside [0, 1]")
            continue
        if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
            expected = set(question.criteria)
            if answer.choice not in expected:
                raise ValueError("Choice response contains an unknown option")
        elif isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
            expected = {str(index) for index in range(len(question.criteria))}
            if set(answer.legend) != expected or not math.isfinite(answer.score) or not 0 <= answer.score <= len(question.criteria) - 1:
                raise ValueError("Score response contains invalid levels or score")
        else:
            raise ValueError("Decision answer type does not match its question")
        if set(answer.probabilities) != expected:
            raise ValueError("Decision probability keys do not match the question options")
        probabilities = answer.probabilities.values()
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Decision probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(probabilities), 1, abs_tol=0.001):
            raise ValueError("Decision probabilities must sum to one")
        if not math.isfinite(answer.confidence) or not 0 <= answer.confidence <= 1:
            raise ValueError("Decision confidence must be a finite value in [0, 1]")
