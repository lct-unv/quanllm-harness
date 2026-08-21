from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunStatus(str, Enum):
    VERIFIED = "verified"
    VERIFIED_WITH_INPUT_AMBIGUITY = "verified_with_input_ambiguity"
    DEGRADED_DELIVERY = "degraded_delivery"
    FAILED_WITHOUT_ANSWER = "failed_without_answer"


class IssueOrigin(str, Enum):
    MODEL = "model"
    INPUT = "input"
    INFRASTRUCTURE = "infrastructure"


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )
