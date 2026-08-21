from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .common import Usage


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    reasoning: str = ""
    finish_reason: str = "unknown"
    tool_calls: Sequence[ToolCall] = ()
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class RequestPolicy:
    depth: str = "standard"
    suspicious_input: bool = False
    requires_tools: bool = False
    requires_independent_solver: bool = False
    language: str = "zh"
    reason: str = ""
    tool_domains: Sequence[str] = ()


@dataclass(frozen=True)
class Candidate:
    source: str
    answer: str
    reasoning: str = ""
