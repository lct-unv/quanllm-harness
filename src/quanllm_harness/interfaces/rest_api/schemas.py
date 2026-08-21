from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=100_000)


class AnswerResponse(BaseModel):
    request_id: str
    result: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model: str
    configured: bool
    version: str
