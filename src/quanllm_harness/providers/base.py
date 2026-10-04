from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from ..contracts import EventSink, ModelResponse
from ..protocols.json_request import StructuredResponseError, parse_json_object


class ProviderError(RuntimeError):
    pass


class InvalidToolArgumentsError(ProviderError):
    """A malformed provider tool call that is safe to regenerate tool-free."""

    def __init__(self, stage: str, tool: str, raw_arguments: str, parse_error: str):
        super().__init__(f"{stage} 的工具 {tool or '<unknown>'} 参数不是合法 JSON")
        self.stage = stage
        self.tool = tool
        self.raw_arguments = raw_arguments
        self.parse_error = parse_error


class QuanLLMProvider(ABC):
    @abstractmethod
    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        structured: bool = False,
        tools: Sequence[Mapping[str, Any]] = (),
        event_sink: EventSink | None = None,
    ) -> ModelResponse:
        raise NotImplementedError

    def complete_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], ModelResponse]:
        response = self.complete(messages, stage=stage, structured=True, event_sink=event_sink)
        if not response.content.strip():
            raise StructuredResponseError(
                f"{stage} 没有 JSON 正文（finish_reason={response.finish_reason}）"
            )
        if response.finish_reason == "length":
            raise StructuredResponseError(f"{stage} 的 JSON 达到输出上限")
        return parse_json_object(response.content), response
