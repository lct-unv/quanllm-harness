from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from time import monotonic
from typing import Any

from ..config import HarnessSettings
from ..contracts import EventSink, HarnessEvent, ModelResponse, ToolCall, Usage
from ..protocols.json_request import StructuredResponseError, decode_argument_objects
from .base import ProviderError, QuanLLMProvider

_REASONING_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _extract_reasoning_tool_calls(text: str) -> list[ToolCall]:
    """Fallback for models that emit ``<tool_call>{...}</tool_call>`` inside the
    reasoning stream instead of the structured ``tool_calls`` API field."""
    calls: list[ToolCall] = []
    for index, match in enumerate(_REASONING_CALL_RE.finditer(text or "")):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = str(data.get("name") or "")
        arguments_raw = data.get("arguments")
        if not name or arguments_raw is None:
            continue
        if isinstance(arguments_raw, dict):
            argument_objects = [arguments_raw]
        else:
            try:
                argument_objects = decode_argument_objects(str(arguments_raw))
            except (json.JSONDecodeError, StructuredResponseError):
                continue
        for arguments in argument_objects:
            calls.append(ToolCall(f"reasoning-{index}", name, arguments))
    return calls


class OpenAIQuanLLMProvider(QuanLLMProvider):
    """OpenAI-compatible provider with QuanLLM-v2 request invariants."""

    def __init__(self, settings: HarnessSettings, client: Any | None = None):
        settings.validate()
        self.settings = settings
        if client is None:
            if not settings.api_key:
                raise ValueError("初始化真实 Provider 时必须在 APIKEY 文件中提供密钥")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("请安装 openai 依赖") from exc
            client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self.client = client

    @staticmethod
    def _emit(sink: EventSink | None, kind: str, stage: str, **payload: Any) -> None:
        if sink:
            sink(HarnessEvent(kind=kind, stage=stage, payload=payload))

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        structured: bool = False,
        tools: Sequence[Mapping[str, Any]] = (),
        event_sink: EventSink | None = None,
    ) -> ModelResponse:
        if structured and tools:
            raise ValueError("结构化 JSON 请求不能同时执行工具循环")
        profile = self.settings.structured if structured else self.settings.reasoning
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": {"enable_thinking": not structured},
        }
        if structured:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"

        self._emit(event_sink, "agent_started", stage, structured=structured)
        try:
            stream = self.client.with_options(
                timeout=profile.timeout_seconds,
                max_retries=0,
            ).chat.completions.create(**kwargs)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            tool_slots: dict[int, dict[str, str]] = {}
            usage = Usage()
            finish_reason = "unknown"
            started_at = monotonic()
            for chunk in stream:
                if monotonic() - started_at > profile.timeout_seconds:
                    raise TimeoutError(f"{stage} 超过总时长上限 {profile.timeout_seconds:g} 秒")
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    usage = Usage(
                        prompt_tokens=getattr(chunk_usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(chunk_usage, "completion_tokens", 0) or 0,
                    )
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = choice.delta
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    reasoning_parts.append(reasoning)
                    self._emit(event_sink, "reasoning_delta", stage, text=reasoning)
                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                    self._emit(event_sink, "content_delta", stage, text=content)
                for tool_call in getattr(delta, "tool_calls", None) or ():
                    slot = tool_slots.setdefault(
                        tool_call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tool_call.id:
                        slot["id"] = tool_call.id
                    function = getattr(tool_call, "function", None)
                    if function:
                        slot["name"] += getattr(function, "name", None) or ""
                        slot["arguments"] += getattr(function, "arguments", None) or ""
        except Exception as exc:
            self._emit(event_sink, "provider_error", stage, error=type(exc).__name__)
            raise ProviderError(f"{stage} 请求失败：{exc}") from exc

        calls: list[ToolCall] = []
        if not tool_slots:
            calls.extend(_extract_reasoning_tool_calls("".join(reasoning_parts)))
        for slot in tool_slots.values():
            try:
                argument_objects = decode_argument_objects(slot["arguments"])
            except (json.JSONDecodeError, StructuredResponseError) as exc:
                raise ProviderError(f"{stage} 的工具参数不是合法 JSON") from exc
            for index, arguments in enumerate(argument_objects):
                call_id = slot["id"] if index == 0 else f"{slot['id']}-{index + 1}"
                calls.append(ToolCall(call_id, slot["name"], arguments))
        response = ModelResponse(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            finish_reason=finish_reason,
            tool_calls=tuple(calls),
            usage=usage,
        )
        self._emit(
            event_sink,
            "agent_finished",
            stage,
            finish_reason=finish_reason,
            content_chars=len(response.content),
            reasoning_chars=len(response.reasoning),
        )
        return response
