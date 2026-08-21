from __future__ import annotations

import json
from typing import Any


class StructuredResponseError(RuntimeError):
    """The model returned JSON that did not satisfy a fixed protocol."""


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        candidate = "\n".join(lines).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise StructuredResponseError("响应中没有完整 JSON 对象")
    try:
        value = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise StructuredResponseError(f"JSON 解析失败：{exc.msg}") from exc
    if not isinstance(value, dict):
        raise StructuredResponseError("结构化响应必须是 JSON 对象")
    return value


def decode_argument_objects(text: str) -> list[dict[str, Any]]:
    """Decode one or more concatenated tool-argument objects from legacy gateways."""

    decoder = json.JSONDecoder()
    source = (text or "{}").strip()
    values: list[dict[str, Any]] = []
    position = 0
    while position < len(source):
        while position < len(source) and (source[position].isspace() or source[position] == ","):
            position += 1
        if position >= len(source):
            break
        value, position = decoder.raw_decode(source, position)
        if not isinstance(value, dict):
            raise StructuredResponseError("工具参数必须是 JSON 对象")
        values.append(value)
    return values or [{}]
