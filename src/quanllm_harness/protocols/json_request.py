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
    # Locate the first *valid* JSON object with ``raw_decode`` instead of
    # slicing between the first "{" and the last "}". Slicing breaks when a
    # string value contains braces (e.g. "{a, b, c}") or when the model appends
    # explanatory text after the object. Try each "{" until one decodes.
    decoder = json.JSONDecoder()
    position = 0
    while True:
        position = candidate.find("{", position)
        if position < 0:
            raise StructuredResponseError("响应中没有完整 JSON 对象")
        try:
            value, _ = decoder.raw_decode(candidate, position)
        except json.JSONDecodeError:
            position += 1
            continue
        if isinstance(value, dict):
            return value
        position += 1


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
