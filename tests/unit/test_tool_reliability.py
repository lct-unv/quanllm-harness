from __future__ import annotations

import pytest

from quanllm_harness.tools import Tool, ToolRegistry, default_tool_registry
from quanllm_harness.tools.numeric_backend import numeric_integrate_fallback

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
    "additionalProperties": False,
}


class FlakyHandler:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def __call__(self, arguments):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"第 {self.calls} 次失败")
        return {"value": arguments["value"] * 2}


def registry_with(tool: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def tool(**overrides) -> Tool:
    values = {
        "name": "demo.double",
        "description": "double",
        "parameters": SCHEMA,
        "handler": FlakyHandler(0),
    }
    values.update(overrides)
    return Tool(**values)


def test_schema_prevalidation_rejects_before_any_handler_attempt():
    handler = FlakyHandler(0)
    registry = registry_with(tool(handler=handler))
    evidence = registry.execute("demo.double", {"wrong": 1})
    assert evidence.ok is False
    assert evidence.error_kind == "schema"
    assert evidence.attempts == 0
    assert handler.calls == 0
    assert "Schema 预校验失败" in evidence.error


def test_unknown_tool_is_reported_with_its_own_error_kind():
    evidence = ToolRegistry().execute("missing.tool", {})
    assert evidence.ok is False
    assert evidence.error_kind == "unknown"
    assert evidence.attempts == 0


def test_transient_handler_failure_is_retried_and_succeeds():
    handler = FlakyHandler(2)
    registry = registry_with(tool(handler=handler, retry_attempts=3))
    events = []
    evidence = registry.execute("demo.double", {"value": 4}, event_sink=events.append)
    assert evidence.ok is True
    assert evidence.result == {"value": 8}
    assert evidence.attempts == 3
    assert evidence.error_kind == ""
    assert handler.calls == 3
    retries = [event for event in events if event.kind == "tool_retried"]
    assert len(retries) == 2
    assert retries[0].payload["next_attempt"] == 2


def test_exhausted_retries_report_the_attempt_count():
    handler = FlakyHandler(10)
    registry = registry_with(tool(handler=handler, retry_attempts=3))
    evidence = registry.execute("demo.double", {"value": 4})
    assert evidence.ok is False
    assert evidence.error_kind == "handler"
    assert evidence.attempts == 3
    assert handler.calls == 3


def test_default_policy_runs_the_handler_exactly_once():
    handler = FlakyHandler(10)
    registry = registry_with(tool(handler=handler))
    evidence = registry.execute("demo.double", {"value": 4})
    assert evidence.attempts == 1
    assert handler.calls == 1


def test_fallback_result_is_recorded_as_such():
    def broken(arguments):
        raise RuntimeError("原生后端不可用")

    registry = registry_with(
        tool(
            handler=broken,
            retry_attempts=2,
            fallback=lambda arguments: {"value": arguments["value"] * 2},
        )
    )
    events = []
    evidence = registry.execute("demo.double", {"value": 5}, event_sink=events.append)
    assert evidence.ok is True
    assert evidence.result == {"value": 10}
    assert evidence.fallback_used is True
    assert evidence.attempts == 2
    assert "原生后端不可用" in evidence.error
    assert any("回退实现" in item for item in evidence.limitations)
    assert any(event.kind == "tool_fallback_used" for event in events)


def test_fallback_failure_keeps_the_original_error():
    def broken(arguments):
        raise RuntimeError("原生后端不可用")

    def also_broken(arguments):
        raise ValueError("回退后端同样不可用")

    registry = registry_with(tool(handler=broken, fallback=also_broken))
    evidence = registry.execute("demo.double", {"value": 5})
    assert evidence.ok is False
    assert evidence.error_kind == "handler"
    assert "原生后端不可用" in evidence.error
    assert "回退实现同样失败" in evidence.error
    assert any("回退实现失败" in item for item in evidence.limitations)


def test_non_json_result_is_rejected_as_evidence_failure():
    registry = registry_with(tool(handler=lambda arguments: {"values": {1, 2}}))
    evidence = registry.execute("demo.double", {"value": 1})
    assert evidence.ok is False
    assert evidence.error_kind == "evidence"
    assert "不可序列化" in evidence.error


def test_result_with_non_string_keys_is_rejected():
    registry = registry_with(tool(handler=lambda arguments: {1: "a"}))
    evidence = registry.execute("demo.double", {"value": 1})
    assert evidence.ok is False
    assert evidence.error_kind == "evidence"


def test_duplicate_calls_reuse_cached_evidence():
    handler = FlakyHandler(0)
    registry = registry_with(tool(handler=handler))
    events = []
    first = registry.execute("demo.double", {"value": 3})
    second = registry.execute("demo.double", {"value": 3}, event_sink=events.append)
    assert first.id == second.id
    assert handler.calls == 1
    assert any(event.kind == "tool_reused" for event in events)


def test_invalid_retry_policy_is_rejected_at_registration():
    with pytest.raises(ValueError):
        registry_with(tool(retry_attempts=0))
    with pytest.raises(TypeError):
        registry_with(tool(fallback="not callable"))


def test_builtin_numeric_integrate_declares_a_fallback():
    registry = default_tool_registry(include_plugins=False)
    capabilities = registry.capabilities()
    assert capabilities["numeric_integrate"]["has_fallback"] is True
    assert capabilities["numeric_integrate"]["retry_attempts"] == 1


def test_numeric_integrate_fallback_uses_bounded_alternate_strategies():
    finite = numeric_integrate_fallback(
        {"expression": "x**2", "variable": "x", "lower": "0", "upper": "1", "dps": 30}
    )
    assert finite["method"] == "gauss-legendre-subdivided"
    assert abs(float(finite["value"]) - 1 / 3) < 1e-12

    infinite = numeric_integrate_fallback(
        {
            "expression": "exp(-x**2)",
            "variable": "x",
            "lower": "-inf",
            "upper": "inf",
            "dps": 30,
        }
    )
    assert infinite["method"] == "tanh-sinh-higher-precision"
    assert abs(float(infinite["value"]) - 1.7724538509055159) < 1e-9
