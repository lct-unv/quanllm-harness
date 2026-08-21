from __future__ import annotations

from types import SimpleNamespace

from quanllm_harness.tools.registry import Tool, ToolRegistry


def test_entrypoint_tool_plugins_are_loaded(monkeypatch):
    tool = Tool("plugin_check", "test", {"type": "object"}, lambda _: {"ok": True})
    entrypoint = SimpleNamespace(load=lambda: lambda: (tool,))
    discovered = SimpleNamespace(select=lambda **_: (entrypoint,))
    monkeypatch.setattr("quanllm_harness.tools.registry.entry_points", lambda: discovered)
    registry = ToolRegistry()
    registry.load_entrypoint_tools()
    assert registry.execute("plugin_check", {}).ok is True
