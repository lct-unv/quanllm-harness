from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from quanllm_harness.config import HarnessSettings
from quanllm_harness.contracts import HarnessEvent, ModelResponse
from quanllm_harness.plugins import (
    PluginManager,
    PluginManifest,
    PluginPolicy,
    PluginVerificationResult,
    SubprocessToolSpec,
    load_plugin_policy,
    set_plugin_enabled,
)
from quanllm_harness.providers import QuanLLMProvider
from quanllm_harness.tools import Tool


def _entrypoint(name, plugin):
    return SimpleNamespace(name=name, value=f"tests:{name}", load=lambda: plugin)


def _entrypoints(*plugins):
    modern = tuple(_entrypoint(plugin.manifest.name, plugin) for plugin in plugins)
    return lambda: SimpleNamespace(
        select=lambda *, group: modern if group == "quanllm_harness.plugins" else ()
    )


def _policy(*names, permissions=("tools.register",), **overrides):
    return PluginPolicy(enabled=names, allowed_permissions=permissions, **overrides)


def test_modern_plugin_registers_namespaced_tool_and_provenance():
    class DemoPlugin:
        manifest = PluginManifest(
            name="demo",
            version="1.2.3",
            permissions=("tools.register",),
        )

        def setup(self, context):
            context.tools.register(
                Tool(
                    "demo.double",
                    "double",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    lambda args: {"value": args["value"] * 2},
                )
            )

    manager = PluginManager.discover(
        _policy("demo"),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(DemoPlugin()),
    )
    evidence = manager.build_tool_registry(include_builtin=False).execute(
        "demo.double", {"value": 4}
    )
    assert evidence.ok is True
    assert evidence.result == {"value": 8}
    assert evidence.plugin_name == "demo"
    assert evidence.plugin_version == "1.2.3"
    assert evidence.execution_mode == "in_process"


def test_unlisted_plugin_is_disabled_without_running_setup():
    calls = []

    class DisabledPlugin:
        manifest = PluginManifest("disabled", "1", permissions=("tools.register",))

        def setup(self, context):
            calls.append(context)

    manager = PluginManager.discover(
        PluginPolicy(),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(DisabledPlugin()),
    )
    assert calls == []
    assert manager.statuses()[0]["state"] == "disabled"


def test_unlisted_plugin_is_not_imported_but_exposes_digest(tmp_path):
    calls = []
    installed_file = tmp_path / "plugin.py"
    installed_file.write_text("untrusted module must not be imported", encoding="utf-8")
    entrypoint = SimpleNamespace(
        name="disabled",
        value="untrusted:plugin",
        load=lambda: calls.append("imported"),
        dist=SimpleNamespace(
            files=(Path("plugin.py"),),
            locate_file=lambda _: installed_file,
        ),
    )
    manager = PluginManager.discover(
        PluginPolicy(),
        include_legacy_tools=False,
        entry_points_function=lambda: SimpleNamespace(
            select=lambda *, group: (entrypoint,) if group == "quanllm_harness.plugins" else ()
        ),
    )
    assert calls == []
    assert manager.statuses()[0]["state"] == "disabled"
    assert len(manager.statuses()[0]["digest"]) == 64


def test_dependency_services_lifecycle_events_and_request_hooks():
    observed = []

    class BasePlugin:
        manifest = PluginManifest(
            "base",
            "1",
            permissions=("services.provide",),
        )

        def setup(self, context):
            context.services.provide("value", 7)
            observed.append("base:start")
            return lambda: observed.append("base:cleanup")

    class ConsumerPlugin:
        manifest = PluginManifest(
            "consumer",
            "1",
            permissions=("services.consume", "events.subscribe"),
            dependencies=("base",),
        )

        def setup(self, context):
            observed.append(f"consumer:{context.services.get('base.value')}")
            context.events.subscribe(lambda event: observed.append(event.kind))

        def request_started(self, question):
            observed.append(f"request:{question}")

        def request_finished(self, result, error):
            observed.append("request:done")

    policy = _policy(
        "base",
        "consumer",
        permissions=("services.provide", "services.consume", "events.subscribe"),
    )
    manager = PluginManager.discover(
        policy,
        include_legacy_tools=False,
        entry_points_function=_entrypoints(ConsumerPlugin(), BasePlugin()),
    )
    manager.event_sink()(HarnessEvent("custom", "test", {}))
    manager.notify_request_start("Q")
    manager.notify_request_end(object())
    manager.shutdown()
    assert observed == [
        "base:start",
        "consumer:7",
        "custom",
        "request:Q",
        "request:done",
        "base:cleanup",
    ]


def test_denied_permission_and_digest_are_reported_without_crashing_host():
    class NetworkPlugin:
        manifest = PluginManifest(
            "networked",
            "1",
            permissions=("network.outbound",),
        )

        def setup(self, context):
            raise AssertionError("must not run")

    manager = PluginManager.discover(
        _policy("networked", require_trusted_digest=True),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(NetworkPlugin()),
    )
    status = manager.statuses()[0]
    assert status["state"] == "error"
    assert "未授权" in status["reason"] or "摘要" in status["reason"]


def test_subprocess_tool_uses_json_protocol_and_is_attributed():
    script = Path(__file__).parents[1] / "fixtures" / "plugin_subprocess.py"

    class ProcessPlugin:
        manifest = PluginManifest(
            "process",
            "1",
            permissions=("tools.register",),
            execution_mode="subprocess",
        )
        subprocess_tools = (
            SubprocessToolSpec(
                name="process.double",
                description="double",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                command=(sys.executable, str(script)),
            ),
        )

    manager = PluginManager.discover(
        _policy("process"),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(ProcessPlugin()),
    )
    evidence = manager.build_tool_registry(include_builtin=False).execute(
        "process.double", {"value": 9}
    )
    assert evidence.ok is True
    assert evidence.result == {"doubled": 18}
    assert evidence.execution_mode == "subprocess"
    assert evidence.plugin_name == "process"


def test_subprocess_tool_enforces_timeout_and_streaming_output_limit():
    script = Path(__file__).parents[1] / "fixtures" / "plugin_subprocess.py"

    class ProcessPlugin:
        manifest = PluginManifest(
            "bounded",
            "1",
            permissions=("tools.register",),
            execution_mode="subprocess",
        )
        subprocess_tools = (
            SubprocessToolSpec(
                name="bounded.run",
                description="bounded",
                parameters={"type": "object", "additionalProperties": True},
                command=(sys.executable, str(script)),
            ),
        )

    manager = PluginManager.discover(
        _policy(
            "bounded",
            call_timeout_seconds=0.05,
            max_output_bytes=128,
        ),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(ProcessPlugin()),
    )
    registry = manager.build_tool_registry(include_builtin=False)
    timed_out = registry.execute("bounded.run", {"mode": "sleep"})
    overflowed = registry.execute("bounded.run", {"mode": "overflow"})
    assert timed_out.ok is False
    assert "0.05" in timed_out.error
    assert overflowed.ok is False
    assert "字节上限" in overflowed.error


def test_verifier_and_provider_extensions_are_registered_but_cannot_mark_verified():
    class AlternateProvider(QuanLLMProvider):
        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            return ModelResponse(content="ok")

    class ExtensionPlugin:
        manifest = PluginManifest(
            "extensions",
            "1",
            permissions=("verifiers.register", "providers.register", "secrets.api_key"),
        )

        def setup(self, context):
            context.verifiers.register(
                "guard", lambda _: PluginVerificationResult(warnings=("checked",))
            )
            context.providers.register("alternate", lambda settings: AlternateProvider())

    permissions = ("verifiers.register", "providers.register", "secrets.api_key")
    manager = PluginManager.discover(
        _policy("extensions", permissions=permissions),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(ExtensionPlugin()),
    )
    assert manager.verifiers[0][0] == "extensions.guard"
    assert isinstance(
        manager.create_provider("extensions.alternate", HarnessSettings()), AlternateProvider
    )
    assert not hasattr(PluginVerificationResult(), "verified")


def test_plugin_configuration_enable_disable_is_atomic(tmp_path):
    path = tmp_path / "plugins.json"
    set_plugin_enabled("demo", True, path)
    assert load_plugin_policy(path).enabled == ("demo",)
    set_plugin_enabled("demo", False, path)
    policy = load_plugin_policy(path)
    assert policy.enabled == ()
    assert policy.disabled == ("demo",)
    assert json.loads(path.read_text(encoding="utf-8"))["disabled"] == ["demo"]
