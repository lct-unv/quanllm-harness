from __future__ import annotations

from types import SimpleNamespace

from quanllm_harness.contracts import Issue, IssueOrigin, Severity
from quanllm_harness.official_plugins.qm_teaching import QuantumMechanicsTeachingPlugin
from quanllm_harness.plugins import (
    InvariantContext,
    InvariantResult,
    PluginManager,
    PluginManifest,
    PluginPolicy,
    PluginVerificationContext,
)


def _entrypoints(*plugins):
    modern = tuple(
        SimpleNamespace(
            name=plugin.manifest.name,
            value=f"tests:{plugin.manifest.name}",
            load=lambda p=plugin: p,
        )
        for plugin in plugins
    )
    return lambda: SimpleNamespace(
        select=lambda *, group: modern if group == "quanllm_harness.plugins" else ()
    )


def _policy(*names, permissions=("invariants.register",), **overrides):
    return PluginPolicy(enabled=names, allowed_permissions=permissions, **overrides)


def _context() -> PluginVerificationContext:
    return PluginVerificationContext(
        question="演示问题",
        candidate="演示候选",
        claims=(),
        requirements=(),
        evidence=(),
    )


def test_registered_invariant_is_namespaced_and_reported():
    def gate(context: InvariantContext) -> InvariantResult:
        return InvariantResult(
            checked=1,
            issues=(Issue(IssueOrigin.MODEL, Severity.MAJOR, "", "演示不变量未通过", "修正"),),
            notes=("已检查 1 项",),
        )

    class DemoPlugin:
        manifest = PluginManifest("demo", "1.0", permissions=("invariants.register",))

        def setup(self, context):
            context.invariants.register("gate", gate)

    manager = PluginManager.discover(
        _policy("demo"),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(DemoPlugin()),
    )
    assert [name for name, _, _ in manager.invariants] == ["demo.gate"]
    assert manager.statuses()[0]["invariants"] == ["demo.gate"]

    run = manager.run_invariants(_context())
    assert len(run.findings) == 1
    finding = run.findings[0]
    assert (finding.plugin, finding.invariant, finding.checked) == ("demo", "demo.gate", 1)
    assert run.issues[0].problem == "演示不变量未通过"
    assert run.failures == ()


def test_invariant_registration_requires_declared_permission():
    class SneakyPlugin:
        manifest = PluginManifest("sneaky", "1.0")

        def setup(self, context):
            context.invariants.register("gate", lambda context: InvariantResult())

    manager = PluginManager.discover(
        _policy("sneaky"),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(SneakyPlugin()),
    )
    status = manager.statuses()[0]
    assert status["state"] == "error"
    assert "invariants.register" in status["reason"]
    assert manager.invariants == ()


def test_invariant_with_wrong_return_type_is_a_visible_failure():
    class BrokenPlugin:
        manifest = PluginManifest("broken", "1.0", permissions=("invariants.register",))

        def setup(self, context):
            context.invariants.register("gate", lambda context: {"checked": 1})

    manager = PluginManager.discover(
        _policy("broken"),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(BrokenPlugin()),
    )
    run = manager.run_invariants(_context())
    assert run.findings == ()
    assert len(run.failures) == 1
    assert "必须返回 InvariantResult" in run.failures[0]
    assert "不变量 broken.gate 执行失败" in manager.diagnostics()[-1]


def test_duplicate_invariant_names_are_rejected_and_plugin_is_disabled():
    class DuplicatePlugin:
        manifest = PluginManifest("dup", "1.0", permissions=("invariants.register",))

        def setup(self, context):
            context.invariants.register("gate", lambda context: InvariantResult())
            context.invariants.register("gate", lambda context: InvariantResult())

    manager = PluginManager.discover(
        _policy("dup"),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(DuplicatePlugin()),
    )
    assert manager.statuses()[0]["state"] == "error"
    assert "重复注册" in manager.statuses()[0]["reason"]
    assert manager.invariants == ()


def test_official_plugin_registers_the_four_domain_invariants():
    plugin = QuantumMechanicsTeachingPlugin()
    manager = PluginManager.discover(
        _policy("quanllm-qm-teaching", permissions=plugin.manifest.permissions),
        include_legacy_tools=False,
        entry_points_function=_entrypoints(plugin),
    )
    names = {name for name, _, _ in manager.invariants}
    assert names == {
        "quanllm-qm-teaching.projection",
        "quanllm-qm-teaching.unitary",
        "quanllm-qm-teaching.hermitian",
        "quanllm-qm-teaching.measurement",
    }
    assert manager.statuses()[0]["state"] == "active"
