from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from quanllm_harness import HarnessSettings, QuanLLMHarness, RunStatus
from quanllm_harness.contracts import ModelResponse, Usage
from quanllm_harness.official_plugins.qm_teaching import QuantumMechanicsTeachingPlugin
from quanllm_harness.plugins import (
    DomainCandidateUpdate,
    DomainFinalization,
    PluginManager,
    PluginManifest,
    PluginPolicy,
)
from quanllm_harness.provider import ProviderError, QuanLLMProvider
from quanllm_harness.tools import ToolRegistry


class ScriptedProvider(QuanLLMProvider):
    def __init__(self, scripts):
        self.scripts = {stage: deque(values) for stage, values in scripts.items()}
        self.calls = []

    def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
        self.calls.append((stage, structured, bool(tools)))
        value = self.scripts[stage].popleft()
        if isinstance(value, Exception):
            raise value
        return ModelResponse(
            content=value,
            reasoning="测试推理" if not structured else "",
            finish_reason="stop",
            usage=Usage(10, 5),
        )


def no_tools() -> ToolRegistry:
    return ToolRegistry()


def qm_teaching_manager() -> PluginManager:
    plugin = QuantumMechanicsTeachingPlugin()
    entrypoint = SimpleNamespace(
        name=plugin.manifest.name,
        value="quanllm_harness.official_plugins.qm_teaching:plugin",
        load=lambda: plugin,
    )
    return PluginManager.discover(
        PluginPolicy(
            enabled=(plugin.manifest.name,),
            allowed_permissions=plugin.manifest.permissions,
        ),
        include_legacy_tools=False,
        entry_points_function=lambda: SimpleNamespace(
            select=lambda *, group: (entrypoint,) if group == "quanllm_harness.plugins" else ()
        ),
    )


def test_single_solver_failure_falls_back_to_independent_solver():
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"计算题"}'
            ],
            "主求解": [ProviderError("超时")],
            "独立求解": ["备用答案：1/21。"],
            "断言提取": ['{"claims":[],"requirements":[]}'],
            "工具核验计划": ['{"checks":[],"not_checkable":[]}'],
            "形式与学科核验": ['{"canonical_core":[],"issues":[],"summary":"通过"}'],
        }
    )

    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("请计算投影比。")

    assert result.status is RunStatus.DEGRADED_DELIVERY
    assert result.answer == "备用答案：1/21。"
    assert any(event.kind == "solver_fallback" for event in result.events)


def test_verified_single_solver_flow_uses_separate_request_modes():
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"概念题"}'
            ],
            "主求解": ["候选答案：纯态满足 rho^2=rho。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"纯态满足 rho^2=rho",'
                '"kind":"equation","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"概念陈述"}]}'
            ],
            "形式与学科核验": ['{"canonical_core":[],"issues":[],"summary":"通过"}'],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("纯态的密度矩阵判据是什么？")

    assert result.status is RunStatus.VERIFIED
    assert result.answer.startswith("候选答案")
    assert result.usage == Usage(50, 25)
    modes = {stage: structured for stage, structured, _ in provider.calls}
    assert modes["主求解"] is False
    assert modes["任务路由"] is True
    assert modes["断言提取"] is True
    assert modes["工具核验计划"] is True
    assert modes["形式与学科核验"] is True


def test_model_issue_triggers_repair_and_full_reverification():
    bad = "答案说能同时精确确定位置和动量。"
    good = "位置和动量算符不对易，不能在同一量子态中同时具有确定值。"
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"概念辨析"}'
            ],
            "主求解": [bad],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"能同时精确确定位置和动量",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
                '{"claims":[{"id":"C-001","quote":"不能在同一量子态中同时具有确定值",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"概念陈述"}]}',
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"概念陈述"}]}',
            ],
            "形式与学科核验": [
                '{"issues":[{"origin":"model","severity":"major",'
                '"quote":"能同时精确确定位置和动量","problem":"违反不确定关系",'
                '"correction":"说明算符不对易","evidence_ids":[]}],"summary":"需修复"}',
                '{"issues":[],"summary":"通过"}',
            ],
            "问题裁决": [
                '{"decision":"valid","severity":"major","problem":"违反不确定关系",'
                '"correction":"说明算符不对易","reason":"算符不对易"}'
            ],
            "定向修复": [good],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("仪器足够精确能否同时确定位置和动量？")

    assert result.status is RunStatus.VERIFIED
    assert result.answer == good
    assert result.repair_rounds == 1
    assert [call[0] for call in provider.calls].count("断言提取") == 2
    assert [call[0] for call in provider.calls].count("形式与学科核验") == 2


def test_protocol_warning_never_becomes_verified():
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"测试"}'
            ],
            "主求解": ["答案包含断言 A。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"并不存在的引文",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(
            max_tool_rounds=0,
            semantic_verifier_count=1,
            protocol_retry_count=0,
        ),
        tools=no_tools(),
    ).answer("测试问题")

    assert result.status is RunStatus.DEGRADED_DELIVERY
    assert result.answer == "答案包含断言 A。"
    assert result.verification.protocol_warnings


def test_optional_verifier_failure_does_not_downgrade_successfully_verified_answer():
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"测试"}'
            ],
            "主求解": ["答案包含结论 A。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"结论 A",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"概念结论"}]}'
            ],
            "形式与学科核验": ['{"issues":[],"summary":"学科核验通过"}'],
            "要求与教学核验": ["not json", "still not json"],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=2),
        tools=no_tools(),
    ).answer("测试问题")

    assert result.status is RunStatus.VERIFIED
    assert result.verification.protocol_warnings
    assert result.verification.verifier_summaries == ("学科核验通过",)


def test_input_issue_is_rejected_when_candidate_discloses_no_ambiguity():
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"清晰计算题"}'
            ],
            "主求解": ["答案完整且结论正确。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"结论正确",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"测试"}]}'
            ],
            "形式与学科核验": [
                '{"issues":[{"origin":"input","severity":"minor",'
                '"quote":"清晰问题","problem":"误报为输入歧义",'
                '"correction":"","evidence_ids":[]}],"summary":"误报"}'
            ],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("清晰问题")

    assert result.status is RunStatus.VERIFIED
    assert result.verification.input_issues == []
    assert all(stage != "问题裁决" for stage, _, _ in provider.calls)


def test_issue_adjudication_protocol_failure_never_triggers_repair():
    candidate = "答案使用标准高斯积分公式得到归一化结果。"
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"积分题"}'
            ],
            "主求解": [candidate],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"归一化结果",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"测试"}]}'
            ],
            "形式与学科核验": [
                '{"issues":[{"origin":"model","severity":"minor",'
                '"quote":"标准高斯积分公式","problem":"没有从头证明",'
                '"correction":"补充证明","evidence_ids":[]}],"summary":"报告问题"}'
            ],
            "问题裁决": ["not json"],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(
            max_tool_rounds=0,
            semantic_verifier_count=1,
            protocol_retry_count=0,
        ),
        tools=no_tools(),
    ).answer("计算归一化积分。")

    assert result.answer == candidate
    assert result.repair_rounds == 0
    assert result.verification.model_issues == []
    assert any("问题裁决未完成" in item for item in result.verification.protocol_warnings)


def test_duplicate_verifier_reports_are_adjudicated_once():
    candidate = "答案使用标准高斯积分公式得到归一化结果。"
    issue = (
        '{"issues":[{"origin":"model","severity":"minor",'
        '"quote":"标准高斯积分公式","problem":"没有从头证明",'
        '"correction":"补充证明","evidence_ids":[]}],"summary":"报告问题"}'
    )
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"积分题"}'
            ],
            "主求解": [candidate],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"归一化结果",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"测试"}]}'
            ],
            "形式与学科核验": [issue],
            "要求与教学核验": [issue],
            "问题裁决": [
                '{"decision":"invalid","severity":"minor","problem":"",'
                '"correction":"","reason":"用户没有要求重证标准公式"}'
            ],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=2),
        tools=no_tools(),
    ).answer("计算归一化积分。")

    assert result.status is RunStatus.VERIFIED
    assert result.repair_rounds == 0
    assert [stage for stage, _, _ in provider.calls].count("问题裁决") == 1


def test_synthesis_failure_falls_back_to_deliverable_solver_candidate():
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"deep","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":true,"language":"zh","reason":"复杂题"}'
            ],
            "主求解": ["主求解已恢复的正确答案。"],
            "独立求解": ["独立正确答案。"],
            "候选综合": [ProviderError("候选综合的工具参数不是合法 JSON")],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"正确答案",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"测试"}]}'
            ],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(
            max_tool_rounds=0,
            semantic_verifier_count=1,
            parallel_solvers=False,
        ),
        tools=no_tools(),
    ).answer("复杂测试")

    assert result.answer == "主求解已恢复的正确答案。"
    assert result.status is RunStatus.DEGRADED_DELIVERY
    assert any(event.kind == "synthesis_fallback" for event in result.events)


def test_synthesis_regression_is_blocked_by_deterministic_invariant():
    question = r"""
    U=\frac{1}{\sqrt{2}}\begin{bmatrix}1&i\\i&1\end{bmatrix},
    V=\begin{bmatrix}1&0\\0&i\end{bmatrix}. Compute W=VU.
    """
    correct = r"""
    正确乘积：$W=VU=\frac{1}{\sqrt{2}}
    \begin{bmatrix}1&i\\-1&i\end{bmatrix}$.
    """
    regressed = r"""
    错误综合：$W=VU=\begin{bmatrix}
    1/\sqrt{2}&-i/\sqrt{2}\\i/\sqrt{2}&1/\sqrt{2}
    \end{bmatrix}$.
    """
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"deep","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":true,"language":"zh","reason":"矩阵题"}'
            ],
            "主求解": [correct],
            "独立求解": [correct],
            "候选综合": [regressed],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"正确乘积",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"回归测试"}]}'
            ],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(
            max_tool_rounds=0,
            semantic_verifier_count=1,
            parallel_solvers=False,
        ),
        tools=no_tools(),
        plugin_manager=qm_teaching_manager(),
    ).answer(question)

    assert result.answer == correct
    assert result.status is RunStatus.VERIFIED
    assert any(event.kind == "synthesis_regression_blocked" for event in result.events)


def test_domain_finalization_mutation_cannot_reuse_verified_status():
    class FinalizingStrategy:
        def matches(self, question):
            return True

        def candidate_issues(self, question, candidate):
            return ()

        def fallback_answer(self, question):
            return ""

        def correct_candidate(self, question, candidate):
            return DomainCandidateUpdate(candidate)

        def repair_instructions(self, question, candidate):
            return ()

        def repair_hint(self, question):
            return ""

        def finalize(self, question, candidate):
            return DomainFinalization(candidate + " [domain finalization]")

    class FinalizingPlugin:
        manifest = PluginManifest(
            name="finalizing-domain",
            version="1.0.0",
            permissions=("domains.register",),
        )

        def setup(self, context):
            context.domains.register("test", FinalizingStrategy())

    plugin = FinalizingPlugin()
    entrypoint = SimpleNamespace(
        name=plugin.manifest.name, value="test:plugin", load=lambda: plugin
    )
    manager = PluginManager.discover(
        PluginPolicy(enabled=(plugin.manifest.name,)),
        include_legacy_tools=False,
        entry_points_function=lambda: SimpleNamespace(
            select=lambda *, group: (entrypoint,) if group == "quanllm_harness.plugins" else ()
        ),
    )
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"test"}'
            ],
            "主求解": ["verified candidate"],
            "断言提取": ['{"claims":[],"requirements":[]}'],
            "工具核验计划": ['{"checks":[],"not_checkable":[]}'],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )

    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
        plugin_manager=manager,
    ).answer("generic test")

    assert result.answer.endswith("[domain finalization]")
    assert result.status is RunStatus.DEGRADED_DELIVERY
    assert any(event.kind == "domain_finalization_applied" for event in result.events)


def test_plugin_verifier_failure_cannot_reuse_successful_semantic_status():
    class BrokenVerifierPlugin:
        manifest = PluginManifest(
            name="broken-verifier",
            version="1.0.0",
            permissions=("verifiers.register",),
        )

        def setup(self, context):
            def fail(_verification):
                raise RuntimeError("deterministic gate unavailable")

            context.verifiers.register("gate", fail)

    plugin = BrokenVerifierPlugin()
    entrypoint = SimpleNamespace(
        name=plugin.manifest.name, value="test:plugin", load=lambda: plugin
    )
    manager = PluginManager.discover(
        PluginPolicy(enabled=(plugin.manifest.name,)),
        include_legacy_tools=False,
        entry_points_function=lambda: SimpleNamespace(
            select=lambda *, group: (entrypoint,) if group == "quanllm_harness.plugins" else ()
        ),
    )
    provider = ScriptedProvider(
        {
            "任务路由": [
                '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","reason":"test"}'
            ],
            "主求解": ["candidate"],
            "断言提取": ['{"claims":[],"requirements":[]}'],
            "工具核验计划": ['{"checks":[],"not_checkable":[]}'],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )

    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
        plugin_manager=manager,
    ).answer("generic test")

    assert result.status is RunStatus.DEGRADED_DELIVERY
    assert result.verification.infrastructure_issues
    assert any(
        "deterministic gate unavailable" in item.problem for item in result.verification.issues
    )
