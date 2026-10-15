from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

from quanllm_harness import HarnessSettings, QuanLLMHarness, RunStatus
from quanllm_harness.contracts import (
    CandidateState,
    Issue,
    IssueOrigin,
    ModelResponse,
    Severity,
    ToolCall,
    Usage,
    answer_digest,
)
from quanllm_harness.plugins import (
    InvariantResult,
    PluginManager,
    PluginManifest,
    PluginPolicy,
)
from quanllm_harness.provider import QuanLLMProvider
from quanllm_harness.tools import Tool, ToolRegistry


class ScriptedProvider(QuanLLMProvider):
    def __init__(self, scripts):
        self.scripts = {stage: deque(values) for stage, values in scripts.items()}
        self.calls = []

    def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
        self.calls.append((stage, structured, bool(tools)))
        value = self.scripts[stage].popleft()
        if isinstance(value, Exception):
            raise value
        if isinstance(value, ModelResponse):
            return value
        return ModelResponse(
            content=value,
            reasoning="测试推理" if not structured else "",
            finish_reason="stop",
            usage=Usage(10, 5),
        )


def no_tools() -> ToolRegistry:
    return ToolRegistry()


ROUTE = (
    '{"depth":"standard","suspicious_input":false,"requires_tools":false,'
    '"requires_independent_solver":false,"language":"zh","reason":"测试"}'
)
NOT_CHECKABLE = '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"测试"}]}'


def bad_issue(quote: str) -> str:
    return (
        '{"issues":[{"origin":"model","severity":"major","quote":"' + quote + '",'
        '"problem":"结论错误","correction":"重新推导","evidence_ids":[]}],"summary":"需修复"}'
    )


ADJUDICATION = (
    '{"decision":"valid","severity":"major","problem":"结论错误",'
    '"correction":"重新推导","reason":"确定性证据一致"}'
)


def test_repair_cannot_resurrect_an_invalidated_candidate():
    bad = "答案是 x=1。"
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": [bad],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"答案是 x=1",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [NOT_CHECKABLE],
            "形式与学科核验": [bad_issue("答案是 x=1")],
            "问题裁决": [ADJUDICATION],
            "定向修复": [bad],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("请给出答案。")

    assert result.status is RunStatus.DEGRADED_DELIVERY
    kinds = [event.kind for event in result.events]
    assert "invalid_state_blocked" in kinds
    assert "delivery_guard_applied" in kinds
    assert result.lifecycle.invalid_state_leakage >= 1
    assert result.recovery.detected is True
    assert result.recovery.recovered is False
    assert result.recovery.invalid_state_leakage >= 1
    assert result.recovery.delivery_status == "degraded_delivery"
    states = [record.state for record in result.lifecycle.records]
    assert CandidateState.INVALID in states
    assert any(
        transition.from_state is CandidateState.PROPOSED
        and transition.to_state is CandidateState.INVALID
        for transition in result.lifecycle.transitions
    )


def test_successful_repair_reaches_verified_lifecycle_state():
    bad = "答案是 x=1。"
    good = "答案是 x=2。"
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": [bad],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"答案是 x=1",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
                '{"claims":[{"id":"C-001","quote":"答案是 x=2",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
            ],
            "工具核验计划": [NOT_CHECKABLE, NOT_CHECKABLE],
            "形式与学科核验": [bad_issue("答案是 x=1"), '{"issues":[],"summary":"通过"}'],
            "问题裁决": [ADJUDICATION],
            "定向修复": [good],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("请给出答案。")

    assert result.status is RunStatus.VERIFIED
    assert result.answer == good
    assert result.lifecycle.invalid_state_leakage == 0
    assert result.lifecycle.checkpoints
    verified = [
        record for record in result.lifecycle.records if record.state is CandidateState.VERIFIED
    ]
    assert verified and verified[0].digest == answer_digest(good)
    assert ledger_sources(result) == ["求解·主求解", "定向修复"]
    metrics = result.recovery
    assert metrics.detected is True
    assert metrics.detected_kinds == ("model_issue",)
    # The problem was reported by the model path, which does not localize the
    # earliest incorrect step: the run recovered textually but is not counted as
    # a controlled (localized) recovery.
    assert metrics.localized is False
    assert metrics.recovered is False
    assert metrics.repair_iterations == 1
    assert metrics.verified_rounds == 1
    assert metrics.invalid_state_leakage == 0
    assert metrics.delivery_status == "verified"
    assert metrics.detection_latency_seconds >= 0.0


def ledger_sources(result) -> list[str]:
    return [record.source for record in result.lifecycle.records]


def test_lifecycle_snapshot_is_serializable():
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": ["答案是 x=2。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"答案是 x=2",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [NOT_CHECKABLE],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("请给出答案。")
    payload = result.to_dict()
    lifecycle = payload["lifecycle"]
    assert lifecycle["invalid_state_leakage"] == 0
    assert lifecycle["records"][0]["state"] == "verified"
    assert lifecycle["checkpoints"][0]["digest"] == answer_digest("答案是 x=2。")
    recovery = payload["recovery"]
    assert recovery["detected"] is False
    assert recovery["recovered"] is False
    assert recovery["verified_rounds"] == 1
    assert recovery["repair_iterations"] == 0
    assert recovery["recovery_prompt_tokens"] == 0


def equivalence_tool() -> ToolRegistry:
    def handler(arguments):
        return {
            "equivalent": arguments["left"] == arguments["right"],
            "resolved": True,
        }

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fake.equivalence",
            description="比较两个字符串表达式是否等价",
            parameters={
                "type": "object",
                "properties": {"left": {"type": "string"}, "right": {"type": "string"}},
                "required": ["left", "right"],
                "additionalProperties": False,
            },
            handler=handler,
            claim_kinds=("equation",),
        )
    )
    return registry


def tool_review(left: str, right: str, anchor: str, expected: bool) -> str:
    return json.dumps(
        {
            "decision": "approve",
            "tool": "fake.equivalence",
            "arguments": {"left": left, "right": right},
            "source_anchors": [anchor],
            "expected_boolean": expected,
            "reason": "依据候选断言核验两个表达式是否等价",
        },
        ensure_ascii=False,
    )


def test_contradicting_tool_evidence_localizes_earliest_step_and_repairs():
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": ["由计算得 C=9。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"C=9",'
                '"kind":"equation","importance":"major"}],"requirements":[]}',
                '{"claims":[{"id":"C-001","quote":"C=15",'
                '"kind":"equation","importance":"major"}],"requirements":[]}',
            ],
            "工具核验计划": [
                '{"checks":[{"claim_id":"C-001","tool":"fake.equivalence",'
                '"arguments":{"left":"9","right":"15"},"purpose":"核验 C 是否等于 9"}],'
                '"not_checkable":[]}',
                '{"checks":[{"claim_id":"C-001","tool":"fake.equivalence",'
                '"arguments":{"left":"15","right":"15"},"purpose":"核验 C 是否等于 15"}],'
                '"not_checkable":[]}',
            ],
            "工具调用审查·fake.equivalence": [
                tool_review("9", "15", "C=9", True),
                tool_review("15", "15", "C=15", True),
            ],
            "形式与学科核验": [
                '{"issues":[],"summary":"通过"}',
                '{"issues":[],"summary":"通过"}',
            ],
            "定向修复": ["由计算得 C=15。"],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=equivalence_tool(),
    ).answer("计算 C 的值。")

    assert result.status is RunStatus.VERIFIED
    assert result.answer == "由计算得 C=15。"
    assert result.repair_rounds == 1
    assert any(item.supports_claim is False for item in result.verification.evidence)
    localized = [
        problem
        for event in result.events
        if event.kind == "candidate_invalidated"
        for problem in event.payload["issues"]
    ]
    assert any("最早出现的不一致步骤" in problem for problem in localized)
    assert result.verification.assertion_timeline.rollback_target is None
    metrics = result.recovery
    assert metrics.detected_kinds == ("tool_contradiction",)
    assert metrics.localized_claim_id == "C-001"
    assert metrics.localization_source == "assertion_timeline"
    assert metrics.recovered is True
    assert metrics.recovery_prompt_tokens > 0
    assert metrics.verified_rounds == 1


class StrictMarkerPlugin:
    """Demo domain plugin whose invariant is registered through the host API."""

    manifest = PluginManifest(
        "demo-strict",
        "1.0",
        permissions=("invariants.register",),
    )

    def setup(self, context) -> None:
        def invariant(invariant_context) -> InvariantResult:
            if "错误标记" not in invariant_context.candidate:
                return InvariantResult(checked=1)
            return InvariantResult(
                checked=1,
                issues=(
                    Issue(
                        IssueOrigin.MODEL,
                        Severity.MAJOR,
                        "错误标记",
                        "领域不变量 demo-strict.marker 未通过：候选包含错误标记",
                        "移除错误标记并重新推导",
                    ),
                ),
            )

        context.invariants.register("marker", invariant)


def strict_marker_manager() -> PluginManager:
    plugin = StrictMarkerPlugin()
    entrypoint = SimpleNamespace(
        name=plugin.manifest.name, value="tests:demo-strict", load=lambda: plugin
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


def test_plugin_invariant_issue_triggers_deterministic_repair_without_adjudication():
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": ["答案是 x=1，错误标记。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"答案是 x=1",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
                '{"claims":[{"id":"C-001","quote":"答案是 x=2",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
            ],
            "工具核验计划": [NOT_CHECKABLE, NOT_CHECKABLE],
            "形式与学科核验": [
                '{"issues":[],"summary":"通过"}',
                '{"issues":[],"summary":"通过"}',
            ],
            "定向修复": ["答案是 x=2，正确标记。"],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=no_tools(),
        plugin_manager=strict_marker_manager(),
    ).answer("请给出答案。")

    assert result.status is RunStatus.VERIFIED
    assert result.answer == "答案是 x=2，正确标记。"
    assert result.repair_rounds == 1
    assert "问题裁决" not in [stage for stage, _, _ in provider.calls]
    problems = [
        problem
        for event in result.events
        if event.kind == "candidate_invalidated"
        for problem in event.payload["issues"]
    ]
    assert any("demo-strict.marker" in problem for problem in problems)


def always_failing_tool() -> ToolRegistry:
    def handler(arguments):
        raise RuntimeError("后端崩溃")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fake.fail",
            description="总是失败的工具",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
            handler=handler,
            claim_kinds=("equation",),
        )
    )
    return registry


def test_tool_failure_degrades_honestly_without_rewriting_the_answer():
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": ["由计算得 C=9。"],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"C=9",'
                '"kind":"equation","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [
                '{"checks":[{"claim_id":"C-001","tool":"fake.fail",'
                '"arguments":{"a":"C=9"},"purpose":"核验 C=9"}],"not_checkable":[]}'
            ],
            "工具调用审查·fake.fail": [tool_review_approve("fake.fail", {"a": "C=9"}, "C=9")],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=0, semantic_verifier_count=1),
        tools=always_failing_tool(),
    ).answer("求 C。")

    assert result.status is RunStatus.DEGRADED_DELIVERY
    assert result.answer == "由计算得 C=9。"
    assert result.repair_rounds == 0
    assert result.verification.model_issues == []
    assert any(
        "缺少成功的工具证据" in issue.problem for issue in result.verification.infrastructure_issues
    )
    failed = [item for item in result.verification.evidence if not item.ok]
    assert failed and failed[0].error_kind == "handler"
    # A tool failure must not invalidate the reasoning state itself: only the
    # objective-verification claim is downgraded.
    assert [record.state for record in result.lifecycle.records] == [CandidateState.VERIFIED]


def tool_review_approve(tool_name: str, arguments, anchor: str) -> str:
    return json.dumps(
        {
            "decision": "approve",
            "tool": tool_name,
            "arguments": arguments,
            "source_anchors": [anchor],
            "expected_boolean": True,
            "reason": "依据候选断言核验该关系",
        },
        ensure_ascii=False,
    )


def test_structurally_invalid_tool_call_skips_the_semantic_reviewer():
    provider = ScriptedProvider(
        {
            "任务路由": [ROUTE],
            "主求解": [
                ModelResponse(
                    content="",
                    reasoning="先调用工具",
                    finish_reason="tool_calls",
                    tool_calls=(ToolCall(id="call-1", name="unknown.tool", arguments={"a": 1}),),
                    usage=Usage(10, 5),
                ),
                "答案：直接推导结果。",
            ],
            "断言提取": [
                '{"claims":[{"id":"C-001","quote":"直接推导结果",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}'
            ],
            "工具核验计划": [NOT_CHECKABLE],
            "形式与学科核验": ['{"issues":[],"summary":"通过"}'],
        }
    )
    result = QuanLLMHarness(
        provider=provider,
        settings=HarnessSettings(max_tool_rounds=1, semantic_verifier_count=1),
        tools=no_tools(),
    ).answer("请给出答案。")

    stages = [stage for stage, _, _ in provider.calls]
    assert not any(stage.startswith("工具调用审查") for stage in stages)
    assert result.status is RunStatus.VERIFIED
    assert result.answer == "答案：直接推导结果。"
    rejected = [item for item in result.verification.evidence if item.error_kind == "schema"]
    assert rejected and rejected[0].tool == "unknown.tool"
    assert rejected[0].attempts == 0
