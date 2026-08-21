from __future__ import annotations

from collections import deque

from quanllm_harness import HarnessSettings, QuanLLMHarness, RunStatus
from quanllm_harness.contracts import ModelResponse, Usage
from quanllm_harness.provider import QuanLLMProvider
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
