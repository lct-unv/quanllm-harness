from __future__ import annotations

import json
from threading import Barrier

import pytest

from quanllm_harness import CancellationToken, HarnessSettings, QuanLLMHarness, RunStatus
from quanllm_harness.agents import AgentRuntime, HarnessAgents
from quanllm_harness.cli import main
from quanllm_harness.contracts import Claim, Evidence, ModelResponse, ToolCall, Usage
from quanllm_harness.orchestration import RunCancelled
from quanllm_harness.provider import QuanLLMProvider
from quanllm_harness.tools import ToolRegistry, default_tool_registry
from quanllm_harness.tools.sympy_backend import (
    boundary_match,
    dimension_check,
    fock_ladder_expectation,
)
from quanllm_harness.verification import MathematicalVerifier, VerificationEngine


class ParallelProvider(QuanLLMProvider):
    def __init__(self):
        self.barrier = Barrier(2, timeout=2)
        self.calls = []

    def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
        self.calls.append(stage)
        if stage in {"主求解", "独立求解"}:
            self.barrier.wait()
            content = "主解：结论正确。" if stage == "主求解" else "独立解：结论正确。"
        else:
            content = {
                "任务路由": '{"depth":"deep","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":true,"language":"zh","tool_domains":[],"reason":"复杂题"}',
                "候选综合": "综合答案：结论正确。",
                "断言提取": '{"claims":[{"id":"C-001","quote":"结论正确",'
                '"kind":"conclusion","importance":"major"}],"requirements":[]}',
                "工具核验计划": '{"checks":[],"not_checkable":[{"claim_id":"C-001","reason":"测试断言"}]}',
                "形式与学科核验": '{"canonical_core":[],"issues":[],"summary":"通过"}',
            }[stage]
        return ModelResponse(
            content=content,
            reasoning="r" if not structured else "",
            finish_reason="stop",
            usage=Usage(1, 1),
        )


def test_deep_solvers_really_run_in_parallel():
    provider = ParallelProvider()
    settings = HarnessSettings(
        max_tool_rounds=0,
        parallel_solvers=True,
        semantic_verifier_count=1,
        protocol_retry_count=0,
    )
    result = QuanLLMHarness(provider=provider, settings=settings, tools=ToolRegistry()).answer(
        "复杂测试"
    )
    assert result.status is RunStatus.VERIFIED
    assert "综合答案" in result.answer
    assert "主求解" in provider.calls and "独立求解" in provider.calls


def test_protocol_validation_retries_once():
    class RetryProvider(QuanLLMProvider):
        def __init__(self):
            self.count = 0

        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            self.count += 1
            content = (
                '{"depth":"wrong"}'
                if self.count == 1
                else '{"depth":"simple","suspicious_input":false,"requires_tools":false,'
                '"requires_independent_solver":false,"language":"zh","tool_domains":[],"reason":"重试成功"}'
            )
            return ModelResponse(content=content, finish_reason="stop")

    provider = RetryProvider()
    runtime = AgentRuntime(provider, ToolRegistry(), HarnessSettings(protocol_retry_count=1))
    policy = HarnessAgents(runtime).route("测试")
    assert policy.depth == "simple"
    assert provider.count == 2


def test_scalar_tool_rejects_ket_notation_before_execution():
    registry = default_tool_registry()
    with pytest.raises(ValueError, match="态矢或抽象算符"):
        registry.validate_call(
            "symbolic_calculate",
            {"operation": "simplify", "expression": "|+z⟩"},
            claim_kind="equation",
        )


def test_cached_evidence_is_upserted_instead_of_duplicated():
    registry = default_tool_registry()
    runtime = AgentRuntime(ParallelProvider(), registry, HarnessSettings(protocol_retry_count=0))
    arguments = {"operation": "simplify", "expression": "1 + 1"}

    first = registry.execute("symbolic_calculate", arguments)
    second = registry.execute("symbolic_calculate", arguments)
    runtime.upsert_evidence(first)
    runtime.upsert_evidence(second)

    assert first.id == second.id
    assert runtime.evidence == [first]


def test_absorb_deduplicates_evidence_ids():
    settings = HarnessSettings()
    registry = ToolRegistry()
    target = AgentRuntime(ParallelProvider(), registry, settings)
    source = AgentRuntime(ParallelProvider(), registry, settings)
    evidence = Evidence("E-0001", "test", "test", {}, True)
    source.evidence.extend((evidence, evidence))

    target.absorb(source)

    assert target.evidence == [evidence]


def test_quantum_specific_objective_tools():
    assert fock_ladder_expectation({"power": 4})["expectation"] == "6*n**2 + 6*n + 3"
    dims = dimension_check({"expression": "hbar*Hz", "target_expression": "J"})
    assert dims["equivalent"] is True
    boundary = boundary_match(
        {
            "left_expression": "A*cos(k*x)",
            "right_expression": "B*exp(-q*(x-a))",
            "variable": "x",
            "point": "a",
            "symbols": ["A", "B", "k", "q", "a"],
            "derivative_order": 1,
        }
    )
    assert boundary["checks"][0]["difference"] == "A*cos(a*k) - B"
    assert boundary["checks"][1]["difference"] == "-A*k*sin(a*k) + B*q"


def test_capabilities_command_is_offline(capsys):
    assert main(["--capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "symbolic_calculate" in payload
    assert payload["quantum_backend_status"]["planner_visible"] is False


def test_execution_graph_command_is_offline(capsys):
    assert main(["--graph"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "route"
    assert any(node["repeatable"] for node in payload)


def test_pre_cancelled_run_never_calls_provider():
    class NeverProvider(QuanLLMProvider):
        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            raise AssertionError("provider must not be called")

    token = CancellationToken()
    token.cancel()
    harness = QuanLLMHarness(
        provider=NeverProvider(), settings=HarnessSettings(), tools=ToolRegistry()
    )
    with pytest.raises(RunCancelled):
        harness.answer("测试", cancellation=token)


def test_advisory_checkpoint_failure_does_not_stop_other_verifiers():
    class CheckpointProvider(QuanLLMProvider):
        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            if stage.startswith("断言提取"):
                content = (
                    '{"claims":[{"id":"C-001","quote":"定义正确","kind":"definition",'
                    '"importance":"major"}],"requirements":[]}'
                )
            elif stage.startswith("工具核验计划"):
                content = (
                    '{"checks":[{"claim_id":"C-001","tool":"symbolic_calculate",'
                    '"arguments":{"operation":"simplify","expression":"1"}}],'
                    '"not_checkable":[]}'
                )
            elif stage.startswith("工具调用审查"):
                content = (
                    '{"decision":"approve","tool":"symbolic_calculate",'
                    '"arguments":{},"reason":"参数忠实"}'
                )
            elif stage.startswith("形式与学科核验"):
                content = "not json"
            elif stage.startswith("要求与教学核验"):
                content = '{"issues":[],"summary":"教学通过"}'
            else:
                raise AssertionError(stage)
            return ModelResponse(content=content, finish_reason="stop")

    runtime = AgentRuntime(
        CheckpointProvider(),
        default_tool_registry(),
        HarnessSettings(protocol_retry_count=1, semantic_verifier_count=2),
    )
    report = VerificationEngine(runtime).verify("解释定义", "定义正确")
    assert report.verifier_summaries == ("教学通过",)
    assert len(report.protocol_warnings) == 2
    assert "工具核验计划未完成" in report.protocol_warnings[0]
    assert "形式与学科核验未完成" in report.protocol_warnings[1]


def test_tool_call_reviewer_corrects_lost_imaginary_unit_once():
    class ReviewProvider(QuanLLMProvider):
        def __init__(self):
            self.review_calls = 0

        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            if stage == "工具核验计划":
                content = (
                    '{"checks":[{"claim_id":"C-001","tool":"matrix_calculate",'
                    '"arguments":{"operation":"commutator","matrix":[[0,1],[1,0]],'
                    '"other_matrix":[[0,-1],[1,0]]},"purpose":"核验泡利矩阵对易子"}],'
                    '"not_checkable":[]}'
                )
            elif stage == "工具调用审查·matrix_calculate":
                self.review_calls += 1
                content = (
                    '{"decision":"correct","tool":"matrix_calculate",'
                    '"arguments":{"operation":"commutator","matrix":[[0,1],[1,0]],'
                    '"other_matrix":[[0,"-I"],["I",0]]},'
                    '"source_anchors":["σy=[[0,-I],[I,0]]"],'
                    '"reason":"原调用丢失了题目中的虚数单位 I"}'
                )
            else:
                raise AssertionError(stage)
            return ModelResponse(content=content, finish_reason="stop")

    provider = ReviewProvider()
    runtime = AgentRuntime(
        provider, default_tool_registry(), HarnessSettings(protocol_retry_count=1)
    )
    evidence, warnings = MathematicalVerifier(runtime).collect(
        "σy=[[0,-I],[I,0]]",
        "对易子为2*I乘以σz",
        [Claim("C-001", "对易子为2*I乘以σz", "equation")],
    )

    assert warnings == []
    assert provider.review_calls == 1
    assert evidence[0].result["matrix"] == [["2*I", "0"], ["0", "-2*I"]]
    assert evidence[0].input_verified is True
    assert evidence[0].claim_ids == ("C-001",)


def test_tool_call_review_protocol_failure_is_not_retried_or_executed():
    class InvalidReviewProvider(QuanLLMProvider):
        def __init__(self):
            self.review_calls = 0

        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            if stage == "工具核验计划":
                content = (
                    '{"checks":[{"claim_id":"C-001","tool":"symbolic_calculate",'
                    '"arguments":{"operation":"simplify","expression":"1"},'
                    '"purpose":"核验数值"}],"not_checkable":[]}'
                )
            elif stage == "工具调用审查·symbolic_calculate":
                self.review_calls += 1
                content = "not json"
            else:
                raise AssertionError(stage)
            return ModelResponse(content=content, finish_reason="stop")

    provider = InvalidReviewProvider()
    runtime = AgentRuntime(
        provider, default_tool_registry(), HarnessSettings(protocol_retry_count=1)
    )
    evidence, warnings = MathematicalVerifier(runtime).collect(
        "计算一加一", "结果是2", [Claim("C-001", "结果是2", "equation")]
    )

    assert provider.review_calls == 1
    assert evidence == []
    assert len(warnings) == 1
    assert runtime.evidence == []


def test_solver_tool_call_is_corrected_before_execution():
    class SolverToolProvider(QuanLLMProvider):
        def __init__(self):
            self.solver_calls = 0
            self.review_calls = 0

        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            if stage == "工具调用审查·matrix_calculate":
                self.review_calls += 1
                return ModelResponse(
                    content=(
                        '{"decision":"correct","tool":"matrix_calculate",'
                        '"arguments":{"operation":"commutator","matrix":[[0,1],[1,0]],'
                        '"other_matrix":[[0,"-I"],["I",0]]},'
                        '"source_anchors":["σy包含虚数单位I"],'
                        '"reason":"补回原题中的虚数单位 I"}'
                    ),
                    finish_reason="stop",
                )
            if stage != "主求解":
                raise AssertionError(stage)
            self.solver_calls += 1
            if self.solver_calls == 1:
                return ModelResponse(
                    content="",
                    reasoning="需要计算矩阵对易子",
                    finish_reason="tool_calls",
                    tool_calls=(
                        ToolCall(
                            "call-1",
                            "matrix_calculate",
                            {
                                "operation": "commutator",
                                "matrix": [[0, 1], [1, 0]],
                                "other_matrix": [[0, -1], [1, 0]],
                            },
                        ),
                    ),
                )
            assert "2*I" in str(messages[-1])
            return ModelResponse(content="对易子为2Iσz。", finish_reason="stop")

    provider = SolverToolProvider()
    runtime = AgentRuntime(
        provider,
        default_tool_registry(),
        HarnessSettings(max_tool_rounds=2, protocol_retry_count=1),
    )

    candidate = runtime.reason("求解", "σy包含虚数单位I", stage="主求解")

    assert candidate.answer == "对易子为2Iσz。"
    assert provider.review_calls == 1
    assert runtime.evidence[0].result["matrix"] == [["2*I", "0"], ["0", "-2*I"]]
    assert runtime.evidence[0].input_verified is True


def test_comparison_support_respects_claim_polarity():
    class NegativeClaimProvider(QuanLLMProvider):
        def complete(self, messages, *, stage, structured=False, tools=(), event_sink=None):
            if stage == "工具核验计划":
                content = (
                    '{"checks":[{"claim_id":"C-001","tool":"compare_expressions",'
                    '"arguments":{"lhs":"1","rhs":"2"},"purpose":"核验两式不相等"}],'
                    '"not_checkable":[]}'
                )
            elif stage == "工具调用审查·compare_expressions":
                content = (
                    '{"decision":"approve","tool":"compare_expressions",'
                    '"arguments":{"lhs":"1","rhs":"2"},'
                    '"source_anchors":["一和二不相等"],'
                    '"expected_boolean":false,"reason":"不相等断言要求 equivalent=false"}'
                )
            else:
                raise AssertionError(stage)
            return ModelResponse(content=content, finish_reason="stop")

    runtime = AgentRuntime(NegativeClaimProvider(), default_tool_registry(), HarnessSettings())
    evidence, warnings = MathematicalVerifier(runtime).collect(
        "判断一和二是否相等", "一和二不相等", [Claim("C-001", "一和二不相等", "conclusion")]
    )

    assert warnings == []
    assert evidence[0].result["equivalent"] is False
    assert evidence[0].supports_claim is True
