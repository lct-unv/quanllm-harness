from __future__ import annotations

import pytest

from quanllm_harness.contracts import Issue, IssueOrigin, Severity
from quanllm_harness.orchestration import (
    CancellationToken,
    ConvergenceAction,
    ConvergencePolicy,
    ExecutionGraph,
    GraphNode,
    RunCancelled,
    RunController,
    RunDeadlineExceeded,
)


def test_execution_graph_rejects_cycles_and_reports_ready_nodes():
    graph = ExecutionGraph((GraphNode("a"), GraphNode("b", ("a",))))
    assert graph.ready(set()) == ("a",)
    assert graph.ready({"a"}) == ("b",)
    with pytest.raises(ValueError, match="存在环"):
        ExecutionGraph((GraphNode("a", ("b",)), GraphNode("b", ("a",))))


def test_run_controller_supports_cancellation_and_global_deadline():
    now = [10.0]
    token = CancellationToken()
    controller = RunController(5, token, clock=lambda: now[0])
    controller.checkpoint("start")
    now[0] = 16.0
    with pytest.raises(RunDeadlineExceeded):
        controller.checkpoint("late")
    token.cancel()
    with pytest.raises(RunCancelled):
        controller.checkpoint("cancelled")


def test_convergence_policy_is_bounded_by_rounds_and_recurrence():
    issue = Issue(IssueOrigin.MODEL, Severity.MAJOR, "x", "wrong")
    policy = ConvergencePolicy(max_repair_rounds=3, duplicate_issue_limit=1)
    assert policy.evaluate([issue], completed_repair_rounds=0).action is ConvergenceAction.REPAIR
    decision = policy.evaluate([issue], completed_repair_rounds=1)
    assert decision.action is ConvergenceAction.STOP
    assert "重复" in decision.reason
