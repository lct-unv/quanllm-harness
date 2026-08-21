from __future__ import annotations

from quanllm_harness.contracts import (
    HarnessEvent,
    HarnessResult,
    RequestPolicy,
    RunStatus,
    Usage,
    VerificationReport,
)
from quanllm_harness.interfaces.cli import app as cli_app
from quanllm_harness.interfaces.cli.app import TerminalEvents, _interactive


def test_interactive_cli_again_reuses_only_the_last_question(monkeypatch, capsys):
    answers = iter(["第一题", ":again", ":quit"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))

    class FakeService:
        def __init__(self):
            self.questions = []

        def answer(self, question, *, event_sink=None):
            self.questions.append(question)
            return HarnessResult(
                status=RunStatus.VERIFIED,
                answer="答案",
                policy=RequestPolicy(),
                verification=VerificationReport(),
                events=(),
                usage=Usage(1, 1),
            )

    service = FakeService()
    assert _interactive(service) == 0
    assert service.questions == ["第一题", "第一题"]
    assert capsys.readouterr().out.count("助手：答案") == 2


def test_terminal_buffers_parallel_reasoning_by_stage(capsys):
    sink = TerminalEvents(show_reasoning=True)
    sink(HarnessEvent("reasoning_delta", "主求解", {"text": "主一"}))
    sink(HarnessEvent("reasoning_delta", "独立求解", {"text": "独一"}))
    sink(HarnessEvent("reasoning_delta", "主求解", {"text": "主二"}))
    sink(HarnessEvent("reasoning_delta", "独立求解", {"text": "独二"}))
    sink(HarnessEvent("agent_finished", "主求解", {}))
    sink(HarnessEvent("agent_finished", "独立求解", {}))

    output = capsys.readouterr().out
    assert "主一主二" in output
    assert "独一独二" in output
    assert "主一独一" not in output
    assert output.count("原始思维链") == 2


def test_terminal_process_lines_include_live_elapsed_time(monkeypatch, capsys):
    ticks = iter([10.0, 11.0, 72.0])
    monkeypatch.setattr(cli_app, "monotonic", lambda: next(ticks))
    sink = TerminalEvents(show_reasoning=True)
    sink(HarnessEvent("agent_started", "主求解", {}))
    sink(HarnessEvent("tool_finished", "operator_algebra", {"ok": True}))

    output = capsys.readouterr().out
    assert "[00小时00分钟01秒] [主求解]" in output
    assert "[00小时01分钟02秒] [工具 operator_algebra：成功]" in output
