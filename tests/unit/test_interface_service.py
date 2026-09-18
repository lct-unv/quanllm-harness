from __future__ import annotations

from quanllm_harness.config import HarnessSettings
from quanllm_harness.contracts import (
    HarnessResult,
    RequestPolicy,
    RunStatus,
    Usage,
    VerificationReport,
)
from quanllm_harness.interfaces.service import HarnessService


def _result(answer: str = "ok") -> HarnessResult:
    return HarnessResult(
        status=RunStatus.VERIFIED,
        answer=answer,
        policy=RequestPolicy(),
        verification=VerificationReport(),
        events=(),
        usage=Usage(1, 2),
    )


def test_service_builds_a_fresh_harness_for_every_answer(monkeypatch):
    built = []

    class FakeHarness:
        def __init__(self, index):
            self.index = index

        def answer(self, question, *, cancellation=None):
            return _result(f"{self.index}:{question}")

    def fake_create(settings, *, event_sink=None, plugin_manager=None):
        harness = FakeHarness(len(built) + 1)
        built.append((harness, event_sink, plugin_manager))
        return harness

    monkeypatch.setattr("quanllm_harness.interfaces.service.create_harness", fake_create)
    service = HarnessService(HarnessSettings())
    assert service.answer("A").answer == "1:A"
    assert service.answer("B").answer == "2:B"
    assert len(built) == 2
