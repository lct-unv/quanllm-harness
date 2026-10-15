from __future__ import annotations

from quanllm_harness.contracts import RecoveryMetrics, Usage
from quanllm_harness.orchestration import RecoveryTracker


class FakeClock:
    def __init__(self, *values: float):
        self.values = list(values)
        self.index = 0

    def __call__(self) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


def test_default_metrics_describe_a_first_pass_run_without_recovery():
    metrics = RecoveryMetrics()
    assert metrics.detected is False
    assert metrics.localized is False
    assert metrics.recovered is False
    assert metrics.repair_iterations == 0
    assert metrics.invalid_state_leakage == 0


def test_tracker_records_latency_localization_and_success():
    tracker = RecoveryTracker(clock=FakeClock(0.0, 1.5, 3.0))
    tracker.note_detection("tool_contradiction", round_index=0, usage=Usage(100, 20))
    tracker.note_localization("C-002", source="assertion_timeline", rollback_distance=3)
    tracker.note_verified()
    metrics = tracker.finish(
        usage=Usage(400, 90),
        status="verified",
        repair_iterations=1,
        invalid_state_leakage=0,
    )
    assert metrics.detected is True
    assert metrics.detected_kinds == ("tool_contradiction",)
    assert metrics.detection_latency_seconds == 1.5
    assert metrics.localization_latency_seconds == 3.0
    assert metrics.localized_claim_id == "C-002"
    assert metrics.localization_source == "assertion_timeline"
    assert metrics.rollback_distance == 3
    assert metrics.verified_rounds == 1
    assert metrics.recovered is True
    assert metrics.recovery_prompt_tokens == 300
    assert metrics.recovery_completion_tokens == 70
    assert metrics.delivery_status == "verified"


def test_first_detection_and_localization_win():
    tracker = RecoveryTracker(clock=FakeClock(0.0, 1.0, 2.0))
    tracker.note_detection("model_issue")
    tracker.note_detection("tool_contradiction")
    tracker.note_localization("C-001", source="assertion_timeline", rollback_distance=2)
    tracker.note_localization("C-009", source="model", rollback_distance=9)
    metrics = tracker.finish(
        usage=Usage(10, 10),
        status="verified",
        repair_iterations=1,
        invalid_state_leakage=0,
    )
    assert metrics.detected_kinds == ("model_issue",)
    assert metrics.detected_round == 0
    assert metrics.localized_claim_id == "C-001"
    assert metrics.rollback_distance == 2


def test_detection_without_localization_is_not_a_successful_recovery():
    tracker = RecoveryTracker(clock=FakeClock(0.0, 1.0))
    tracker.note_detection("model_issue")
    metrics = tracker.finish(
        usage=Usage(50, 10),
        status="degraded_delivery",
        repair_iterations=2,
        invalid_state_leakage=0,
    )
    assert metrics.detected is True
    assert metrics.localized is False
    assert metrics.recovered is False
    assert metrics.localization_latency_seconds == 0.0


def test_first_pass_correct_answer_is_not_counted_as_recovery():
    tracker = RecoveryTracker(clock=FakeClock(0.0))
    tracker.note_verified()
    metrics = tracker.finish(
        usage=Usage(80, 30),
        status="verified",
        repair_iterations=0,
        invalid_state_leakage=0,
    )
    assert metrics.detected is False
    assert metrics.recovered is False
    assert metrics.verified_rounds == 1
    assert metrics.recovery_prompt_tokens == 0


def test_invalid_state_leakage_blocks_recovery_success():
    tracker = RecoveryTracker(clock=FakeClock(0.0, 0.5, 1.0))
    tracker.note_detection("model_issue")
    tracker.note_localization("C-001", source="assertion_timeline", rollback_distance=0)
    metrics = tracker.finish(
        usage=Usage(100, 20),
        status="verified",
        repair_iterations=1,
        invalid_state_leakage=1,
    )
    assert metrics.recovered is False
    assert metrics.invalid_state_leakage == 1


def test_token_cost_is_clamped_when_detection_usage_is_missing():
    tracker = RecoveryTracker(clock=FakeClock(0.0, 1.0))
    tracker.note_detection("model_issue")
    metrics = tracker.finish(
        usage=Usage(20, 5),
        status="degraded_delivery",
        repair_iterations=1,
        invalid_state_leakage=0,
    )
    assert metrics.recovery_prompt_tokens == 20
    assert metrics.recovery_completion_tokens == 5
