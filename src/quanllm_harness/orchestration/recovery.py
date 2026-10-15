from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import monotonic

from ..contracts import RecoveryMetrics, Usage


@dataclass
class RecoveryTracker:
    """Clock- and counter-based collector for Issue 005 recovery metrics."""

    clock: Callable[[], float] = monotonic
    detected: bool = False
    localized: bool = False
    detected_at: float | None = None
    localized_at: float | None = None
    detected_round: int = 0
    localized_claim_id: str = ""
    localization_source: str = ""
    rollback_distance: int = 0
    verified_rounds: int = 0
    usage_at_detection: Usage | None = None
    _detected_kinds: list[str] = field(default_factory=list)
    _started_at: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        self._started_at = self.clock()

    def note_detection(
        self, kind: str, *, round_index: int = 0, usage: Usage | None = None
    ) -> None:
        """Record the first time an anomaly, contradiction or issue was found."""

        if self.detected:
            return
        self.detected = True
        self.detected_at = self.clock()
        self.detected_round = round_index
        self.usage_at_detection = usage
        self._detected_kinds.append(kind)

    def note_localization(
        self,
        claim_id: str,
        *,
        source: str,
        rollback_distance: int = 0,
    ) -> None:
        """Record the earliest incorrect step once it has been identified."""

        if self.localized:
            return
        self.localized = True
        self.localized_at = self.clock()
        self.localized_claim_id = claim_id
        self.localization_source = source
        self.rollback_distance = rollback_distance

    def note_verified(self) -> None:
        self.verified_rounds += 1

    def finish(
        self,
        *,
        usage: Usage,
        status: str,
        repair_iterations: int,
        invalid_state_leakage: int,
        notes: Sequence[str] = (),
    ) -> RecoveryMetrics:
        detection_latency = (
            max(0.0, (self.detected_at or self._started_at) - self._started_at)
            if self.detected
            else 0.0
        )
        localization_latency = (
            max(0.0, (self.localized_at or self._started_at) - self._started_at)
            if self.localized
            else 0.0
        )
        if self.detected:
            # Token cost of the recovery path only: everything spent after the
            # anomaly was first detected. A run that never detected anything has
            # no recovery cost at all.
            baseline = self.usage_at_detection or Usage()
            recovery_prompt = max(0, usage.prompt_tokens - baseline.prompt_tokens)
            recovery_completion = max(0, usage.completion_tokens - baseline.completion_tokens)
        else:
            recovery_prompt = 0
            recovery_completion = 0
        recovered = (
            self.detected
            and self.localized
            and repair_iterations > 0
            and invalid_state_leakage == 0
            and status in {"verified", "verified_with_input_ambiguity"}
        )
        return RecoveryMetrics(
            detected=self.detected,
            detected_round=self.detected_round,
            detected_kinds=tuple(self._detected_kinds),
            detection_latency_seconds=round(detection_latency, 3),
            localized=self.localized,
            localized_claim_id=self.localized_claim_id,
            localization_source=self.localization_source,
            localization_latency_seconds=round(localization_latency, 3),
            rollback_distance=self.rollback_distance,
            repair_iterations=repair_iterations,
            verified_rounds=self.verified_rounds,
            recovered=recovered,
            invalid_state_leakage=invalid_state_leakage,
            recovery_prompt_tokens=recovery_prompt,
            recovery_completion_tokens=recovery_completion,
            delivery_status=status,
            notes=tuple(notes),
        )
