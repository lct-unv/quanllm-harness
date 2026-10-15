from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryMetrics:
    """Issue 005 - objective, comparable recovery telemetry for one run.

    Accuracy alone cannot measure recovery quality: a run may deliver the right
    answer after silently accepting an injected error, or may loop without ever
    locating the earliest incorrect step. These counters make the whole
    detection -> localization -> rollback -> repair -> verification path
    measurable without changing the answer semantics.
    """

    detected: bool = False
    detected_round: int = 0
    detected_kinds: Sequence[str] = ()
    detection_latency_seconds: float = 0.0
    localized: bool = False
    localized_claim_id: str = ""
    localization_source: str = ""
    localization_latency_seconds: float = 0.0
    rollback_distance: int = 0
    repair_iterations: int = 0
    verified_rounds: int = 0
    recovered: bool = False
    invalid_state_leakage: int = 0
    recovery_prompt_tokens: int = 0
    recovery_completion_tokens: int = 0
    delivery_status: str = ""
    notes: Sequence[str] = ()
