from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Issue 002 - earliest error localization.
#
# Verification claims arrive in candidate reading order, which is the order in
# which a derivation presents its steps. That linear order is used as the
# computation dependency order: the first contradicted assertion is the earliest
# incorrect step, and every later assertion is treated as downstream of it.


class AssertionStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    NOT_CHECKABLE = "not_checkable"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class AssertionEntry:
    index: int
    claim_id: str
    quote: str
    kind: str
    status: AssertionStatus
    evidence_ids: Sequence[str] = ()
    contradiction_evidence_ids: Sequence[str] = ()


@dataclass(frozen=True)
class RollbackTarget:
    """Earliest incorrect step plus the assertions that depend on it."""

    claim_id: str
    index: int
    quote: str
    evidence_ids: Sequence[str] = ()
    downstream_claim_ids: Sequence[str] = ()
    rollback_distance: int = 0


@dataclass(frozen=True)
class AssertionTimeline:
    entries: Sequence[AssertionEntry] = ()

    @property
    def contradicted(self) -> tuple[AssertionEntry, ...]:
        return tuple(
            entry for entry in self.entries if entry.status is AssertionStatus.CONTRADICTED
        )

    @property
    def earliest_violation(self) -> AssertionEntry | None:
        violations = self.contradicted
        if not violations:
            return None
        return min(violations, key=lambda entry: entry.index)

    @property
    def rollback_target(self) -> RollbackTarget | None:
        earliest = self.earliest_violation
        if earliest is None:
            return None
        downstream = tuple(entry.claim_id for entry in self.entries if entry.index > earliest.index)
        return RollbackTarget(
            claim_id=earliest.claim_id,
            index=earliest.index,
            quote=earliest.quote,
            evidence_ids=tuple(earliest.contradiction_evidence_ids),
            downstream_claim_ids=downstream,
            rollback_distance=len(downstream),
        )

    @property
    def localized(self) -> bool:
        return self.earliest_violation is not None
