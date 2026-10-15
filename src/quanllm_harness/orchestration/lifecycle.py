from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..contracts import (
    CandidateLifecycle,
    CandidateRecord,
    CandidateState,
    CandidateTransition,
    VerifiedCheckpoint,
    answer_digest,
)


@dataclass(frozen=True)
class Admission:
    admitted: bool
    record: CandidateRecord | None = None
    reason: str = ""


@dataclass
class CandidateLedger:
    """Strict, append-only candidate lifecycle owned by the host orchestrator.

    The ledger is deliberately dependency-free so the state machine can be
    unit-tested without a provider, and so the same rules apply to model
    candidates, repaired candidates and domain-plugin rewrites alike.
    """

    invalid_state_leakage: int = 0
    _records: dict[str, CandidateRecord] = field(default_factory=dict, repr=False)
    _history: list[CandidateRecord] = field(default_factory=list, repr=False)
    _transitions: list[CandidateTransition] = field(default_factory=list, repr=False)
    _checkpoints: list[VerifiedCheckpoint] = field(default_factory=list, repr=False)
    _tombstones: dict[str, str] = field(default_factory=dict, repr=False)
    _answers: dict[str, str] = field(default_factory=dict, repr=False)
    _counter: int = 0
    _current: str = ""

    # -- admission ---------------------------------------------------------
    def admit(self, source: str, answer: str, *, round_index: int = 0) -> Admission:
        """Admit a candidate unless its content was already proven invalid."""

        digest = answer_digest(answer)
        self._answers[digest] = answer
        tombstone = self._tombstones.get(digest)
        if tombstone:
            self.invalid_state_leakage += 1
            return Admission(False, None, f"候选内容命中已失效状态：{tombstone}")
        known = self._by_digest(digest)
        if known is not None:
            if known.state is CandidateState.INVALID:
                self.invalid_state_leakage += 1
                return Admission(False, None, f"候选内容命中已失效状态：{known.reason}")
            if known.state in {CandidateState.PROPOSED, CandidateState.VERIFIED}:
                self._current = known.id
                return Admission(True, known, "候选已登记，复用现有状态")
        self._counter += 1
        record = CandidateRecord(
            id=f"K-{self._counter:03d}",
            source=source,
            state=CandidateState.PROPOSED,
            digest=digest,
            round=round_index,
        )
        self._records[record.id] = record
        self._history.append(record)
        self._transitions.append(
            CandidateTransition(record.id, source, None, CandidateState.PROPOSED, round_index)
        )
        self._current = record.id
        return Admission(True, record, "")

    # -- transitions -------------------------------------------------------
    def mark_verified(self, candidate_id: str, *, reason: str = "") -> CandidateRecord:
        existing = self._records.get(candidate_id)
        if existing is None:
            raise KeyError(f"候选不存在：{candidate_id}")
        if existing.state is CandidateState.VERIFIED:
            return existing
        record = self._require(candidate_id, {CandidateState.PROPOSED})
        updated = self._transition(record, CandidateState.VERIFIED, reason)
        self._checkpoints.append(
            VerifiedCheckpoint(
                candidate_id=updated.id,
                source=updated.source,
                digest=updated.digest,
                round=updated.round,
            )
        )
        return updated

    def mark_invalid(
        self,
        candidate_id: str,
        *,
        reason: str,
        issue_fingerprints: Sequence[str] = (),
    ) -> CandidateRecord:
        existing = self._records.get(candidate_id)
        if existing is None:
            raise KeyError(f"候选不存在：{candidate_id}")
        if existing.state is CandidateState.INVALID:
            return existing
        record = self._require(candidate_id, {CandidateState.PROPOSED, CandidateState.VERIFIED})
        updated = self._transition(
            record,
            CandidateState.INVALID,
            reason,
            issue_fingerprints=tuple(issue_fingerprints),
        )
        self._tombstones.setdefault(updated.digest, reason or "已被核验判为失效")
        self._revoke_checkpoints(updated.digest, reason)
        return updated

    def archive(self, candidate_id: str, *, superseded_by: str = "") -> CandidateRecord:
        record = self._require(candidate_id, {CandidateState.PROPOSED, CandidateState.VERIFIED})
        return self._transition(
            record,
            CandidateState.ARCHIVED,
            f"被 {superseded_by or '新候选'} 取代",
        )

    # -- queries -----------------------------------------------------------
    @property
    def current(self) -> CandidateRecord | None:
        return self._records.get(self._current)

    def record(self, candidate_id: str) -> CandidateRecord | None:
        return self._records.get(candidate_id)

    def is_tombstoned(self, answer: str) -> bool:
        return answer_digest(answer) in self._tombstones

    def synthesis_eligible(self, candidate_id: str) -> bool:
        record = self._records.get(candidate_id)
        return record is not None and record.state in {
            CandidateState.PROPOSED,
            CandidateState.VERIFIED,
        }

    def gate(self, candidate_id: str) -> Admission:
        """Consumption gate: only live candidates may feed synthesis/delivery."""

        record = self._records.get(candidate_id)
        if record is None:
            self.invalid_state_leakage += 1
            return Admission(False, None, "候选未登记")
        if record.state not in {CandidateState.PROPOSED, CandidateState.VERIFIED}:
            self.invalid_state_leakage += 1
            return Admission(False, record, f"候选状态 {record.state.value} 不可用于合成或交付")
        return Admission(True, record, "")

    def latest_checkpoint(self) -> VerifiedCheckpoint | None:
        for checkpoint in reversed(self._checkpoints):
            if not checkpoint.revoked:
                return checkpoint
        return None

    def verified_answer(self) -> str:
        checkpoint = self.latest_checkpoint()
        if checkpoint is None:
            return ""
        return self._answers.get(checkpoint.digest, "")

    def delivery_guard(self, answer: str) -> tuple[str, str]:
        """Return a deliverable answer plus the reason when a rewrite was applied."""

        if not self.is_tombstoned(answer):
            return answer, ""
        fallback = self.verified_answer()
        if fallback:
            return fallback, "交付前命中已失效候选，回退到最近一次已核验状态"
        return answer, "交付内容命中已失效候选，且没有可回退的已核验状态"

    def snapshot(self) -> CandidateLifecycle:
        return CandidateLifecycle(
            records=tuple(self._records.values()),
            transitions=tuple(self._transitions),
            checkpoints=tuple(self._checkpoints),
            invalid_state_leakage=self.invalid_state_leakage,
        )

    # -- internals ---------------------------------------------------------
    def _by_digest(self, digest: str) -> CandidateRecord | None:
        for record in self._records.values():
            if record.digest == digest:
                return record
        return None

    def _require(self, candidate_id: str, allowed: set[CandidateState]) -> CandidateRecord:
        record = self._records.get(candidate_id)
        if record is None:
            raise KeyError(f"候选不存在：{candidate_id}")
        if record.state not in allowed:
            raise ValueError(f"候选 {candidate_id} 处于 {record.state.value}，不允许该状态迁移")
        return record

    def _transition(
        self,
        record: CandidateRecord,
        state: CandidateState,
        reason: str,
        *,
        issue_fingerprints: Sequence[str] = (),
    ) -> CandidateRecord:
        updated = CandidateRecord(
            id=record.id,
            source=record.source,
            state=state,
            digest=record.digest,
            round=record.round,
            reason=reason,
            superseded_by=record.superseded_by,
            issue_fingerprints=tuple(issue_fingerprints),
        )
        self._records[record.id] = updated
        self._history.append(updated)
        self._transitions.append(
            CandidateTransition(record.id, record.source, record.state, state, record.round, reason)
        )
        self._current = (
            "" if state in {CandidateState.INVALID, CandidateState.ARCHIVED} else record.id
        )
        return updated

    def _revoke_checkpoints(self, digest: str, reason: str) -> None:
        for index, checkpoint in enumerate(self._checkpoints):
            if checkpoint.digest != digest or checkpoint.revoked:
                continue
            self._checkpoints[index] = VerifiedCheckpoint(
                candidate_id=checkpoint.candidate_id,
                source=checkpoint.source,
                digest=checkpoint.digest,
                round=checkpoint.round,
                revoked=True,
                revocation_reason=reason,
            )
