from __future__ import annotations

import pytest

from quanllm_harness.contracts import CandidateState, answer_digest
from quanllm_harness.orchestration.lifecycle import CandidateLedger


def test_proposed_candidate_becomes_verified_checkpoint():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "答案 A")
    assert admission.admitted and admission.record is not None
    assert admission.record.state is CandidateState.PROPOSED

    record = ledger.mark_verified(admission.record.id, reason="核验通过")
    assert record.state is CandidateState.VERIFIED
    assert ledger.latest_checkpoint() is not None
    assert ledger.verified_answer() == "答案 A"
    assert ledger.synthesis_eligible(record.id)


def test_invalid_candidate_is_tombstoned_and_cannot_be_readmitted():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "答案 A")
    assert admission.record is not None
    ledger.mark_invalid(admission.record.id, reason="断言错误")

    repeated = ledger.admit("定向修复", "答案 A")
    assert repeated.admitted is False
    assert "已失效" in repeated.reason
    assert ledger.invalid_state_leakage == 1
    assert ledger.is_tombstoned("答案 A")


def test_digest_ignores_cosmetic_whitespace_drift():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "答案 A\n")
    assert admission.record is not None
    ledger.mark_invalid(admission.record.id, reason="断言错误")

    drifted = ledger.admit("定向修复", "答案 A   \n\n")
    assert drifted.admitted is False
    assert answer_digest("答案 A\n") == answer_digest("答案 A   \n\n")


def test_invalidating_verified_candidate_revokes_checkpoint():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "答案 A")
    assert admission.record is not None
    ledger.mark_verified(admission.record.id)
    assert ledger.latest_checkpoint() is not None

    ledger.mark_invalid(admission.record.id, reason="后续核验发现错误")
    assert ledger.latest_checkpoint() is None
    assert ledger.verified_answer() == ""
    snapshot = ledger.snapshot()
    assert any(checkpoint.revoked for checkpoint in snapshot.checkpoints)


def test_gate_blocks_invalid_and_archived_candidates_and_counts_leakage():
    ledger = CandidateLedger()
    invalid = ledger.admit("求解·主求解", "答案 A")
    assert invalid.record is not None
    ledger.mark_invalid(invalid.record.id, reason="断言错误")
    blocked = ledger.gate(invalid.record.id)
    assert blocked.admitted is False
    assert ledger.invalid_state_leakage == 1

    archived = ledger.admit("求解·独立求解", "答案 B")
    assert archived.record is not None
    ledger.archive(archived.record.id, superseded_by="K-003")
    assert ledger.gate(archived.record.id).admitted is False
    assert ledger.synthesis_eligible(archived.record.id) is False
    assert ledger.invalid_state_leakage == 2


def test_mark_verified_is_idempotent():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "答案 A")
    assert admission.record is not None
    first = ledger.mark_verified(admission.record.id)
    second = ledger.mark_verified(admission.record.id)
    assert first == second
    assert len(ledger.snapshot().checkpoints) == 1


def test_transitions_are_append_only_and_reject_illegal_moves():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "答案 A")
    assert admission.record is not None
    ledger.mark_invalid(admission.record.id, reason="断言错误")
    with pytest.raises(ValueError):
        ledger.mark_verified(admission.record.id)
    kinds = [(item.from_state, item.to_state) for item in ledger.snapshot().transitions]
    assert kinds == [
        (None, CandidateState.PROPOSED),
        (CandidateState.PROPOSED, CandidateState.INVALID),
    ]


def test_delivery_guard_falls_back_to_verified_checkpoint():
    ledger = CandidateLedger()
    verified = ledger.admit("求解·主求解", "已核验答案")
    assert verified.record is not None
    ledger.mark_verified(verified.record.id)

    stale = ledger.admit("候选综合", "失效答案")
    assert stale.record is not None
    ledger.mark_invalid(stale.record.id, reason="综合引入失效状态")

    answer, reason = ledger.delivery_guard("失效答案")
    assert answer == "已核验答案"
    assert "回退" in reason


def test_delivery_guard_without_checkpoint_reports_no_fallback():
    ledger = CandidateLedger()
    admission = ledger.admit("求解·主求解", "失效答案")
    assert admission.record is not None
    ledger.mark_invalid(admission.record.id, reason="断言错误")

    answer, reason = ledger.delivery_guard("失效答案")
    assert answer == "失效答案"
    assert "没有可回退" in reason
