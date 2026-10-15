from __future__ import annotations

from quanllm_harness.contracts import (
    AssertionStatus,
    Claim,
    Evidence,
    IssueOrigin,
    answer_digest,
)
from quanllm_harness.verification import build_assertion_timeline, timeline_issues


def claim(claim_id: str, quote: str, kind: str = "equation") -> Claim:
    return Claim(claim_id, quote, kind)


def evidence(evidence_id: str, claim_id: str, *, ok: bool = True, supports: bool | None = True):
    return Evidence(
        id=evidence_id,
        tool="matrix_calculate",
        operation="determinant",
        arguments={},
        ok=ok,
        result={"value": 1},
        claim_ids=(claim_id,),
        input_verified=True,
        supports_claim=supports,
    )


def test_timeline_classifies_supported_contradicted_and_unverified_claims():
    claims = [
        claim("C-001", "det(A)=5"),
        claim("C-002", "det(C)=9"),
        claim("C-003", "det(C)=det(A)det(B)"),
        claim("C-004", "因此结论成立", "conclusion"),
    ]
    evidence_items = [
        evidence("E-0001", "C-001", supports=True),
        evidence("E-0002", "C-002", supports=False),
    ]
    timeline = build_assertion_timeline(claims, evidence_items)
    statuses = [entry.status for entry in timeline.entries]
    assert statuses == [
        AssertionStatus.SUPPORTED,
        AssertionStatus.CONTRADICTED,
        AssertionStatus.UNVERIFIED,
        AssertionStatus.DEFERRED,
    ]
    assert timeline.earliest_violation is not None
    assert timeline.earliest_violation.claim_id == "C-002"
    assert timeline.localized is True


def test_rollback_target_reports_downstream_claims_and_distance():
    claims = [
        claim("C-001", "det(A)=5"),
        claim("C-002", "det(C)=9"),
        claim("C-003", "C 的第二行元素为 2"),
        claim("C-004", "det(C)=det(A)det(B)"),
    ]
    timeline = build_assertion_timeline(
        claims,
        [evidence("E-0001", "C-001"), evidence("E-0002", "C-002", supports=False)],
    )
    target = timeline.rollback_target
    assert target is not None
    assert target.claim_id == "C-002"
    assert target.index == 2
    assert target.rollback_distance == 2
    assert target.downstream_claim_ids == ("C-003", "C-004")
    assert target.evidence_ids == ("E-0002",)


def test_timeline_issue_names_earliest_step_and_forbids_downstream_patching():
    claims = [claim("C-001", "det(C)=9"), claim("C-002", "det(C)=15")]
    timeline = build_assertion_timeline(
        claims,
        [evidence("E-0001", "C-001", supports=False), evidence("E-0002", "C-002")],
    )
    issues = timeline_issues(timeline)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.origin is IssueOrigin.MODEL
    assert issue.quote == "det(C)=9"
    assert "最早出现的不一致步骤" in issue.problem
    assert "C-002" in issue.problem
    assert "回滚" in issue.correction
    assert issue.evidence_ids == ("E-0001",)


def test_multiple_contradictions_are_ordered_with_earliest_first():
    claims = [claim("C-001", "第一步"), claim("C-002", "第二步"), claim("C-003", "第三步")]
    timeline = build_assertion_timeline(
        claims,
        [
            evidence("E-0001", "C-001", supports=False),
            evidence("E-0002", "C-003", supports=False),
        ],
    )
    issues = timeline_issues(timeline)
    assert [issue.quote for issue in issues] == ["第一步", "第三步"]
    assert "最早出现的不一致步骤" in issues[0].problem
    assert "位于最早不一致步骤 C-001 的下游" in issues[1].problem


def test_failed_or_inconclusive_evidence_never_becomes_a_contradiction():
    claims = [claim("C-001", "det(C)=9"), claim("C-002", "det(C)=15")]
    evidence_items = [
        Evidence(
            id="E-0001",
            tool="matrix_calculate",
            operation="determinant",
            arguments={},
            ok=False,
            error="RuntimeError: 失败",
            claim_ids=("C-001",),
            supports_claim=False,
        ),
        evidence("E-0002", "C-002", supports=None),
    ]
    timeline = build_assertion_timeline(claims, evidence_items)
    assert timeline.contradicted == ()
    assert timeline.entries[0].status is AssertionStatus.UNVERIFIED
    assert timeline.entries[1].status is AssertionStatus.SUPPORTED
    assert timeline_issues(timeline) == []


def test_not_checkable_math_claim_is_exempt_from_unverified_status():
    claims = [claim("C-001", "det(C)=9")]
    timeline = build_assertion_timeline(claims, [], ["C-001"])
    assert timeline.entries[0].status is AssertionStatus.NOT_CHECKABLE


def test_evidence_from_another_candidate_is_ignored():
    claims = [claim("C-001", "C=15")]
    stale = Evidence(
        id="E-0001",
        tool="fake.equivalence",
        operation="equivalence",
        arguments={},
        ok=True,
        result={"equivalent": False},
        claim_ids=("C-001",),
        supports_claim=False,
        candidate_digest=answer_digest("由计算得 C=9。"),
    )
    fresh_digest = answer_digest("由计算得 C=15。")
    fresh = Evidence(
        id="E-0002",
        tool="fake.equivalence",
        operation="equivalence",
        arguments={},
        ok=True,
        result={"equivalent": True},
        claim_ids=("C-001",),
        supports_claim=True,
        candidate_digest=fresh_digest,
    )
    timeline = build_assertion_timeline(claims, [stale, fresh], candidate_digest=fresh_digest)
    assert timeline.contradicted == ()
    assert timeline.entries[0].status is AssertionStatus.SUPPORTED
    assert timeline_issues(timeline) == []


def test_unscoped_evidence_still_matches_the_current_candidate():
    claims = [claim("C-001", "det(C)=9")]
    timeline = build_assertion_timeline(
        claims,
        [evidence("E-0001", "C-001", supports=False)],
        candidate_digest=answer_digest("任意候选"),
    )
    assert timeline.contradicted[0].claim_id == "C-001"
