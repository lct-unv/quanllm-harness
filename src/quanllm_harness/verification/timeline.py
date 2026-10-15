from __future__ import annotations

from collections.abc import Sequence

from ..contracts import (
    AssertionEntry,
    AssertionStatus,
    AssertionTimeline,
    Claim,
    Evidence,
    Issue,
    IssueOrigin,
    Severity,
)

# Only claims that state a checkable mathematical relation take part in the
# strict evidence timeline. Conceptual claims stay with the semantic verifier.
TIMELINE_KINDS = frozenset({"equation", "derivation"})


def build_assertion_timeline(
    claims: Sequence[Claim],
    evidence: Sequence[Evidence],
    not_checkable: Sequence[str] = (),
    *,
    candidate_digest: str = "",
) -> AssertionTimeline:
    """Linearize assertions into an ordered, evidence-backed timeline.

    Evidence that was collected for a different candidate is ignored: claim IDs
    are re-issued per candidate, so matching by ID alone would let an earlier
    round's contradiction leak into the current, already-repaired answer.
    """

    exempt = set(not_checkable)
    scoped = [
        item
        for item in evidence
        if not item.candidate_digest
        or not candidate_digest
        or item.candidate_digest == candidate_digest
    ]
    evidence = scoped
    entries: list[AssertionEntry] = []
    for index, claim in enumerate(claims, 1):
        related = [item for item in evidence if claim.id in item.claim_ids]
        ok_evidence = [item for item in related if item.ok]
        contradictions = [item for item in ok_evidence if item.supports_claim is False]
        if contradictions:
            status = AssertionStatus.CONTRADICTED
        elif ok_evidence:
            status = AssertionStatus.SUPPORTED
        elif claim.id in exempt:
            status = AssertionStatus.NOT_CHECKABLE
        elif claim.kind in TIMELINE_KINDS:
            status = AssertionStatus.UNVERIFIED
        else:
            status = AssertionStatus.DEFERRED
        entries.append(
            AssertionEntry(
                index=index,
                claim_id=claim.id,
                quote=claim.quote,
                kind=claim.kind,
                status=status,
                evidence_ids=tuple(item.id for item in ok_evidence),
                contradiction_evidence_ids=tuple(item.id for item in contradictions),
            )
        )
    return AssertionTimeline(entries=tuple(entries))


def timeline_issues(timeline: AssertionTimeline) -> list[Issue]:
    """Deterministic findings that localize the earliest incorrect step.

    These findings are produced by Python, not by a model, so they are never
    subject to adjudication and can always trigger a targeted repair.
    """

    violations = timeline.contradicted
    if not violations:
        return []
    target = timeline.rollback_target
    assert target is not None
    issues: list[Issue] = []
    for entry in sorted(violations, key=lambda item: item.index):
        evidence_text = "、".join(entry.contradiction_evidence_ids) or "工具证据"
        if entry.claim_id == target.claim_id:
            downstream = "、".join(target.downstream_claim_ids) or "无"
            problem = (
                f"断言时间线定位：第 {entry.index} 步断言 {entry.claim_id} 与工具证据 "
                f"{evidence_text} 相反，是最早出现的不一致步骤；其下游断言（{downstream}）"
                f"共 {target.rollback_distance} 条一并视为待重算，禁止在更下游做补偿性修改。"
            )
        else:
            problem = (
                f"断言时间线：第 {entry.index} 步断言 {entry.claim_id} 同样与工具证据 "
                f"{evidence_text} 相反，位于最早不一致步骤 {target.claim_id} 的下游。"
            )
        issues.append(
            Issue(
                origin=IssueOrigin.MODEL,
                severity=Severity.MAJOR,
                quote=entry.quote,
                problem=problem,
                correction=(
                    f"回滚到断言 {target.claim_id}（第 {target.index} 步），按工具证据重算该步，"
                    "然后重新核验该断言及其全部下游断言。"
                ),
                evidence_ids=entry.contradiction_evidence_ids,
            )
        )
    return issues
