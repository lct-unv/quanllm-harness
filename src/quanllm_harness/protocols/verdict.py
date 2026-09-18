from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..contracts import Evidence, Issue, IssueOrigin, Severity
from .claim_extraction import contains_quote
from .json_request import StructuredResponseError


def parse_verifier_response(
    data: dict[str, Any],
    *,
    question: str,
    candidate: str,
    evidence: Sequence[Evidence],
) -> tuple[list[Issue], str]:
    raw_issues = data.get("issues")
    if not isinstance(raw_issues, list):
        raise StructuredResponseError("语义核验 issues 不是数组")
    valid_evidence = {item.id for item in evidence}
    issues: list[Issue] = []
    skipped = 0
    for _index, item in enumerate(raw_issues, 1):
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            origin = IssueOrigin(str(item.get("origin")))
            severity = Severity(str(item.get("severity")))
        except ValueError:
            skipped += 1
            continue
        if origin not in {IssueOrigin.MODEL, IssueOrigin.INPUT}:
            skipped += 1
            continue
        quote = str(item.get("quote") or "").strip()
        source = candidate if origin is IssueOrigin.MODEL else question
        if not contains_quote(quote, source):
            skipped += 1
            continue
        problem = str(item.get("problem") or "").strip()
        if not problem:
            skipped += 1
            continue
        raw_ids = item.get("evidence_ids") or []
        if not isinstance(raw_ids, list) or any(value not in valid_evidence for value in raw_ids):
            skipped += 1
            continue
        issues.append(
            Issue(
                origin=origin,
                severity=severity,
                quote=quote,
                problem=problem,
                correction=str(item.get("correction") or "").strip(),
                evidence_ids=tuple(raw_ids),
            )
        )
    # A malformed entry among several valid ones must not discard the whole
    # response; but if every reported issue failed validation, the response is
    # genuinely unusable and the stage should degrade visibly.
    if raw_issues and skipped and not issues:
        raise StructuredResponseError("核验结果中所有问题均未通过引文或证据校验")
    return issues, str(data.get("summary") or "").strip()
