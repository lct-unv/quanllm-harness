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
    for index, item in enumerate(raw_issues, 1):
        if not isinstance(item, dict):
            raise StructuredResponseError(f"第 {index} 个问题不是对象")
        try:
            origin = IssueOrigin(str(item.get("origin")))
            severity = Severity(str(item.get("severity")))
        except ValueError as exc:
            raise StructuredResponseError(f"第 {index} 个问题枚举非法") from exc
        if origin not in {IssueOrigin.MODEL, IssueOrigin.INPUT}:
            raise StructuredResponseError(f"第 {index} 个问题 origin 非法")
        quote = str(item.get("quote") or "").strip()
        source = candidate if origin is IssueOrigin.MODEL else question
        if not contains_quote(quote, source):
            raise StructuredResponseError(f"第 {index} 个问题引文无法定位")
        problem = str(item.get("problem") or "").strip()
        if not problem:
            raise StructuredResponseError(f"第 {index} 个问题缺少依据")
        raw_ids = item.get("evidence_ids") or []
        if not isinstance(raw_ids, list) or any(value not in valid_evidence for value in raw_ids):
            raise StructuredResponseError(f"第 {index} 个问题引用未知证据")
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
    return issues, str(data.get("summary") or "").strip()
