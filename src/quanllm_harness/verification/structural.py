from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..contracts import Claim, Evidence, Requirement
from ..protocols.claim_extraction import contains_quote


@dataclass(frozen=True)
class StructuralSnapshot:
    claims: Sequence[Claim]
    requirements: Sequence[Requirement]
    evidence: Sequence[Evidence]

    def validate_references(self) -> None:
        claim_ids = [item.id for item in self.claims]
        requirement_ids = [item.id for item in self.requirements]
        evidence_ids = [item.id for item in self.evidence]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("断言 ID 重复")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("要求 ID 重复")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("证据 ID 重复")


__all__ = ["StructuralSnapshot", "contains_quote"]
