from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

# Issue 001 - verified state lifecycle management.
#
# Candidates move through one explicit, monotonic lifecycle. Only a candidate
# that reached VERIFIED may be delivered or enter synthesis. An INVALID
# candidate is tombstoned by content digest so a later repair, synthesis or
# domain rewrite cannot silently resurrect the very text that was already
# proven wrong (recovery Cases 8 and 10).


class CandidateState(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    INVALID = "invalid"
    ARCHIVED = "archived"


def answer_digest(answer: str) -> str:
    """Content address of an answer, stable under cosmetic whitespace drift."""

    normalized = "\n".join(line.rstrip() for line in (answer or "").strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateRecord:
    id: str
    source: str
    state: CandidateState
    digest: str
    round: int = 0
    reason: str = ""
    superseded_by: str = ""
    issue_fingerprints: Sequence[str] = ()


@dataclass(frozen=True)
class CandidateTransition:
    candidate_id: str
    source: str
    from_state: CandidateState | None
    to_state: CandidateState
    round: int = 0
    reason: str = ""


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """An immutable, content-addressed record of one verified candidate."""

    candidate_id: str
    source: str
    digest: str
    round: int = 0
    revoked: bool = False
    revocation_reason: str = ""


@dataclass(frozen=True)
class CandidateLifecycle:
    """Serializable snapshot of the candidate state machine for one run."""

    records: Sequence[CandidateRecord] = ()
    transitions: Sequence[CandidateTransition] = ()
    checkpoints: Sequence[VerifiedCheckpoint] = ()
    invalid_state_leakage: int = 0
