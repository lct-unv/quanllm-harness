from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HarnessEvent:
    kind: str
    stage: str
    payload: Mapping[str, Any] = field(default_factory=dict)


EventSink = Callable[[HarnessEvent], None]
