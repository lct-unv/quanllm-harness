from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from ..contracts import EventSink, HarnessEvent


@dataclass
class EventBus:
    """Thread-safe in-memory event stream with an optional live subscriber."""

    sink: EventSink | None = None
    events: list[HarnessEvent] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def emit(self, kind: str, stage: str, **payload: object) -> None:
        self.publish(HarnessEvent(kind=kind, stage=stage, payload=payload))

    def publish(self, event: HarnessEvent) -> None:
        with self._lock:
            self.events.append(event)
        if self.sink:
            self.sink(event)
