from __future__ import annotations

from dataclasses import dataclass

from ..config import HarnessSettings
from ..contracts import EventSink, HarnessResult
from ..orchestration import DEFAULT_EXECUTION_GRAPH, CancellationToken
from ..public_api import create_harness
from ..tools import default_tool_registry


@dataclass(frozen=True)
class HarnessService:
    """Request-scoped gateway shared by CLI, Web UI and REST API.

    A fresh Provider, tool registry and orchestrator are built for each answer so
    evidence IDs, tool caches and event sinks never leak between users.
    """

    settings: HarnessSettings

    def answer(
        self,
        question: str,
        *,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> HarnessResult:
        harness = create_harness(self.settings, event_sink=event_sink)
        return harness.answer(question, cancellation=cancellation)

    @staticmethod
    def capabilities() -> dict[str, object]:
        return default_tool_registry().capabilities()

    @staticmethod
    def execution_graph() -> list[dict[str, object]]:
        return DEFAULT_EXECUTION_GRAPH.describe()
