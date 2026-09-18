from __future__ import annotations

from dataclasses import dataclass

from ..config import HarnessSettings
from ..contracts import EventSink, HarnessResult
from ..orchestration import DEFAULT_EXECUTION_GRAPH, CancellationToken
from ..plugins import PluginManager
from ..public_api import create_harness


@dataclass
class HarnessService:
    """Request-scoped gateway shared by CLI, Web UI and REST API.

    A fresh Provider, tool registry and orchestrator are built for each answer so
    evidence IDs, tool caches and event sinks never leak between users.
    """

    settings: HarnessSettings
    plugin_manager: PluginManager | None = None

    def __post_init__(self) -> None:
        if self.plugin_manager is None:
            self.plugin_manager = PluginManager.discover(self.settings.plugin_policy)

    def answer(
        self,
        question: str,
        *,
        event_sink: EventSink | None = None,
        cancellation: CancellationToken | None = None,
    ) -> HarnessResult:
        harness = create_harness(
            self.settings,
            event_sink=event_sink,
            plugin_manager=self.plugin_manager,
        )
        return harness.answer(question, cancellation=cancellation)

    def capabilities(self) -> dict[str, object]:
        assert self.plugin_manager is not None
        return self.plugin_manager.build_tool_registry().capabilities()

    def plugins(self) -> dict[str, object]:
        assert self.plugin_manager is not None
        return self.plugin_manager.doctor()

    @staticmethod
    def execution_graph() -> list[dict[str, object]]:
        return DEFAULT_EXECUTION_GRAPH.describe()

    def close(self) -> None:
        if self.plugin_manager:
            self.plugin_manager.shutdown()
