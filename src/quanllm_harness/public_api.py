from __future__ import annotations

from .config import HarnessSettings
from .contracts import EventSink
from .orchestration import DEFAULT_EXECUTION_GRAPH, ExecutionGraph
from .orchestrator import QuanLLMHarness
from .provider import OpenAIQuanLLMProvider, QuanLLMProvider
from .tools import ToolRegistry, default_tool_registry


def create_harness(
    settings: HarnessSettings,
    *,
    provider: QuanLLMProvider | None = None,
    tools: ToolRegistry | None = None,
    event_sink: EventSink | None = None,
    graph: ExecutionGraph = DEFAULT_EXECUTION_GRAPH,
) -> QuanLLMHarness:
    provider = provider or OpenAIQuanLLMProvider(settings)
    return QuanLLMHarness(
        provider=provider,
        settings=settings,
        tools=tools or default_tool_registry(),
        event_sink=event_sink,
        graph=graph,
    )
