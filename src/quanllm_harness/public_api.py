from __future__ import annotations

from .config import HarnessSettings
from .contracts import EventSink
from .orchestration import DEFAULT_EXECUTION_GRAPH, ExecutionGraph
from .orchestrator import QuanLLMHarness
from .plugins import PluginManager
from .provider import OpenAIQuanLLMProvider, QuanLLMProvider
from .tools import ToolRegistry


def create_harness(
    settings: HarnessSettings,
    *,
    provider: QuanLLMProvider | None = None,
    tools: ToolRegistry | None = None,
    event_sink: EventSink | None = None,
    graph: ExecutionGraph = DEFAULT_EXECUTION_GRAPH,
    plugin_manager: PluginManager | None = None,
) -> QuanLLMHarness:
    plugin_manager = plugin_manager or PluginManager.discover(settings.plugin_policy)
    provider = provider or (
        plugin_manager.create_provider(settings.plugin_provider, settings)
        if settings.plugin_provider
        else OpenAIQuanLLMProvider(settings)
    )
    return QuanLLMHarness(
        provider=provider,
        settings=settings,
        tools=tools or plugin_manager.build_tool_registry(),
        event_sink=event_sink,
        graph=graph,
        plugin_manager=plugin_manager,
    )
