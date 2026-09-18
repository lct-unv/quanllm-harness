from importlib.metadata import PackageNotFoundError, version

from .config import HarnessSettings
from .contracts import HarnessEvent, HarnessResult, RunStatus
from .orchestration import CancellationToken, ExecutionGraph
from .orchestrator import QuanLLMHarness
from .plugins import (
    PLUGIN_API_VERSION,
    PluginContext,
    PluginManager,
    PluginManifest,
    PluginPolicy,
    PluginVerificationContext,
    PluginVerificationResult,
    SubprocessToolSpec,
)
from .provider import OpenAIQuanLLMProvider, QuanLLMProvider
from .public_api import create_harness

try:
    __version__ = version("quanllm-harness")
except PackageNotFoundError:
    __version__ = "0.1.3"

__all__ = [
    "HarnessEvent",
    "HarnessResult",
    "HarnessSettings",
    "OpenAIQuanLLMProvider",
    "QuanLLMHarness",
    "QuanLLMProvider",
    "RunStatus",
    "create_harness",
    "CancellationToken",
    "ExecutionGraph",
    "__version__",
    "PLUGIN_API_VERSION",
    "PluginContext",
    "PluginManager",
    "PluginManifest",
    "PluginPolicy",
    "PluginVerificationContext",
    "PluginVerificationResult",
    "SubprocessToolSpec",
]
