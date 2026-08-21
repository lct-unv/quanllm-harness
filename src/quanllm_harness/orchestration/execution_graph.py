from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from threading import Event
from time import monotonic


class RunCancelled(RuntimeError):
    pass


class RunDeadlineExceeded(TimeoutError):
    pass


class CancellationToken:
    """Thread-safe cooperative cancellation token supplied by API clients."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RunCancelled("运行已由调用方取消")


@dataclass(frozen=True)
class GraphNode:
    name: str
    dependencies: tuple[str, ...] = ()
    optional: bool = False
    repeatable: bool = False


class ExecutionGraph:
    """Declarative stage graph used for introspection and dependency validation."""

    def __init__(self, nodes: Iterable[GraphNode]):
        self.nodes = {node.name: node for node in nodes}
        if not self.nodes:
            raise ValueError("执行图不能为空")
        for node in self.nodes.values():
            missing = set(node.dependencies) - self.nodes.keys()
            if missing:
                raise ValueError(f"节点 {node.name} 引用了未知依赖：{sorted(missing)}")
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"执行图存在环：{name}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in self.nodes[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for name in self.nodes:
            visit(name)

    def ready(self, completed: set[str]) -> tuple[str, ...]:
        return tuple(
            name
            for name, node in self.nodes.items()
            if name not in completed and set(node.dependencies) <= completed
        )

    def describe(self) -> list[dict[str, object]]:
        return [
            {
                "name": node.name,
                "dependencies": list(node.dependencies),
                "optional": node.optional,
                "repeatable": node.repeatable,
            }
            for node in self.nodes.values()
        ]


DEFAULT_EXECUTION_GRAPH = ExecutionGraph(
    (
        GraphNode("route"),
        GraphNode("primary_solver", ("route",)),
        GraphNode("independent_solver", ("route",), optional=True),
        GraphNode(
            "synthesis",
            ("primary_solver", "independent_solver"),
            optional=True,
        ),
        GraphNode("claim_extraction", ("primary_solver",)),
        GraphNode("tool_evidence", ("claim_extraction",), optional=True),
        GraphNode("semantic_verification", ("claim_extraction",)),
        GraphNode("repair", ("semantic_verification",), optional=True, repeatable=True),
        GraphNode("persistence", ("semantic_verification",), optional=True),
    )
)


@dataclass
class RunController:
    total_timeout_seconds: float
    cancellation: CancellationToken
    clock: Callable[[], float] = monotonic

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    @property
    def elapsed_seconds(self) -> float:
        return self.clock() - self.started_at

    def checkpoint(self, stage: str) -> None:
        self.cancellation.raise_if_cancelled()
        if self.elapsed_seconds > self.total_timeout_seconds:
            raise RunDeadlineExceeded(
                f"运行在 {stage} 前超过总时长上限 {self.total_timeout_seconds:g} 秒"
            )
