from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from ..agents import (
    AgentRuntime,
    IndependentSolverAgent,
    RepairAgent,
    RouterAgent,
    SolverAgent,
    SynthesizerAgent,
)
from ..config import HarnessSettings
from ..contracts import (
    Candidate,
    EventSink,
    HarnessResult,
    Issue,
    IssueOrigin,
    RequestPolicy,
    RunStatus,
    Severity,
    VerificationReport,
)
from ..events import EventBus
from ..persistence import save_run
from ..providers import QuanLLMProvider
from ..tools import ToolRegistry, default_tool_registry
from ..verification import VerificationEngine
from .convergence import ConvergenceAction, ConvergencePolicy
from .execution_graph import (
    DEFAULT_EXECUTION_GRAPH,
    CancellationToken,
    ExecutionGraph,
    RunCancelled,
    RunController,
)


class QuanLLMHarness:
    def __init__(
        self,
        *,
        provider: QuanLLMProvider,
        settings: HarnessSettings,
        tools: ToolRegistry | None = None,
        event_sink: EventSink | None = None,
        graph: ExecutionGraph = DEFAULT_EXECUTION_GRAPH,
    ):
        settings.validate()
        self.provider = provider
        self.settings = settings
        self.tools = tools or default_tool_registry()
        self.external_event_sink = event_sink
        self.graph = graph

    @staticmethod
    def _infrastructure_issue(problem: str) -> Issue:
        return Issue(IssueOrigin.INFRASTRUCTURE, Severity.MAJOR, "", problem)

    def _runtime(self, bus: EventBus) -> AgentRuntime:
        return AgentRuntime(
            provider=self.provider,
            tools=self.tools,
            settings=self.settings,
            event_sink=bus.publish,
        )

    @staticmethod
    def _checkpoint(controller: RunController, bus: EventBus, stage: str) -> None:
        controller.checkpoint(stage)
        bus.emit("stage_checkpoint", stage, elapsed_seconds=controller.elapsed_seconds)

    def _solve_candidates(
        self,
        question: str,
        policy: RequestPolicy,
        main_runtime: AgentRuntime,
        bus: EventBus,
        controller: RunController,
    ) -> tuple[Candidate, Candidate | None, list[str]]:
        use_independent = policy.requires_independent_solver or (
            policy.depth == "deep" and self.settings.enable_independent_solver_for_deep
        )
        self._checkpoint(controller, bus, "primary_solver")
        if not use_independent:
            single_primary = SolverAgent(main_runtime).solve(
                question, require_tool=policy.requires_tools
            )
            return single_primary, None, []

        primary_runtime = self._runtime(bus)
        independent_runtime = self._runtime(bus)
        failures: list[str] = []
        primary: Candidate | None = None
        independent: Candidate | None = None
        self._checkpoint(controller, bus, "independent_solver")

        if self.settings.parallel_solvers:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="quanllm-solver") as pool:
                primary_future = pool.submit(
                    SolverAgent(primary_runtime).solve,
                    question,
                    require_tool=policy.requires_tools,
                )
                independent_future = pool.submit(
                    IndependentSolverAgent(independent_runtime).solve,
                    question,
                    require_tool=policy.requires_tools,
                )
                for label, future in (("主求解", primary_future), ("独立求解", independent_future)):
                    try:
                        value = future.result()
                    except Exception as exc:
                        failures.append(f"{label}失败：{type(exc).__name__}: {exc}")
                    else:
                        if label == "主求解":
                            primary = value
                        else:
                            independent = value
        else:
            try:
                primary = SolverAgent(primary_runtime).solve(
                    question, require_tool=policy.requires_tools
                )
            except Exception as exc:
                failures.append(f"主求解失败：{type(exc).__name__}: {exc}")
            try:
                independent = IndependentSolverAgent(independent_runtime).solve(
                    question, require_tool=policy.requires_tools
                )
            except Exception as exc:
                failures.append(f"独立求解失败：{type(exc).__name__}: {exc}")

        main_runtime.absorb(primary_runtime)
        main_runtime.absorb(independent_runtime)
        if primary is None and independent is None:
            raise RuntimeError("；".join(failures) or "两个求解 Agent 均未返回答案")
        if primary is None:
            assert independent is not None
            return independent, None, failures
        if independent is None:
            return primary, None, failures
        self._checkpoint(controller, bus, "synthesis")
        synthesized = SynthesizerAgent(main_runtime).synthesize(question, primary, independent)
        return synthesized, independent, failures

    def answer(
        self,
        question: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> HarnessResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question 必须是非空字符串")
        question = question.strip()
        cancellation = cancellation or CancellationToken()
        controller = RunController(self.settings.total_timeout_seconds, cancellation)
        bus = EventBus(self.external_event_sink)
        runtime = self._runtime(bus)
        verifier = VerificationEngine(runtime)
        bus.emit(
            "run_started",
            "Harness",
            model=self.settings.model,
            graph=self.graph.describe(),
        )
        infrastructure_errors: list[str] = []

        try:
            self._checkpoint(controller, bus, "route")
            policy = RouterAgent(runtime).route(question)
        except RunCancelled:
            raise
        except Exception as exc:
            policy = RequestPolicy(
                depth="deep",
                requires_tools=True,
                requires_independent_solver=True,
                reason="路由协议不可用，采用高保障降级路径",
            )
            infrastructure_errors.append(f"任务路由失败：{type(exc).__name__}: {exc}")
            bus.emit("degraded", "任务路由", reason=infrastructure_errors[-1])

        candidate = ""
        reference = ""
        report = VerificationReport()
        repair_rounds = 0
        try:
            selected, independent, solver_failures = self._solve_candidates(
                question, policy, runtime, bus, controller
            )
            candidate = selected.answer
            reference = independent.answer if independent else ""
            infrastructure_errors.extend(solver_failures)
        except RunCancelled:
            raise
        except Exception as exc:
            infrastructure_errors.append(f"求解阶段失败：{type(exc).__name__}: {exc}")
            bus.emit("degraded", "求解", reason=infrastructure_errors[-1])

        convergence = ConvergencePolicy(
            self.settings.max_repair_rounds,
            self.settings.duplicate_issue_limit,
        )
        while candidate:
            try:
                self._checkpoint(controller, bus, "semantic_verification")
                report = verifier.verify(question, candidate, reference=reference)
            except RunCancelled:
                raise
            except Exception as exc:
                message = f"核验协议失败：{type(exc).__name__}: {exc}"
                infrastructure_errors.append(message)
                bus.emit("degraded", "核验", reason=message)
                report = VerificationReport(
                    evidence=tuple(runtime.evidence),
                    issues=(self._infrastructure_issue(message),),
                    protocol_warnings=(message,),
                )
                break
            decision = convergence.evaluate(
                report.model_issues,
                completed_repair_rounds=repair_rounds,
            )
            if decision.action is ConvergenceAction.PASS:
                break
            if decision.action is ConvergenceAction.STOP:
                infrastructure_errors.append(decision.reason)
                break
            bus.emit(
                "repair_started",
                "定向修复",
                round=repair_rounds + 1,
                issue_count=len(report.model_issues),
            )
            try:
                self._checkpoint(controller, bus, "repair")
                repaired = RepairAgent(runtime).repair(
                    question,
                    candidate,
                    [
                        {
                            "quote": issue.quote,
                            "problem": issue.problem,
                            "correction": issue.correction,
                            "evidence_ids": list(issue.evidence_ids),
                        }
                        for issue in report.model_issues
                    ],
                )
            except RunCancelled:
                raise
            except Exception as exc:
                infrastructure_errors.append(f"修复失败：{type(exc).__name__}: {exc}")
                break
            candidate = repaired.answer
            repair_rounds += 1

        if not candidate:
            status = RunStatus.FAILED_WITHOUT_ANSWER
        elif infrastructure_errors or report.protocol_warnings or report.model_issues:
            status = RunStatus.DEGRADED_DELIVERY
        elif report.input_issues:
            status = RunStatus.VERIFIED_WITH_INPUT_AMBIGUITY
        else:
            status = RunStatus.VERIFIED
        if infrastructure_errors:
            issues = list(report.issues)
            known_infrastructure = {
                issue.problem for issue in issues if issue.origin is IssueOrigin.INFRASTRUCTURE
            }
            issues.extend(
                self._infrastructure_issue(value)
                for value in dict.fromkeys(infrastructure_errors)
                if value not in known_infrastructure
            )
            report = replace(
                report,
                issues=tuple(issues),
                protocol_warnings=tuple(
                    dict.fromkeys((*report.protocol_warnings, *infrastructure_errors))
                ),
            )
        bus.emit(
            "run_finished",
            "Harness",
            status=status.value,
            repair_rounds=repair_rounds,
            elapsed_seconds=controller.elapsed_seconds,
        )
        result = HarnessResult(
            status=status,
            answer=candidate,
            policy=policy,
            verification=report,
            events=tuple(bus.events),
            usage=runtime.usage,
            repair_rounds=repair_rounds,
        )
        if self.settings.run_directory:
            try:
                self._checkpoint(controller, bus, "persistence")
                result = save_run(
                    self.settings.run_directory,
                    question=question,
                    result=result,
                    settings=self.settings,
                )
            except RunCancelled:
                raise
            except Exception as exc:
                message = f"运行记录写入失败：{type(exc).__name__}: {exc}"
                bus.emit("degraded", "运行记录", reason=message)
                report = replace(
                    result.verification,
                    issues=(*result.verification.issues, self._infrastructure_issue(message)),
                    protocol_warnings=(*result.verification.protocol_warnings, message),
                )
                result = replace(
                    result,
                    status=RunStatus.DEGRADED_DELIVERY,
                    verification=report,
                    events=tuple(bus.events),
                )
        return result
