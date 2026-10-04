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
from ..plugins import PluginManager
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
        plugin_manager: PluginManager | None = None,
    ):
        settings.validate()
        self.provider = provider
        self.settings = settings
        self.tools = tools or default_tool_registry()
        self.plugin_manager = plugin_manager
        self.external_event_sink = (
            plugin_manager.event_sink(event_sink) if plugin_manager else event_sink
        )
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
            plugin_manager=self.plugin_manager,
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
            try:
                single_primary = SolverAgent(main_runtime).solve(
                    question, require_tool=policy.requires_tools
                )
                return single_primary, None, []
            except Exception as exc:
                failure = f"主求解失败：{type(exc).__name__}: {exc}"
                bus.emit("solver_fallback", "独立求解", reason=failure)
                self._checkpoint(controller, bus, "independent_solver_fallback")
                fallback = IndependentSolverAgent(main_runtime).solve(
                    question, require_tool=policy.requires_tools
                )
                return fallback, None, [failure]

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
        try:
            synthesized = SynthesizerAgent(main_runtime).synthesize(question, primary, independent)
        except Exception as exc:
            failures.append(f"候选综合失败：{type(exc).__name__}: {exc}")
            bus.emit("synthesis_fallback", "候选综合", reason=failures[-1])
            return primary, independent, failures

        # Synthesis may not regress from a source candidate that satisfies more
        # active domain invariants. The generic host owns the monotonicity rule;
        # plugins own the domain-specific checks.
        domain_issues = self.plugin_manager.domain_issues if self.plugin_manager else None
        synthesized_issues = domain_issues(question, synthesized.answer) if domain_issues else ()

        def score(found) -> int:
            return sum(10 if issue.severity is Severity.MAJOR else 1 for issue in found)

        source_scores = [
            (
                score(domain_issues(question, primary.answer) if domain_issues else ()),
                primary,
                independent,
            ),
            (
                score(domain_issues(question, independent.answer) if domain_issues else ()),
                independent,
                primary,
            ),
        ]
        best_score, best_source, other_source = min(source_scores, key=lambda item: item[0])
        synthesized_score = score(synthesized_issues)
        if synthesized_score > best_score:
            bus.emit(
                "synthesis_regression_blocked",
                "候选综合",
                synthesized_issue_score=synthesized_score,
                fallback_issue_score=best_score,
                problems=[issue.problem for issue in synthesized_issues],
            )
            return best_source, other_source, failures
        return synthesized, independent, failures

    def answer(
        self,
        question: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> HarnessResult:
        if self.plugin_manager:
            self.plugin_manager.notify_request_start(question)
        try:
            result = self._answer(question, cancellation=cancellation)
        except BaseException as exc:
            if self.plugin_manager:
                self.plugin_manager.notify_request_end(None, exc)
            raise
        if self.plugin_manager:
            self.plugin_manager.notify_request_end(result, None)
        return result

    def _answer(
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
            fallback_name = ""
            if self.plugin_manager:
                candidate, fallback_name = self.plugin_manager.domain_fallback(question)
            if candidate:
                bus.emit(
                    "domain_solver_fallback",
                    "求解",
                    strategy=fallback_name,
                    reason="模型求解器均未返回，使用领域插件降级结果",
                )

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

        # Active domain plugins can request bounded model rewrites while the host
        # retains the repair budget and final delivery authority.
        if candidate and self.plugin_manager:
            max_extra = min(max(0, self.settings.max_repair_rounds - repair_rounds), 4)
            for _ in range(max_extra):
                instructions, hint = self.plugin_manager.domain_repair_instructions(
                    question, candidate
                )
                if not instructions:
                    break
                bus.emit(
                    "repair_started",
                    "领域结果修复",
                    round=repair_rounds + 1,
                    issue_count=len(instructions),
                )
                try:
                    repaired = RepairAgent(runtime).repair(
                        question,
                        candidate,
                        [
                            {
                                "quote": item.quote,
                                "problem": item.problem,
                                "correction": item.correction,
                                "evidence_ids": list(item.evidence_ids),
                            }
                            for item in instructions
                        ],
                        hint=hint,
                    )
                except Exception as exc:
                    infrastructure_errors.append(f"领域结果修复失败：{type(exc).__name__}: {exc}")
                    break
                candidate = repaired.answer
                repair_rounds += 1

            # No model-written state may follow the deterministic correction.
            # This ordering prevents a late repair from resurrecting an earlier,
            # invalid candidate (the failure mode covered by recovery Case 10).
            candidate, domain_notes = self.plugin_manager.correct_domain_candidate(
                question, candidate
            )
            if domain_notes:
                infrastructure_errors.append("领域确定性回写：" + "；".join(domain_notes))
                bus.emit(
                    "domain_correction_applied",
                    "领域确定性回写",
                    corrections=domain_notes,
                )

            finalized = self.plugin_manager.finalize_domains(question, candidate)
            finalized_changed = finalized.candidate != candidate
            candidate = finalized.candidate
            if finalized_changed:
                message = "领域插件在核验后修正了终稿；本次交付按降级结果处理"
                infrastructure_errors.append(message)
                bus.emit("domain_finalization_applied", "领域终结", reason=message)
            if finalized.issues or finalized.warnings:
                report = replace(
                    report,
                    issues=(*report.issues, *finalized.issues),
                    protocol_warnings=(*report.protocol_warnings, *finalized.warnings),
                )

        if not candidate:
            status = RunStatus.FAILED_WITHOUT_ANSWER
        elif (
            infrastructure_errors
            or report.model_issues
            or report.infrastructure_issues
            or (report.protocol_warnings and not report.verifier_summaries)
        ):
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
