# Architecture

## Execution graph

```text
input
  -> structured router
  -> primary solver -----------+
  -> optional isolated solver -+  (parallel, isolated contexts)
  -> optional synthesis
  -> structured claim + requirement extraction
  -> structured tool plan
  -> deterministic evidence
  -> formal/scientific verifier -----+
  -> requirement/pedagogy verifier --+  (independent passes)
  -> one-shot adjudication for single-source findings
  -> targeted repair -> full re-verification (bounded loop)
```

The graph is represented by `ExecutionGraph` and controlled by Python. It can be inspected through the CLI and injected through the public API. Agents do not freely delegate, vote, or decide that a run has passed. `RunController` applies cooperative cancellation and a run-wide deadline between graph stages; each Provider request also has its own stricter transport/stream deadline.

Agent roles are separate modules rather than a free-form Agent group. The legacy `HarnessAgents` facade delegates to those modules and exists only for source compatibility.

## Request modes

QuanLLM-v2.0-qm rejects thinking and JSON mode in the same request. The provider therefore exposes two mutually exclusive profiles:

| Profile | Thinking | JSON mode | Intended stages |
|---|---:|---:|---|
| reasoning | on | off | solver, synthesis, repair |
| structured | off | on | routing, claims, tool plan, verdict |

This invariant lives in the provider and cannot be overridden by an agent.

Structured stages validate semantic contracts after JSON decoding. A failed contract receives exactly the configured number of retries with the validation error as feedback. Truncation (`finish_reason=length`) is a protocol failure, not partial success.

Each stage has both a transport read timeout and a streaming wall-clock deadline. SDK transport retries are disabled so they cannot silently multiply that deadline; protocol retries remain explicit, bounded and visible to the orchestrator.

## Evidence model

Every tool invocation produces an immutable `Evidence` record with its exact arguments, result, limitations and stable ID. Evidence also records linked claim IDs, whether a one-shot semantic preflight verified input fidelity, and whether a comparison-style result explicitly supports or contradicts the claim. `ok` means only that the Python handler completed. A failed tool invocation is never evidence that a claim is true.

Current backends:

- SymPy for scalar algebra, calculus, exact matrices, angular momentum coefficients and a
  structured exact operator backend for canonical, spin and single-mode Fock algebras.
- mpmath for high-precision integration, local root finding and truncation convergence checks.
- QuTiP for finite-dimensional numerical state/operator checks.
- pycommute for bosonic, fermionic and spin operator algebra.
- OpenFermion for many-body normal ordering and operator algebra.

The planner must account for every extracted claim by either producing a valid tool call or an explicit `not_checkable` entry. Before every Solver or verification tool execution, a one-shot `ToolCallReviewerAgent` compares the proposed tool and arguments with the original problem and exact claim. It may approve, uniquely correct, or reject the call; review protocol failure never falls through to execution. Python then validates the tool name, JSON Schema, supported claim kind and backend-specific constraints. Exact duplicate computations are cached, while runtime evidence is upserted by stable ID.

Evidence planning and verifier passes are advisory checkpoints, not delivery gates. If a checkpoint still violates its protocol after the bounded retry, the run records a protocol warning and continues through every independent checkpoint that remains usable. Such a run can deliver an answer but cannot receive `verified` status. Every reported candidate issue is adjudicated exactly once; an adjudication protocol failure remains a warning and can never trigger answer repair.

Scalar SymPy operations deliberately reject Dirac notation and abstract operator expressions.
The built-in operator backend accepts a typed sum-of-monomials AST, preserves multiplication
order, and reports `zero` plus `resolved`. It rejects mixed algebra families and multi-mode Fock
relations instead of silently assuming tensor-product commutativity. pycommute and OpenFermion
are the dedicated default backends for wider many-body algebra; matrices and QuTiP only establish
facts about the supplied finite-dimensional representation.

## Plugin architecture

The complete plugin contract is part of the main package under `quanllm_harness.plugins`; no
independent plugin SDK is built or released. Installed distributions advertise a plugin through
the `quanllm_harness.plugins` entry-point group. The entry-point name is the stable plugin ID and
must equal `PluginManifest.name`.

Discovery first applies the host enable/disable policy by entry-point name. Disabled or unlisted
plugins are reported without importing their Python module. Enabled plugins then pass distribution
digest, API version, harness version, permission, dependency, and manifest validation. Dependencies
start in topological order and stop in reverse order. A plugin receives only permission-gated
registrars for Tool, Verifier, Provider, Event, and Service extensions; cross-plugin services also
require an explicit dependency.

In-process plugins are trusted Python and can return a cleanup callback. Subprocess Tool plugins
use a one-request UTF-8 JSON protocol without a shell, with a deadline, output limit, and reduced
environment. This reduces accidental coupling but is not an OS sandbox. Tool evidence records the
plugin name, version, digest, and execution mode. Plugin verifier results can add issues, warnings,
and summaries but have no API for granting `verified`; only the core verification engine owns that
decision. Extension and lifecycle failures are isolated into status or diagnostics where the core
can safely continue.

The legacy `quanllm_harness.tools` group is policy-controlled and compatibility-only.

## Trust boundaries

- The user question and every candidate answer are data blocks, never system instructions.
- Solvers cannot spawn arbitrary agents or mark their own answers verified.
- Verifiers must quote exact text from either the question or candidate; Python rejects unlocatable findings.
- Tool evidence IDs must exist in the current run before a verifier may cite them.
- Every verifier finding, including agreement between both verifiers, requires exactly one independent adjudication request before it may trigger repair.
- An independent solver answer is advisory evidence, not a majority vote.

## Convergence and delivery

A repaired answer goes through claim extraction and verification again. A run is `verified` only when no model-origin issue remains and the verification protocol completed without warnings.

If a provider, protocol, tool or convergence check fails after an answer exists, the answer may still be returned as `degraded_delivery`; it must not be presented as verified. If no answer exists, the status is `failed_without_answer`.

The repair loop is bounded by both a maximum round count and a duplicate-finding threshold. Every repaired answer is treated as a new candidate: claims are extracted again, tool plans are rebuilt and both verifier passes run again. A previous pass cannot certify changed text.

`ConvergencePolicy` owns recurrence counts and stop decisions. The orchestrator consumes only its `PASS`, `REPAIR` or `STOP` decision, allowing policy tests and future replacement without changing Agent code.

## Concurrency and state

Only the two Solver calls run concurrently, using isolated `AgentRuntime` instances. The Provider, event bus and tool registry are thread-safe at their shared boundaries. Usage and evidence are merged into the main runtime after both futures finish. Later synthesis, verification and repair remain deterministic sequential graph stages.

If `run_directory` is configured, the completed result is serialized to a temporary file, flushed and atomically replaced into its final JSON path. Secrets are excluded from the settings snapshot. Failure to persist changes the status to `degraded_delivery` but does not discard an available answer.

## Interface boundary

`QuanLLMHarness.answer()` is the public synchronous API. `create_harness()` is the normal construction helper. `interfaces.service.HarnessService` is the only boundary used by CLI, Web and REST adapters. It constructs a fresh Harness per request so evidence IDs, exact-call caches and event sinks cannot cross user requests.

The CLI supports one-shot/stdin/JSON operation and an interactive `:again`/`:paste` session. The REST adapter runs synchronous Harness work in worker threads and exposes both JSON and POST/SSE endpoints. Its bounded event queue provides backpressure; browser disconnects set a cooperative cancellation token. Every SSE payload carries numeric and formatted request elapsed time, while the CLI uses the same formatter. The zero-build Web UI consumes that SSE stream, continuously renders a server-calibrated timer, keeps its execution timeline in a fixed-height scroll container, and inserts all model text with `textContent`, never as executable HTML.

Gateway credentials come exclusively from the ignored `APIKEY` file in the server working directory. The gateway endpoint is fixed in the configuration layer and constructed at runtime without storing its complete plaintext form in source or release archives. REST request bodies cannot override the model, gateway URL or API Key. Optional Bearer authentication protects answer endpoints when `QUANLLM_SERVER_TOKEN` is configured. The default bind address is loopback.

## Package boundaries

- `config` contains immutable settings and generation profiles.
- `contracts` contains data-only public records and enums.
- `providers` owns transport behavior and QuanLLM-v2 mode invariants.
- `protocols` parses and validates structured model messages.
- `agents` owns one fixed role per module.
- `orchestration` owns graph state, run control and convergence.
- `tools` owns deterministic capabilities and evidence creation.
- `plugins` owns the integrated extension contract, policy, discovery, lifecycle and subprocess protocol.
- `verification` composes structural, mathematical and semantic checks.
- `events` owns thread-safe event delivery.
- `interfaces` owns CLI, Web and REST presentation plus request isolation.

Top-level `provider.py`, `orchestrator.py` and `verification/engine.py` are compatibility shims and contain no business logic.
