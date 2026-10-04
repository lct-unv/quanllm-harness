# Changelog

## Unreleased

## 0.1.5 - 2026-10-05

- Split quantum-mechanics teaching behavior out of the generic core into the bundled, opt-in
  `quanllm-qm-teaching` domain plugin. The core now owns orchestration and generic symbolic,
  matrix, and numerical capabilities, while the plugin owns domain prompts, deterministic
  verification, recovery policy, standard results, and quantum-specific tools.
- Added prompt contributors and domain strategies to the stable plugin surface. Domain plugins can
  now contribute stage-aware system-prompt fragments, candidate gates, bounded repair instructions,
  deterministic fallbacks, atomic candidate rewrites, and delivery-time finalization without
  taking ownership of the host status machine.
- Bundled the official plugin in the Harness wheel but left it disabled by default. Source checkouts
  and installed wheels discover it consistently, and explicit activation registers ten namespaced
  tools plus its prompt, domain, and verifier extensions.
- Moved QuTiP, pycommute, and OpenFermion to the `qm-teaching` optional dependency group so the
  default installation remains domain-neutral. The supported Windows installer opts into that
  extra explicitly.
- Prevented plugin verifier failures from reusing a successful semantic status. Infrastructure
  failures now force honest degraded delivery, while exact domain fallbacks can still preserve a
  correct answer.
- Added atomic recovery for coupled complex-projection and ordered unitary-chain results. A repair
  replaces the complete dependent result set and re-verifies it, preventing candidate synthesis
  from restoring invalidated matrices, states, norms, or determinants.
- Re-ran all ten reasoning/recovery reproductions with the plugin active. Every delivered answer
  passed the independent deterministic checks; Cases 6, 7, and 10 retained degraded telemetry for
  real provider or tool timeouts instead of reporting false verification.

### Contributor

- Hxttt1 <3034557373@qq.com>
- [lct-unv](https://github.com/lct-unv)

## 0.1.4 - 2026-10-04

- Hardened the five-case reasoning/recovery path: structured claim extraction now salvages valid
  fields and uses source-backed fallbacks, optional verifier failures remain telemetry when another
  semantic pass succeeds, malformed tool arguments are repaired or regenerated tool-free, repeated
  tool failures have a per-tool budget, and displayed matrix products receive a deterministic
  transcription check before delivery.
- Added round-two recovery monotonicity guards: synthesis failure falls back to the latest solver
  candidate, synthesized candidates cannot replace a source with a higher deterministic-correctness
  score, and deterministic checks now localize inverse row/column verification errors, validate
  named matrix products such as `W=VU`, and enforce explicit per-eigenvalue vector checks.
- Added exact recovery gates for scaled/plain matrices, complex projectors and norm ratios,
  inverse matrices, eigenspectra and eigenpairs, ordered state transformations, and final quantum
  states. Locally provable corrections are delivered with explicit degraded telemetry when model
  repair or synthesis fails to converge.
- Added a fallback independent solver and an exact local projection fallback so provider timeouts
  no longer discard a verified result, while preserving timeout and protocol failures in the run
  record.
- Added the ten-case reasoning/recovery reproduction runner and regression coverage for every
  failure mode found during real-gateway validation.
- Removed an intermittent REST integration shutdown deadlock by isolating synchronous answer
  execution from the event loop's default executor.

### Contributor

- Hxttt1 <3034557373@qq.com>

## 0.1.3 - 2026-09-18

- Added the complete four-stage plugin platform directly to `quanllm-harness`: stable manifest and
  context APIs, default-deny discovery and dependency lifecycle, permission/digest policy and
  bounded subprocess tools, plus Tool/Verifier/Provider/Event/Service extensions, provenance,
  CLI/REST diagnostics, documentation, examples, and tests. No separate plugin SDK is published.
- Strengthened multi-part quantum-answer verification with more tolerant but traceable claim
  extraction, stricter tool-call reconstruction, evidence coverage checks, and deterministic
  checks for common Pauli, eigenpair, boundary-condition, and density-matrix results.
- Changed REST answer endpoints to fail closed when no server token is configured, added an
  explicit insecure opt-in for trusted local deployments, and used constant-time token comparison.
- Added a Windows `install.ps1` that downloads the official pycommute 1.0.0 source with pip,
  verifies its SHA-256 digest, applies the required MSVC compatibility patches in a temporary
  directory, installs QuanLLM Harness, and verifies fermionic operator algebra.
- Documented the dedicated Windows installation path in both Chinese and English.

### Contributor

- Hxttt1 <3034557373@qq.com>

## 0.1.2 - 2026-08-22

- Reworked the project README into complete, content-equivalent Chinese and English sections for
  both GitHub and PyPI.
- Replaced the release-version badge with an exact static version badge so a newly published
  version cannot temporarily display stale PyPI cache data.

### Contributors

- [Hxttt1](https://github.com/Hxttt1)
- [fanfan32123](https://github.com/fanfan32123)

## 0.1.1 - 2026-08-22

- Removed the complete managed gateway endpoint from source, documentation, tests and release
  archives while preserving a fixed, non-overridable runtime destination.
- Added wheel and sdist content scanning that rejects any regression containing the complete
  endpoint in plaintext.
- Refreshed PyPI badge cache keys for the new release.

### Contributor

- [fanfan32123](https://github.com/fanfan32123)

## 0.1.0 - 2026-08-22

- Added a controlled multi-Agent DAG with parallel isolated solvers.
- Enforced mutually exclusive reasoning and structured Provider modes for QuanLLM-v2.0-qm.
- Added claim/requirement extraction, typed tool planning, dual verification, issue adjudication and bounded full re-verification.
- Added SymPy, QuTiP, pycommute and OpenFermion tool adapters with capability discovery.
- Added typed events, usage aggregation, atomic run records, public Python API and one-shot CLI.
- Added offline unit, concurrency, protocol, persistence, packaging and default-backend tests.
- Split configuration, contracts, providers, Agents, protocols, orchestration, verification,
  events and quantum backends into stable package boundaries with legacy import shims.
- Added a declarative execution graph, cooperative cancellation, a run-wide deadline and an
  independent convergence policy.
- Added high-precision numerical integration/root finding, truncation convergence checks and
  trusted Python entry-point tool plugins.
- Added layered unit/integration/regression fixtures, streaming example, type marker, CI quality
  gates, sdist/wheel verification and a Trusted Publishing release workflow.
- Added an isolated shared interface service, interactive QuanLLM CLI, FastAPI REST API, POST/SSE
  streaming endpoint, disconnect cancellation and a responsive zero-build Web UI.
- Simplified runtime setup to a single ignored `APIKEY` file and locked all entry points to the
  managed QuanLLM gateway.
- Prevented parallel Solver reasoning streams from interleaving by grouping raw reasoning per
  Agent in both the CLI and Web UI.
- Added one-shot semantic preflight for Solver and verifier tool calls, matrix-specific symbolic
  comparison, normalized matrix evidence, evidence-ID upserts, and strict separation between
  protocol warnings and repair-triggering candidate issues.
- Added a default structured operator-algebra backend for canonical position/momentum, angular
  momentum, and single-mode bosonic/fermionic (anti)commutators, with explicit rejection of
  unsupported mixed-family and multi-mode assumptions.
- Added shared live elapsed-time reporting across CLI, Web and REST SSE interfaces, and changed
  the Web execution timeline to a fixed-height, internally scrolling panel.
- Changed the default REST/Web bind port from 8000 to 3921; explicit `--port` and
  `QUANLLM_PORT` overrides remain supported.
- Moved FastAPI, Uvicorn, QuTiP, pycommute and OpenFermion into the default installation and
  removed the former `server`, `quantum` and `manybody` installation choices.
- Released the project under the MIT License.

### Contributors

- [fanfan32123](https://github.com/fanfan32123)
- [Hxttt1](https://github.com/Hxttt1)
