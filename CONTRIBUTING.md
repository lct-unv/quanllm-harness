# Contributing

Use Python 3.10 or newer and create an isolated virtual environment.

```bash
python -m pip install -e '.[dev]'
ruff format .
ruff check .
mypy
pytest --cov=quanllm_harness
python -m build
python -m twine check dist/*
```

New backends must return immutable `Evidence`, define a closed JSON Schema, document limitations,
and include unit tests. Protocol or prompt changes must add a regression fixture reproducing the
failure they address. Never add API keys, gateway credentials, run records, or student data.

CI deliberately does not enable the `setup-python` pip cache. pycommute 1.0.0 builds its Linux
extension with `-march=native`; reusing that locally built wheel on a hosted runner with a different
CPU can terminate Python with an illegal-instruction error. Do not re-enable cross-run wheel caching
unless the upstream build becomes architecture-portable.

Windows installation is handled by `install.ps1` until upstream pycommute publishes a compatible
Windows wheel. Keep its pycommute version, source SHA-256, source transformations, and runtime
self-check together. Use `--no-binary=pycommute`, never `--no-binary=:all:`, so pip may continue to
use wheels for build dependencies. Any installer change must be checked on a clean supported
Windows/Python environment before release.

Keep public imports backward compatible within a minor release. A breaking contract change needs
a major version and a migration note.

The plugin API is maintained inside `src/quanllm_harness/plugins` and must remain part of the main
`quanllm-harness` distribution. Do not split it into a separately versioned SDK. Public plugin
contract changes require compatibility tests, a security-boundary review, and matching updates to
`docs/PLUGIN_DEVELOPMENT.md` and `docs/PLUGIN_SECURITY.md`.
