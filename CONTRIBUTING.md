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

Keep public imports backward compatible within a minor release. A breaking contract change needs
a major version and a migration note.
