# Release process

QuanLLM Harness releases exactly two public package assets: one Python wheel (`.whl`) and one
source distribution (`.tar.gz`). Do not attach executables, application bundles, ZIP archives or
installer packages.

1. Confirm the distribution license is present and reflected in `pyproject.toml` before the first
   public release.
2. Update the version in `pyproject.toml` and the source fallback in
   `src/quanllm_harness/__init__.py`.
3. Add a dated entry to `CHANGELOG.md` and bilingual notes at
   `release-notes/vX.Y.Z.md`. Credit all contributors.
4. Run `ruff format --check .`, `ruff check .`, `mypy`, the full test suite with coverage, and
   `node --check src/quanllm_harness/interfaces/web/static/app.js`.
5. Build with `python -m build --wheel --sdist`, run
   `python scripts/check_release_artifacts.py dist`, then run `python -m twine check dist/*`.
6. Inspect both archives for credentials, `APIKEY`, run records, caches, the complete managed
   gateway endpoint and unintended artifacts. The artifact validator must reject a plaintext
   endpoint regression.
7. Install the wheel into a clean Python environment. Smoke-test `--version`, `--graph`,
   `--capabilities`, the REST health endpoint and the Web UI.
   Do not share pip's locally built pycommute wheel cache across heterogeneous Linux CPUs.
8. Create a reviewed release commit. Add one contiguous trailer per actual co-author, with no
   blank lines between trailers. For `v0.1.2`, use:

   ```text
   Co-authored-by: Hxttt1 <3034557373@qq.com>
   Co-authored-by: fanfan32123 <fanfan13736@gmail.com>
   ```

9. Create an annotated `vX.Y.Z` tag from that commit and push the branch and tag. The tag workflow
   must publish only the wheel and sdist to the GitHub Release and PyPI.
10. Verify the GitHub Release body, both downloadable assets and the PyPI project page, then run
    one authorized end-to-end gateway request.

Never overwrite an existing public version or move a released tag. Increment the version for every
published artifact. PyPI Trusted Publishing must use the protected `pypi` environment with
reviewers configured before the tag is pushed.
