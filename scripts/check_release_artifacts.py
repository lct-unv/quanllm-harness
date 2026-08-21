from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

FORBIDDEN_PARTS = {
    ".env",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "APIKEY",
    "runs",
}


def _members(archive: Path) -> list[str]:
    if archive.name.endswith(".whl"):
        with zipfile.ZipFile(archive) as handle:
            return handle.namelist()
    with tarfile.open(archive, "r:gz") as handle:
        return handle.getnames()


def _validate_members(archive: Path, members: list[str]) -> None:
    if not members:
        raise ValueError(f"{archive.name} is empty")
    for member in members:
        path = PurePosixPath(member)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{archive.name} contains unsafe path: {member}")
        forbidden = FORBIDDEN_PARTS.intersection(path.parts)
        if forbidden:
            raise ValueError(
                f"{archive.name} contains forbidden path component {sorted(forbidden)!r}: {member}"
            )


def validate_release_directory(directory: Path) -> tuple[Path, Path]:
    assets = sorted(path for path in directory.iterdir() if path.is_file())
    wheels = [path for path in assets if path.name.endswith(".whl")]
    sdists = [path for path in assets if path.name.endswith(".tar.gz")]
    if len(assets) != 2 or len(wheels) != 1 or len(sdists) != 1:
        names = [path.name for path in assets]
        raise ValueError(f"expected exactly one wheel and one sdist, found: {names}")
    for archive in assets:
        _validate_members(archive, _members(archive))
    return wheels[0], sdists[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QuanLLM Harness release artifacts")
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    args = parser.parse_args()
    wheel, sdist = validate_release_directory(args.directory)
    print(f"wheel: {wheel.name}")
    print(f"sdist: {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
