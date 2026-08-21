import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

_MANAGED_ENDPOINT = bytes(
    (
        104,
        116,
        116,
        112,
        58,
        47,
        47,
        52,
        55,
        46,
        57,
        55,
        46,
        52,
        54,
        46,
        55,
        52,
        58,
        51,
        48,
        48,
        48,
        47,
        118,
        49,
    )
)
_VALIDATOR = Path(__file__).parents[2] / "scripts" / "check_release_artifacts.py"


def _write_wheel(path, content: bytes = b"clean") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("quanllm_harness/config.py", content)


def _write_sdist(path, content: bytes = b"clean") -> None:
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("quanllm_harness-0.1.1/README.md")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))


def _validate(directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_VALIDATOR), str(directory)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_release_artifact_validator_accepts_clean_archives(tmp_path):
    wheel = tmp_path / "quanllm_harness-0.1.1-py3-none-any.whl"
    sdist = tmp_path / "quanllm_harness-0.1.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    completed = _validate(tmp_path)

    assert completed.returncode == 0
    assert wheel.name in completed.stdout
    assert sdist.name in completed.stdout


@pytest.mark.parametrize("contaminated", ["wheel", "sdist"])
def test_release_artifact_validator_rejects_plaintext_endpoint(tmp_path, contaminated):
    wheel = tmp_path / "quanllm_harness-0.1.1-py3-none-any.whl"
    sdist = tmp_path / "quanllm_harness-0.1.1.tar.gz"
    _write_wheel(wheel, _MANAGED_ENDPOINT if contaminated == "wheel" else b"clean")
    _write_sdist(sdist, _MANAGED_ENDPOINT if contaminated == "sdist" else b"clean")

    completed = _validate(tmp_path)

    assert completed.returncode != 0
    assert "forbidden endpoint content" in completed.stderr
