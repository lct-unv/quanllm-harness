import io
import tarfile
import zipfile

import pytest

from scripts.check_release_artifacts import validate_release_directory

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


def _write_wheel(path, content: bytes = b"clean") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("quanllm_harness/config.py", content)


def _write_sdist(path, content: bytes = b"clean") -> None:
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("quanllm_harness-0.1.1/README.md")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))


def test_release_artifact_validator_accepts_clean_archives(tmp_path):
    wheel = tmp_path / "quanllm_harness-0.1.1-py3-none-any.whl"
    sdist = tmp_path / "quanllm_harness-0.1.1.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    assert validate_release_directory(tmp_path) == (wheel, sdist)


@pytest.mark.parametrize("contaminated", ["wheel", "sdist"])
def test_release_artifact_validator_rejects_plaintext_endpoint(tmp_path, contaminated):
    wheel = tmp_path / "quanllm_harness-0.1.1-py3-none-any.whl"
    sdist = tmp_path / "quanllm_harness-0.1.1.tar.gz"
    _write_wheel(wheel, _MANAGED_ENDPOINT if contaminated == "wheel" else b"clean")
    _write_sdist(sdist, _MANAGED_ENDPOINT if contaminated == "sdist" else b"clean")

    with pytest.raises(ValueError, match="forbidden endpoint content"):
        validate_release_directory(tmp_path)
