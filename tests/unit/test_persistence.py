import json

from quanllm_harness import HarnessSettings, RunStatus
from quanllm_harness.contracts import HarnessResult, RequestPolicy, Usage, VerificationReport
from quanllm_harness.persistence import save_run


def test_run_record_is_atomic_and_never_contains_api_key(tmp_path):
    settings = HarnessSettings(api_key="super-secret")
    result = HarnessResult(
        status=RunStatus.VERIFIED,
        answer="answer",
        policy=RequestPolicy(depth="simple"),
        verification=VerificationReport(),
        events=(),
        usage=Usage(3, 4),
    )
    saved = save_run(tmp_path, question="question", result=result, settings=settings)
    payload = json.loads((tmp_path / f"{saved.run_id}.json").read_text())
    assert saved.record_path
    assert payload["result"]["status"] == "verified"
    assert "super-secret" not in json.dumps(payload)
    assert not list(tmp_path.glob("*.tmp"))
