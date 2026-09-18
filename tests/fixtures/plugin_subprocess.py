from __future__ import annotations

import json
import sys
import time


def main() -> int:
    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    arguments = request.get("arguments") or {}
    if arguments.get("mode") == "overflow":
        sys.stdout.write("x" * 100_000)
        return 0
    if arguments.get("mode") == "sleep":
        time.sleep(1)
    response = {
        "protocol": 1,
        "ok": True,
        "result": {"doubled": arguments.get("value", 0) * 2},
    }
    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
