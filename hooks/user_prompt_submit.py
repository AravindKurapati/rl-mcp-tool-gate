"""Claude Code UserPromptSubmit hook.

POSTs the user prompt to the tool-gate proxy's control channel so the next
tools/list call uses the user's actual query as the gate signal.
"""
from __future__ import annotations
import json
import sys
import urllib.request
import urllib.error

PORT = 17800  # must match control_channel_port in upstreams.toml


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        sys.exit(0)
    prompt = data.get("prompt", "").strip()
    if not prompt:
        sys.exit(0)
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/query",
        data=json.dumps({"text": prompt}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=0.5):
            pass
    except (urllib.error.URLError, TimeoutError, OSError):
        pass  # proxy down -> degrade to B1; never block the prompt
    sys.exit(0)


if __name__ == "__main__":
    main()
