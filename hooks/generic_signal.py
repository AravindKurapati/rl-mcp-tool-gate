"""Generic, client-agnostic B2 signal shim.

Feeds a user query to the tool-gate proxy's control channel so the next tools/list
gates against the real query. Unlike hooks/user_prompt_submit.py (which parses the
Claude Code hook JSON on stdin), this accepts the query from several sources, so any
client/wrapper (Cursor, Continue, a shell alias, CI) can drive the B2 signal.

Query resolution order:
  1. --query "..."            (explicit flag)
  2. all positional args      (joined with spaces)
  3. stdin: raw text, OR JSON containing one of: prompt | query | text | message

Examples:
  python -m hooks.generic_signal "refactor the auth middleware"
  echo '{"prompt":"list my files"}' | python -m hooks.generic_signal
  python -m hooks.generic_signal --port 17800 --query "open the PR"

Always exits 0 and never blocks: if the proxy is down it silently no-ops (degrade
to B1), so it is safe to wire into any prompt path.
"""
from __future__ import annotations
import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_PORT = 17800
_JSON_KEYS = ("prompt", "query", "text", "message")


def resolve_query(args_query: str | None, positionals: list[str], stdin_data: str) -> str:
    if args_query:
        return args_query.strip()
    if positionals:
        return " ".join(positionals).strip()
    raw = (stdin_data or "").strip()
    if not raw:
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for k in _JSON_KEYS:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return ""
    except (ValueError, TypeError):
        pass
    return raw  # treat as raw prompt text


def post_query(query: str, port: int, timeout: float = 0.5) -> bool:
    if not query:
        return False
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/query",
        data=json.dumps({"text": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False  # proxy down -> degrade to B1


def main():
    ap = argparse.ArgumentParser(description="Feed a query to the tool-gate control channel.")
    ap.add_argument("positionals", nargs="*", help="query words (joined with spaces)")
    ap.add_argument("--query", default=None, help="explicit query string")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    stdin_data = ""
    if not args.query and not args.positionals and not sys.stdin.isatty():
        try:
            stdin_data = sys.stdin.read()
        except Exception:
            stdin_data = ""
    query = resolve_query(args.query, args.positionals, stdin_data)
    post_query(query, args.port)
    sys.exit(0)


if __name__ == "__main__":
    main()
