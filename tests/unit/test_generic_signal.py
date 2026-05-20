import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hooks.generic_signal import resolve_query, post_query


def test_resolve_explicit_query_wins():
    assert resolve_query("explicit", ["pos", "words"], '{"prompt":"json"}') == "explicit"


def test_resolve_positionals():
    assert resolve_query(None, ["open", "the", "PR"], "") == "open the PR"


def test_resolve_stdin_json_keys():
    assert resolve_query(None, [], '{"prompt":"do the thing"}') == "do the thing"
    assert resolve_query(None, [], '{"query":"q"}') == "q"
    assert resolve_query(None, [], '{"text":"t"}') == "t"
    assert resolve_query(None, [], '{"message":"m"}') == "m"


def test_resolve_stdin_raw_text():
    assert resolve_query(None, [], "just raw text") == "just raw text"


def test_resolve_empty():
    assert resolve_query(None, [], "") == ""
    assert resolve_query(None, [], "{}") == ""


def test_post_query_noop_when_down():
    # Nothing listening on this port -> returns False, never raises.
    assert post_query("hello", port=59999, timeout=0.2) is False
    # Empty query short-circuits.
    assert post_query("", port=17800) is False
