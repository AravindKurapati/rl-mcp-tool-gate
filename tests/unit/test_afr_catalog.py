import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.afr_catalog import build_augmented_catalog, describe_tool, BUILTIN_TOOL_DESCS


def _synthetic():
    return [
        {"name": "github.list_issues", "server": "github", "tool": "list_issues",
         "description": "x", "args": "", "embed_text": "github.list_issues: x"},
        {"name": "slack.post_message", "server": "slack", "tool": "post_message",
         "description": "y", "args": "", "embed_text": "slack.post_message: y"},
    ]


def test_augmented_catalog_merges_and_dedups():
    observed = ["Bash", "Read", "mcp__exa__web_search_exa", "github.list_issues"]
    cat = build_augmented_catalog(_synthetic(), observed)
    names = [t["name"] for t in cat]
    # synthetic preserved
    assert "slack.post_message" in names
    # real tools injected
    assert "Bash" in names and "mcp__exa__web_search_exa" in names
    # dup (github.list_issues already synthetic) not duplicated
    assert names.count("github.list_issues") == 1
    # no duplicate names overall
    assert len(names) == len(set(names))


def test_real_tool_entry_shape_and_mcp_parse():
    cat = build_augmented_catalog([], ["mcp__exa__web_search_exa", "Read"])
    by = {t["name"]: t for t in cat}
    exa = by["mcp__exa__web_search_exa"]
    assert exa["server"] == "exa"
    assert exa["tool"] == "web_search_exa"
    assert exa["embed_text"].startswith("mcp__exa__web_search_exa: ")
    assert by["Read"]["server"] == "builtin"
    assert by["Read"]["description"] == BUILTIN_TOOL_DESCS["Read"]


def test_describe_unknown_tool_does_not_crash():
    d = describe_tool("mcp__weird__some_new_tool")
    assert "weird" in d and "some new tool" in d
