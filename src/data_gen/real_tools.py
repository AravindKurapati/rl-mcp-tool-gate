"""Real-format MCP tools observed in agent-flight-recorder traffic, plus realistic
queries that target them.

The v1 model overfit because it learned synthetic names (brave_search.web_search,
github.search_code, ...) while real traffic uses mcp__exa__* / mcp__github__* names.
These tools + seeds teach the model the real-format names for the capabilities that
actually appear in real Claude Code sessions.

Names match afr exactly so the trained model lines up with real ground truth.
"""
from __future__ import annotations
from typing import Any

# (name, description) for the MCP tools seen in real traffic.
_REAL_TOOL_DEFS: list[tuple[str, str]] = [
    ("mcp__exa__web_search_exa", "Search the web for current information, news, facts, people, or companies and return clean text from top results."),
    ("mcp__exa__web_fetch_exa", "Fetch a specific URL and extract its full page contents as clean text."),
    ("mcp__github__get_file_contents", "Read the contents of a file or directory from a GitHub repository."),
    ("mcp__github__create_pull_request", "Open a new pull request on GitHub from a head branch into a base branch."),
    ("mcp__github__create_or_update_file", "Create a new file or update an existing file in a GitHub repository."),
    ("mcp__github__create_repository", "Create a new GitHub repository."),
    ("mcp__github__get_pull_request", "Get the details and metadata of a specific GitHub pull request."),
    ("mcp__github__get_pull_request_comments", "Get the review comments on a GitHub pull request."),
    ("mcp__github__get_pull_request_files", "List the files changed in a GitHub pull request."),
    ("mcp__github__get_pull_request_status", "Get the CI and check status for a GitHub pull request."),
    ("mcp__github__list_commits", "List commits in a GitHub repository or on a branch."),
    ("mcp__github__list_pull_requests", "List the pull requests in a GitHub repository, optionally filtered by state."),
    ("mcp__github__merge_pull_request", "Merge an open GitHub pull request."),
    ("mcp__github__search_repositories", "Search GitHub for repositories matching a query."),
    ("mcp__github__search_users", "Search GitHub for users matching a query."),
]


def real_tool_catalog() -> list[dict[str, Any]]:
    out = []
    for name, desc in _REAL_TOOL_DEFS:
        parts = name.split("__")
        server, tool = parts[1], parts[-1]
        out.append({
            "name": name,
            "server": server,
            "tool": tool,
            "description": desc,
            "args": "",
            "embed_text": f"{name}: {desc}",
        })
    return out


# Realistic training queries -> real-format ground truth. Phrased like real goals
# (no provider name leaked), single- and multi-tool. category="real_mcp".
REAL_SEEDS: list[dict] = [
    {"query": "search the web for the latest news on the EU AI Act", "ground_truth": ["mcp__exa__web_search_exa"], "min_k": 1},
    {"query": "look up current best practices for GRPO fine-tuning", "ground_truth": ["mcp__exa__web_search_exa"], "min_k": 1},
    {"query": "find out who the CEO of Anthropic is", "ground_truth": ["mcp__exa__web_search_exa"], "min_k": 1},
    {"query": "what's the latest on the OpenAI o3 release", "ground_truth": ["mcp__exa__web_search_exa"], "min_k": 1},
    {"query": "fetch the full text of https://example.com/blog/post", "ground_truth": ["mcp__exa__web_fetch_exa"], "min_k": 1},
    {"query": "open this url and pull the article content for me", "ground_truth": ["mcp__exa__web_fetch_exa"], "min_k": 1},
    {"query": "search the web for the FastAPI middleware docs and fetch the top result", "ground_truth": ["mcp__exa__web_search_exa", "mcp__exa__web_fetch_exa"], "min_k": 2},
    {"query": "research recent papers on tool selection for agents and read the best one", "ground_truth": ["mcp__exa__web_search_exa", "mcp__exa__web_fetch_exa"], "min_k": 2},
    {"query": "read the README from the anthropics/claude-code repo on github", "ground_truth": ["mcp__github__get_file_contents"], "min_k": 1},
    {"query": "show me the contents of src/main.py in my github repo", "ground_truth": ["mcp__github__get_file_contents"], "min_k": 1},
    {"query": "open a pull request from my feature branch into main", "ground_truth": ["mcp__github__create_pull_request"], "min_k": 1},
    {"query": "create a PR for the changes I just pushed", "ground_truth": ["mcp__github__create_pull_request"], "min_k": 1},
    {"query": "commit this updated config file to the repo on github", "ground_truth": ["mcp__github__create_or_update_file"], "min_k": 1},
    {"query": "spin up a new github repository called tool-gate", "ground_truth": ["mcp__github__create_repository"], "min_k": 1},
    {"query": "get the details of pull request number 42", "ground_truth": ["mcp__github__get_pull_request"], "min_k": 1},
    {"query": "show me the review comments on PR 42", "ground_truth": ["mcp__github__get_pull_request_comments"], "min_k": 1},
    {"query": "what files were changed in pull request 42", "ground_truth": ["mcp__github__get_pull_request_files"], "min_k": 1},
    {"query": "did the CI checks pass on PR 42", "ground_truth": ["mcp__github__get_pull_request_status"], "min_k": 1},
    {"query": "list the recent commits on the main branch", "ground_truth": ["mcp__github__list_commits"], "min_k": 1},
    {"query": "show me all the open pull requests in this repo", "ground_truth": ["mcp__github__list_pull_requests"], "min_k": 1},
    {"query": "merge pull request 42 now that it's approved", "ground_truth": ["mcp__github__merge_pull_request"], "min_k": 1},
    {"query": "find popular open-source RAG repositories on github", "ground_truth": ["mcp__github__search_repositories"], "min_k": 1},
    {"query": "search github for the user asgeirtj", "ground_truth": ["mcp__github__search_users"], "min_k": 1},
    {"query": "review PR 42 for me: pull the changed files and the comments", "ground_truth": ["mcp__github__get_pull_request_files", "mcp__github__get_pull_request_comments"], "min_k": 2},
    {"query": "find the locus repo on github and read its README", "ground_truth": ["mcp__github__search_repositories", "mcp__github__get_file_contents"], "min_k": 2},
    {"query": "check if PR 42 is green and then merge it", "ground_truth": ["mcp__github__get_pull_request_status", "mcp__github__merge_pull_request"], "min_k": 2},
]

# Held-out real-format queries (distinct phrasing, never trained on) for in-dist eval.
REAL_HELDOUT: list[dict] = [
    {"query": "look up the current price of bitcoin online", "ground_truth": ["mcp__exa__web_search_exa"], "category": "real_mcp", "min_k": 1},
    {"query": "find recent coverage of the latest Claude model", "ground_truth": ["mcp__exa__web_search_exa"], "category": "real_mcp", "min_k": 1},
    {"query": "grab the contents of this documentation page for me", "ground_truth": ["mcp__exa__web_fetch_exa"], "category": "real_mcp", "min_k": 1},
    {"query": "search online for vLLM tuning guides and read the top hit", "ground_truth": ["mcp__exa__web_search_exa", "mcp__exa__web_fetch_exa"], "category": "real_mcp", "min_k": 2},
    {"query": "read pyproject.toml from my repository on github", "ground_truth": ["mcp__github__get_file_contents"], "category": "real_mcp", "min_k": 1},
    {"query": "raise a pull request with these edits", "ground_truth": ["mcp__github__create_pull_request"], "category": "real_mcp", "min_k": 1},
    {"query": "update the version file and push it to github", "ground_truth": ["mcp__github__create_or_update_file"], "category": "real_mcp", "min_k": 1},
    {"query": "list the latest commits so I can write release notes", "ground_truth": ["mcp__github__list_commits"], "category": "real_mcp", "min_k": 1},
    {"query": "show me what's changed in pull request 7", "ground_truth": ["mcp__github__get_pull_request_files"], "category": "real_mcp", "min_k": 1},
    {"query": "search github for transformer training repositories", "ground_truth": ["mcp__github__search_repositories"], "category": "real_mcp", "min_k": 1},
    {"query": "find the repo for this project and read its setup file", "ground_truth": ["mcp__github__search_repositories", "mcp__github__get_file_contents"], "category": "real_mcp", "min_k": 2},
    {"query": "verify the checks on PR 7 then merge it", "ground_truth": ["mcp__github__get_pull_request_status", "mcp__github__merge_pull_request"], "category": "real_mcp", "min_k": 2},
]
