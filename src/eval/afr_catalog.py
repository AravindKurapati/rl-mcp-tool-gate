"""Build the augmented catalog for real-traffic (afr) evaluation.

The gate was trained on a synthetic 99-tool MCP catalog. Real Claude Code traffic
uses mostly built-in tools (Bash, Read, ...) plus a few MCP tools. To evaluate the
gate on real queries against a realistically large catalog, we embed the observed
real tools (under their real names, with derived descriptions) INTO the synthetic
catalog as distractors. The synthetic tools the run never used act as the noise the
gate must prune through.
"""
from __future__ import annotations
from typing import Any

# One-line descriptions for the built-in / MCP tools observed in afr traffic.
BUILTIN_TOOL_DESCS: dict[str, str] = {
    "Bash": "Run a shell command in a persistent bash session.",
    "PowerShell": "Run a PowerShell command on Windows.",
    "Read": "Read the contents of a file from the local filesystem.",
    "Write": "Write or overwrite a file on the local filesystem.",
    "Edit": "Make an exact string replacement edit inside an existing file.",
    "Glob": "Find files by glob pattern across the codebase.",
    "Grep": "Search file contents with a regular expression.",
    "Agent": "Launch a subagent to handle a complex multi-step task.",
    "Skill": "Invoke a named skill within the conversation.",
    "ToolSearch": "Search for and load deferred tool schemas by keyword.",
    "Monitor": "Stream events from a long-running background process.",
    "AskUserQuestion": "Ask the user clarifying questions with multiple-choice options.",
    "ExitPlanMode": "Exit planning mode and present the plan for approval.",
    "ScheduleWakeup": "Schedule when to resume work in a self-paced loop.",
    "WebFetch": "Fetch and extract the contents of a web page by URL.",
    "TaskCreate": "Create new tasks to track and plan work.",
    "TaskUpdate": "Update the status or fields of an existing task.",
    "TaskList": "List the current tasks and their statuses.",
    "TaskOutput": "Read the output produced by a task or background process.",
    "TaskStop": "Stop a running task or background process.",
    "mcp__exa__web_search_exa": "Search the web for current information via Exa.",
    "mcp__exa__web_fetch_exa": "Fetch and extract a web page's contents via Exa.",
    "mcp__github__get_file_contents": "Read a file's contents from a GitHub repository.",
    "mcp__github__create_pull_request": "Open a new pull request on GitHub.",
    "mcp__github__create_or_update_file": "Create or update a file in a GitHub repo.",
    "mcp__github__create_repository": "Create a new GitHub repository.",
    "mcp__github__get_pull_request": "Get the details of a GitHub pull request.",
    "mcp__github__get_pull_request_comments": "Get review comments on a GitHub pull request.",
    "mcp__github__get_pull_request_files": "List the files changed in a GitHub pull request.",
    "mcp__github__get_pull_request_status": "Get CI/check status for a GitHub pull request.",
    "mcp__github__list_commits": "List commits in a GitHub repository or branch.",
    "mcp__github__list_pull_requests": "List pull requests in a GitHub repository.",
    "mcp__github__merge_pull_request": "Merge a GitHub pull request.",
    "mcp__github__search_repositories": "Search for GitHub repositories.",
    "mcp__github__search_users": "Search for GitHub users.",
}


def describe_tool(name: str) -> str:
    if name in BUILTIN_TOOL_DESCS:
        return BUILTIN_TOOL_DESCS[name]
    # Fall back to a readable description derived from the name so eval never crashes.
    pretty = name.replace("mcp__", "").replace("__", " ").replace("_", " ").replace(".", " ")
    return f"Tool: {pretty}."


def _real_tool_entry(name: str) -> dict[str, Any]:
    desc = describe_tool(name)
    if name.startswith("mcp__"):
        parts = name.split("__")
        server = parts[1] if len(parts) > 1 else "mcp"
        tool = parts[-1]
    else:
        server, tool = "builtin", name
    return {
        "name": name,
        "server": server,
        "tool": tool,
        "description": desc,
        "args": "",
        "embed_text": f"{name}: {desc}",
    }


def build_augmented_catalog(
    synthetic_catalog: list[dict[str, Any]],
    observed_tool_names: list[str],
) -> list[dict[str, Any]]:
    """Merge observed real tools into the synthetic catalog, de-duped by name."""
    by_name: dict[str, dict[str, Any]] = {t["name"]: t for t in synthetic_catalog}
    for name in observed_tool_names:
        if name and name not in by_name:
            by_name[name] = _real_tool_entry(name)
    return list(by_name.values())
