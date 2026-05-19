"""Assemble a catalog of MCP tool schemas from raw server definitions.

We ship hardcoded schemas in this module rather than fetching at runtime —
upstream MCP repos shift, and we need a reproducible artifact.
"""
import json
from pathlib import Path
from typing import Any

_HARDCODED_TOOLS: dict[str, list[dict[str, Any]]] = {
    "github": [
        {"name": "list_issues", "description": "List issues in a GitHub repository, optionally filtered by state, assignee, or labels.", "args": "owner, repo, state, assignee, labels"},
        {"name": "get_issue", "description": "Get a single issue by number from a GitHub repo.", "args": "owner, repo, issue_number"},
        {"name": "create_issue", "description": "Create a new GitHub issue with title and body.", "args": "owner, repo, title, body, labels"},
        {"name": "add_issue_comment", "description": "Post a comment on an existing GitHub issue.", "args": "owner, repo, issue_number, body"},
        {"name": "list_pull_requests", "description": "List pull requests in a GitHub repo, filtered by state.", "args": "owner, repo, state"},
        {"name": "get_pull_request", "description": "Fetch a single pull request by number.", "args": "owner, repo, pr_number"},
        {"name": "merge_pull_request", "description": "Merge a GitHub pull request.", "args": "owner, repo, pr_number, merge_method"},
        {"name": "create_pull_request_review", "description": "Submit a review (approve/comment/request_changes) on a PR.", "args": "owner, repo, pr_number, event, body"},
        {"name": "list_commits", "description": "List commits on a branch.", "args": "owner, repo, branch"},
        {"name": "search_code", "description": "Search code across GitHub.", "args": "query, repo"},
        {"name": "get_file_contents", "description": "Read a file from a GitHub repo at a path.", "args": "owner, repo, path, ref"},
        {"name": "create_or_update_file", "description": "Create or update a file in a GitHub repo via the API.", "args": "owner, repo, path, content, message, branch"},
        {"name": "list_branches", "description": "List branches in a GitHub repo.", "args": "owner, repo"},
        {"name": "create_branch", "description": "Create a new branch from a base ref.", "args": "owner, repo, branch_name, from_ref"},
        {"name": "search_issues", "description": "Search GitHub issues across repos.", "args": "query"},
    ],
    "filesystem": [
        {"name": "read_file", "description": "Read the contents of a file from the local filesystem.", "args": "path"},
        {"name": "write_file", "description": "Write content to a file on the local filesystem.", "args": "path, content"},
        {"name": "edit_file", "description": "Apply edits to an existing file by replacing strings.", "args": "path, edits"},
        {"name": "list_directory", "description": "List entries (files and subdirectories) in a directory.", "args": "path"},
        {"name": "create_directory", "description": "Create a new directory on the local filesystem.", "args": "path"},
        {"name": "move_file", "description": "Move or rename a file or directory.", "args": "source, destination"},
        {"name": "search_files", "description": "Search for files by name pattern recursively.", "args": "root, pattern"},
        {"name": "get_file_info", "description": "Get metadata (size, mtime, type) for a path.", "args": "path"},
    ],
    "slack": [
        {"name": "post_message", "description": "Post a message to a Slack channel.", "args": "channel, text, thread_ts"},
        {"name": "reply_to_thread", "description": "Reply in a Slack message thread.", "args": "channel, thread_ts, text"},
        {"name": "list_channels", "description": "List Slack channels the bot is in.", "args": ""},
        {"name": "get_channel_history", "description": "Fetch recent messages from a Slack channel.", "args": "channel, limit"},
        {"name": "add_reaction", "description": "Add an emoji reaction to a Slack message.", "args": "channel, ts, name"},
        {"name": "search_messages", "description": "Search Slack messages by text.", "args": "query"},
        {"name": "list_users", "description": "List Slack workspace users.", "args": ""},
    ],
    "postgres": [
        {"name": "query", "description": "Execute a read-only SQL query against a Postgres database.", "args": "sql"},
        {"name": "list_tables", "description": "List tables in the connected Postgres database.", "args": "schema"},
        {"name": "describe_table", "description": "Get column types and constraints for a Postgres table.", "args": "table"},
        {"name": "explain", "description": "Run EXPLAIN on a SQL query to get the plan.", "args": "sql"},
    ],
    "brave_search": [
        {"name": "web_search", "description": "Search the web for a query using Brave Search.", "args": "query, count"},
        {"name": "news_search", "description": "Search recent news articles for a query.", "args": "query"},
        {"name": "local_search", "description": "Search local businesses near a location.", "args": "query, location"},
    ],
    "gmail": [
        {"name": "send_email", "description": "Send an email via Gmail.", "args": "to, subject, body, cc"},
        {"name": "list_emails", "description": "List emails matching a query in Gmail.", "args": "query, max_results"},
        {"name": "get_email", "description": "Fetch a single email by ID.", "args": "id"},
        {"name": "create_draft", "description": "Create a draft email in Gmail.", "args": "to, subject, body"},
        {"name": "search_emails", "description": "Search emails in Gmail using Gmail query syntax.", "args": "query"},
        {"name": "list_labels", "description": "List Gmail labels.", "args": ""},
        {"name": "label_email", "description": "Apply a label to a Gmail message.", "args": "id, label"},
    ],
    "gdrive": [
        {"name": "list_files", "description": "List files in a Google Drive folder.", "args": "folder_id"},
        {"name": "read_doc", "description": "Read the text content of a Google Doc.", "args": "doc_id"},
        {"name": "create_doc", "description": "Create a new Google Doc with content.", "args": "title, content"},
        {"name": "search_drive", "description": "Search Google Drive by file name or content.", "args": "query"},
    ],
    "time": [
        {"name": "get_current_time", "description": "Get the current time in a given timezone.", "args": "timezone"},
        {"name": "convert_timezone", "description": "Convert a timestamp from one timezone to another.", "args": "time, from_tz, to_tz"},
        {"name": "schedule_reminder", "description": "Schedule a reminder at a future time.", "args": "when, message"},
    ],
    "memory": [
        {"name": "create_entity", "description": "Create a named entity node in the knowledge graph.", "args": "name, type, observations"},
        {"name": "create_relation", "description": "Create a relation between two entities.", "args": "from_entity, to_entity, relation_type"},
        {"name": "search_nodes", "description": "Search the knowledge graph for entities by query.", "args": "query"},
        {"name": "open_nodes", "description": "Read full data for a list of entity names.", "args": "names"},
        {"name": "delete_entity", "description": "Delete an entity from the knowledge graph.", "args": "name"},
    ],
    "fetch": [
        {"name": "fetch_url", "description": "Fetch the content at a URL via HTTP GET.", "args": "url"},
        {"name": "fetch_html", "description": "Fetch a URL and return HTML.", "args": "url"},
        {"name": "fetch_text", "description": "Fetch a URL and return extracted plain text.", "args": "url"},
    ],
}

_FILLERS: dict[str, list[dict[str, Any]]] = {
    "linear": [
        {"name": "create_issue", "description": "Create a new issue in Linear.", "args": "team, title, description"},
        {"name": "list_issues", "description": "List Linear issues filtered by assignee or status.", "args": "assignee, status"},
        {"name": "update_issue", "description": "Update a Linear issue status or assignee.", "args": "id, status, assignee"},
        {"name": "list_teams", "description": "List Linear teams.", "args": ""},
    ],
    "notion": [
        {"name": "create_page", "description": "Create a new Notion page under a parent.", "args": "parent_id, title, content"},
        {"name": "search_pages", "description": "Search Notion pages by title or content.", "args": "query"},
        {"name": "update_page", "description": "Update properties or content of a Notion page.", "args": "page_id, properties"},
        {"name": "list_databases", "description": "List Notion databases.", "args": ""},
        {"name": "query_database", "description": "Query a Notion database with filters.", "args": "database_id, filter"},
    ],
    "jira": [
        {"name": "create_issue", "description": "Create a Jira issue.", "args": "project, summary, type"},
        {"name": "search_issues", "description": "Search Jira issues using JQL.", "args": "jql"},
        {"name": "transition_issue", "description": "Move a Jira issue to a new status.", "args": "issue_key, transition"},
        {"name": "add_comment", "description": "Add a comment to a Jira issue.", "args": "issue_key, body"},
    ],
    "calendar": [
        {"name": "create_event", "description": "Create a calendar event with start/end and attendees.", "args": "title, start, end, attendees"},
        {"name": "list_events", "description": "List upcoming calendar events.", "args": "time_min, time_max"},
        {"name": "cancel_event", "description": "Cancel a scheduled calendar event.", "args": "event_id"},
        {"name": "find_free_time", "description": "Find a free time slot for a meeting given attendees.", "args": "attendees, duration"},
    ],
    "stripe": [
        {"name": "create_customer", "description": "Create a Stripe customer record.", "args": "email, name"},
        {"name": "create_charge", "description": "Charge a payment method via Stripe.", "args": "customer_id, amount, currency"},
        {"name": "list_subscriptions", "description": "List active Stripe subscriptions.", "args": "customer_id"},
        {"name": "refund_charge", "description": "Issue a refund for a Stripe charge.", "args": "charge_id, amount"},
    ],
    "sentry": [
        {"name": "list_issues", "description": "List Sentry error issues for a project.", "args": "project, status"},
        {"name": "get_issue", "description": "Fetch a single Sentry issue with stack traces.", "args": "issue_id"},
        {"name": "resolve_issue", "description": "Mark a Sentry issue as resolved.", "args": "issue_id"},
    ],
    "datadog": [
        {"name": "query_metrics", "description": "Query Datadog time series metrics.", "args": "query, from, to"},
        {"name": "list_monitors", "description": "List Datadog monitors and their state.", "args": "tags"},
        {"name": "search_logs", "description": "Search Datadog logs by query.", "args": "query, from, to"},
    ],
    "kubernetes": [
        {"name": "list_pods", "description": "List Kubernetes pods in a namespace.", "args": "namespace"},
        {"name": "get_logs", "description": "Stream logs from a Kubernetes pod.", "args": "namespace, pod, container"},
        {"name": "describe_pod", "description": "Get detailed status of a Kubernetes pod.", "args": "namespace, pod"},
        {"name": "apply_manifest", "description": "Apply a Kubernetes manifest YAML.", "args": "manifest"},
    ],
    "docker": [
        {"name": "list_containers", "description": "List Docker containers (running or all).", "args": "all"},
        {"name": "inspect_container", "description": "Get full details of a Docker container.", "args": "container_id"},
        {"name": "container_logs", "description": "Read logs from a Docker container.", "args": "container_id"},
    ],
    "aws_s3": [
        {"name": "list_buckets", "description": "List S3 buckets in the AWS account.", "args": ""},
        {"name": "list_objects", "description": "List objects in an S3 bucket prefix.", "args": "bucket, prefix"},
        {"name": "get_object", "description": "Download an S3 object's contents.", "args": "bucket, key"},
        {"name": "put_object", "description": "Upload content to an S3 object.", "args": "bucket, key, content"},
    ],
    "weather": [
        {"name": "current_weather", "description": "Get current weather for a location.", "args": "location"},
        {"name": "forecast", "description": "Get a multi-day weather forecast.", "args": "location, days"},
    ],
}


def assemble_catalog(out_path: Path, target_total: int = 150, raw_dir: Path | None = None) -> None:
    """Assemble the tool catalog JSON file."""
    tools: list[dict[str, Any]] = []
    for server_name, server_tools in {**_HARDCODED_TOOLS, **_FILLERS}.items():
        for t in server_tools:
            full_name = f"{server_name}.{t['name']}"
            embed_text = f"{full_name}: {t['description']}"
            if t.get("args"):
                embed_text += f" Args: {t['args']}."
            tools.append({
                "name": full_name,
                "server": server_name,
                "tool": t["name"],
                "description": t["description"],
                "args": t.get("args", ""),
                "embed_text": embed_text,
            })
    if target_total > 0 and len(tools) < target_total:
        raise RuntimeError(f"Only {len(tools)} tools assembled, need {target_total}.")
    if target_total > 0:
        tools = tools[:target_total]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tools": tools, "count": len(tools)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/synthetic/catalog.json"))
    p.add_argument("--target", type=int, default=0, help="0 = no truncation")
    args = p.parse_args()
    assemble_catalog(args.out, target_total=args.target)
    print(f"Wrote {args.out}")
