"""50 hand-written seed queries across 10 categories."""
import json
from pathlib import Path

SEEDS: list[dict] = [
    # research (5)
    {"query": "find the latest news about OpenAI's o3 model release", "ground_truth": ["brave_search.news_search"], "category": "research", "min_k": 1},
    {"query": "search the web for the current Bitcoin price", "ground_truth": ["brave_search.web_search"], "category": "research", "min_k": 1},
    {"query": "fetch the content of https://anthropic.com/research", "ground_truth": ["fetch.fetch_text"], "category": "research", "min_k": 1},
    {"query": "what's the weather like in NYC this weekend", "ground_truth": ["weather.forecast"], "category": "research", "min_k": 1},
    {"query": "look up recent papers on GRPO and summarize them by fetching the abstracts", "ground_truth": ["brave_search.web_search", "fetch.fetch_text"], "category": "research", "min_k": 2},
    # code-search (5)
    {"query": "find where the function `validate_input` is defined in this repo", "ground_truth": ["filesystem.search_files", "filesystem.read_file"], "category": "code_search", "min_k": 2},
    {"query": "list all Python files in the src directory", "ground_truth": ["filesystem.list_directory"], "category": "code_search", "min_k": 1},
    {"query": "show me the contents of README.md", "ground_truth": ["filesystem.read_file"], "category": "code_search", "min_k": 1},
    {"query": "search GitHub for examples of FastAPI middleware patterns", "ground_truth": ["github.search_code"], "category": "code_search", "min_k": 1},
    {"query": "find the most recently modified file in the project", "ground_truth": ["filesystem.list_directory", "filesystem.get_file_info"], "category": "code_search", "min_k": 2},
    # file-ops (5)
    {"query": "create a new file called notes.md with the heading '# Notes'", "ground_truth": ["filesystem.write_file"], "category": "file_ops", "min_k": 1},
    {"query": "rename old_design.md to archive_design.md", "ground_truth": ["filesystem.move_file"], "category": "file_ops", "min_k": 1},
    {"query": "make a docs/ directory if it doesn't exist", "ground_truth": ["filesystem.create_directory"], "category": "file_ops", "min_k": 1},
    {"query": "replace the version string in pyproject.toml from 0.1.0 to 0.2.0", "ground_truth": ["filesystem.edit_file"], "category": "file_ops", "min_k": 1},
    {"query": "upload data/results.csv to my S3 bucket named experiments", "ground_truth": ["filesystem.read_file", "aws_s3.put_object"], "category": "file_ops", "min_k": 2},
    # comms (5)
    {"query": "post 'standup at 10am' to the #eng channel on Slack", "ground_truth": ["slack.post_message"], "category": "comms", "min_k": 1},
    {"query": "reply to the latest thread in #design saying I'll review it tomorrow", "ground_truth": ["slack.get_channel_history", "slack.reply_to_thread"], "category": "comms", "min_k": 2},
    {"query": "send an email to alice@company.com with subject 'Project update'", "ground_truth": ["gmail.send_email"], "category": "comms", "min_k": 1},
    {"query": "find any unread emails from my manager and summarize them", "ground_truth": ["gmail.search_emails", "gmail.get_email"], "category": "comms", "min_k": 2},
    {"query": "draft an email to the team announcing the new launch but don't send it", "ground_truth": ["gmail.create_draft"], "category": "comms", "min_k": 1},
    # db (5)
    {"query": "show me the top 10 customers by total spend from the postgres database", "ground_truth": ["postgres.query"], "category": "db", "min_k": 1},
    {"query": "what tables exist in our analytics schema", "ground_truth": ["postgres.list_tables"], "category": "db", "min_k": 1},
    {"query": "describe the columns of the users table", "ground_truth": ["postgres.describe_table"], "category": "db", "min_k": 1},
    {"query": "explain why this query is slow: SELECT * FROM orders WHERE created_at > now() - interval '1 day'", "ground_truth": ["postgres.explain"], "category": "db", "min_k": 1},
    {"query": "find which Linear issue corresponds to the user record with email bob@x.com", "ground_truth": ["postgres.query", "linear.list_issues"], "category": "db", "min_k": 2},
    # deploy (4)
    {"query": "list all pods in the production namespace", "ground_truth": ["kubernetes.list_pods"], "category": "deploy", "min_k": 1},
    {"query": "show me logs from the api-server container in production", "ground_truth": ["kubernetes.list_pods", "kubernetes.get_logs"], "category": "deploy", "min_k": 2},
    {"query": "what docker containers are running on my machine right now", "ground_truth": ["docker.list_containers"], "category": "deploy", "min_k": 1},
    {"query": "apply the deployment.yaml manifest to the cluster", "ground_truth": ["kubernetes.apply_manifest"], "category": "deploy", "min_k": 1},
    # scheduling (4)
    {"query": "what time is it in Tokyo right now", "ground_truth": ["time.get_current_time"], "category": "scheduling", "min_k": 1},
    {"query": "find a 30-minute slot tomorrow when alice and bob are both free", "ground_truth": ["calendar.find_free_time"], "category": "scheduling", "min_k": 1},
    {"query": "create a 1-hour meeting at 3pm tomorrow with the design team", "ground_truth": ["calendar.create_event"], "category": "scheduling", "min_k": 1},
    {"query": "what meetings do I have today", "ground_truth": ["calendar.list_events"], "category": "scheduling", "min_k": 1},
    # debugging (4)
    {"query": "show me the most recent Sentry errors for the web project", "ground_truth": ["sentry.list_issues"], "category": "debugging", "min_k": 1},
    {"query": "get the stack trace for Sentry issue 12345", "ground_truth": ["sentry.get_issue"], "category": "debugging", "min_k": 1},
    {"query": "query datadog for API p99 latency in the last 24 hours", "ground_truth": ["datadog.query_metrics"], "category": "debugging", "min_k": 1},
    {"query": "search datadog logs for 'connection refused' errors today", "ground_truth": ["datadog.search_logs"], "category": "debugging", "min_k": 1},
    # web (4)
    {"query": "fetch the HTML of nytimes.com homepage", "ground_truth": ["fetch.fetch_html"], "category": "web", "min_k": 1},
    {"query": "search the web for tutorials on PyTorch DataLoader", "ground_truth": ["brave_search.web_search"], "category": "web", "min_k": 1},
    {"query": "find coffee shops near Washington Square Park", "ground_truth": ["brave_search.local_search"], "category": "web", "min_k": 1},
    {"query": "what does the openai.com landing page say today", "ground_truth": ["fetch.fetch_text"], "category": "web", "min_k": 1},
    # multi-step (9)
    {"query": "find the most recent github issue assigned to me and post a summary to #eng-standup", "ground_truth": ["github.search_issues", "slack.post_message"], "category": "multi_step", "min_k": 2},
    {"query": "search Notion for the launch plan doc and email it to the founders", "ground_truth": ["notion.search_pages", "gmail.send_email"], "category": "multi_step", "min_k": 2},
    {"query": "list open PRs in the repo and DM the authors on Slack to remind them", "ground_truth": ["github.list_pull_requests", "slack.post_message"], "category": "multi_step", "min_k": 2},
    {"query": "query the database for users who signed up today and add them to a Notion page", "ground_truth": ["postgres.query", "notion.create_page"], "category": "multi_step", "min_k": 2},
    {"query": "find the latest Sentry error, look up the user from postgres, and create a Linear ticket", "ground_truth": ["sentry.list_issues", "postgres.query", "linear.create_issue"], "category": "multi_step", "min_k": 3},
    {"query": "search Gmail for the AWS bill, extract the total, and post it to #finance", "ground_truth": ["gmail.search_emails", "gmail.get_email", "slack.post_message"], "category": "multi_step", "min_k": 3},
    {"query": "list k8s pods that are crash-looping and create a Sentry alert linking each one", "ground_truth": ["kubernetes.list_pods", "kubernetes.describe_pod"], "category": "multi_step", "min_k": 2},
    {"query": "read the design doc in Google Drive and create matching Linear issues for each task", "ground_truth": ["gdrive.search_drive", "gdrive.read_doc", "linear.create_issue"], "category": "multi_step", "min_k": 3},
    {"query": "find the customer's email in stripe by name and forward their latest support thread from gmail to slack", "ground_truth": ["stripe.list_subscriptions", "gmail.search_emails", "slack.post_message"], "category": "multi_step", "min_k": 3},
]


def write_seeds(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in SEEDS:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    write_seeds(Path("data/synthetic/seeds.jsonl"))
    print(f"Wrote {len(SEEDS)} seeds")
