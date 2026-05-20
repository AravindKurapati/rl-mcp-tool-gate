import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.proxy.multiplex import NamespacedTool


def test_namespacing():
    t = NamespacedTool(server="github", tool="list_issues", description="x", input_schema={})
    assert t.full_name == "github.list_issues"


def test_parses_namespaced_name():
    server, tool = NamespacedTool.split("github.list_issues")
    assert server == "github"
    assert tool == "list_issues"


def test_split_handles_underscored_tool():
    server, tool = NamespacedTool.split("aws_s3.put_object")
    assert server == "aws_s3"
    assert tool == "put_object"
