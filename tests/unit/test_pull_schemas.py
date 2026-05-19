import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def test_catalog_assembled(tmp_path):
    from src.data_gen.pull_schemas import assemble_catalog
    out = tmp_path / "catalog.json"
    assemble_catalog(out, target_total=20)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["tools"]) >= 20
    assert all("name" in t and "description" in t and "server" in t for t in data["tools"])
    assert all("." in t["name"] for t in data["tools"])


def test_catalog_no_truncation():
    from src.data_gen.pull_schemas import assemble_catalog
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "catalog.json"
        assemble_catalog(out, target_total=0)
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["count"] >= 90  # we have ~140
