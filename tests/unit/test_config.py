"""Unit tests for src/proxy/config.load_config.

load_config parses upstreams.toml and is what wires the live MCP proxy: it expands
env-var references in upstream environments, resolves a relative gate_checkpoint
against the config file's directory (so the LoRA loads regardless of cwd), and fills
in defaults. None of that was covered. Pure/CPU-only — just file parsing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.proxy.config import load_config


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "upstreams.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_parses_scalars_and_upstreams(tmp_path: Path):
    cfg = load_config(_write(tmp_path, """
budget_tokens = 2000
top_k = 7
control_channel_port = 19000

[[upstreams]]
name = "gh"
command = ["python", "-m", "server"]

[[upstreams]]
name = "fs"
command = ["fs-server"]
"""))
    assert cfg.budget_tokens == 2000
    assert cfg.top_k == 7
    assert cfg.control_channel_port == 19000
    assert [u.name for u in cfg.upstreams] == ["gh", "fs"]
    assert cfg.upstreams[0].command == ["python", "-m", "server"]


def test_defaults_when_unspecified(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "gate_checkpoint = \"/abs/ckpt\"\n"))
    assert cfg.budget_tokens == 4000
    assert cfg.top_k == 12
    assert cfg.control_channel_port == 17800
    assert cfg.upstreams == []


def test_env_vars_in_upstream_env_are_expanded(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret123")
    cfg = load_config(_write(tmp_path, """
[[upstreams]]
name = "gh"
command = ["server"]
env = { TOKEN = "$MY_TOKEN", PLAIN = "literal" }
"""))
    assert cfg.upstreams[0].env == {"TOKEN": "secret123", "PLAIN": "literal"}


def test_missing_env_var_left_as_literal(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    cfg = load_config(_write(tmp_path, """
[[upstreams]]
name = "gh"
command = ["server"]
env = { TOKEN = "$DEFINITELY_UNSET_VAR" }
"""))
    # os.path.expandvars leaves unknown vars untouched rather than blanking them.
    assert cfg.upstreams[0].env["TOKEN"] == "$DEFINITELY_UNSET_VAR"


def test_relative_checkpoint_resolved_against_config_dir(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "gate_checkpoint = \"ckpts/sft\"\n"))
    assert Path(cfg.gate_checkpoint).is_absolute()
    assert cfg.gate_checkpoint == str((tmp_path / "ckpts/sft").resolve())


def test_absolute_checkpoint_left_unchanged(tmp_path: Path):
    cfg = load_config(_write(tmp_path, "gate_checkpoint = \"/opt/models/sft\"\n"))
    assert cfg.gate_checkpoint == "/opt/models/sft"


def test_empty_checkpoint_stays_empty(tmp_path: Path):
    # No gate_checkpoint key -> empty string, not resolved into the config dir.
    cfg = load_config(_write(tmp_path, "top_k = 5\n"))
    assert cfg.gate_checkpoint == ""
