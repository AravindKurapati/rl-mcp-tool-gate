"""Load upstreams.toml."""
from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Upstream:
    name: str
    command: list[str]
    env: dict[str, str]


@dataclass
class ProxyConfig:
    budget_tokens: int
    top_k: int
    gate_checkpoint: str
    control_channel_port: int
    upstreams: list[Upstream]
    enable_decision_log: bool = True
    decision_log_path: str = ""


def _expand(s: str) -> str:
    return os.path.expandvars(s)


def load_config(path: Path) -> ProxyConfig:
    path = Path(path).resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)
    upstreams = []
    for u in raw.get("upstreams", []):
        env = {k: _expand(v) for k, v in u.get("env", {}).items()}
        upstreams.append(Upstream(name=u["name"], command=list(u["command"]), env=env))
    # Resolve a relative gate_checkpoint against the config file's directory so the
    # LoRA loads regardless of the process cwd (Claude Code spawns from the project root).
    ckpt = raw.get("gate_checkpoint", "")
    if ckpt and not Path(ckpt).is_absolute():
        ckpt = str((path.parent / ckpt).resolve())
    return ProxyConfig(
        budget_tokens=int(raw.get("budget_tokens", 4000)),
        top_k=int(raw.get("top_k", 12)),
        gate_checkpoint=ckpt,
        control_channel_port=int(raw.get("control_channel_port", 17800)),
        upstreams=upstreams,
        enable_decision_log=bool(raw.get("enable_decision_log", True)),
        decision_log_path=str(raw.get("decision_log_path", "")),
    )
