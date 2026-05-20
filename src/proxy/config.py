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


def _expand(s: str) -> str:
    return os.path.expandvars(s)


def load_config(path: Path) -> ProxyConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    upstreams = []
    for u in raw.get("upstreams", []):
        env = {k: _expand(v) for k, v in u.get("env", {}).items()}
        upstreams.append(Upstream(name=u["name"], command=list(u["command"]), env=env))
    return ProxyConfig(
        budget_tokens=int(raw.get("budget_tokens", 4000)),
        top_k=int(raw.get("top_k", 12)),
        gate_checkpoint=raw.get("gate_checkpoint", ""),
        control_channel_port=int(raw.get("control_channel_port", 17800)),
        upstreams=upstreams,
    )
