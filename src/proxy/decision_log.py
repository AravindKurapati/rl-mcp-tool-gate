"""SQLite logger for gate decisions and downstream tool calls.

Records every tools/list interception and every tools/call so future evals,
the bandit pivot, and live debugging have ground truth about what the gate
surfaced vs what the agent actually used.

The DB lives at ~/.tool_gate/decisions.db by default (separate from afr.db).
Writes are async via a queue so the proxy hot path never blocks on disk.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gate_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    decision_idx INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    signal_kind TEXT NOT NULL,
    signal_text TEXT NOT NULL,
    catalog_hash TEXT NOT NULL,
    catalog_size INTEGER NOT NULL,
    top_k INTEGER NOT NULL,
    budget_tokens INTEGER NOT NULL,
    selected_json TEXT NOT NULL,
    encoder_ckpt TEXT,
    latency_ms REAL NOT NULL,
    UNIQUE(session_id, decision_idx)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    decision_idx INTEGER,
    timestamp REAL NOT NULL,
    tool_name TEXT NOT NULL,
    was_surfaced INTEGER NOT NULL,
    status TEXT,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_decisions_session ON gate_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_calls_session ON tool_calls(session_id);
"""


def default_db_path() -> Path:
    return Path.home() / ".tool_gate" / "decisions.db"


def catalog_hash(catalog: list[dict[str, Any]]) -> str:
    names = sorted(t["name"] for t in catalog)
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


def mint_session_id(boot_uuid: str | None = None) -> str:
    boot_uuid = boot_uuid or uuid.uuid4().hex
    seed = f"{boot_uuid}::{time.time()}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


@dataclass
class DecisionRow:
    signal_kind: str
    signal_text: str
    catalog_hash: str
    catalog_size: int
    top_k: int
    budget_tokens: int
    selected: list[dict[str, Any]]
    encoder_ckpt: str | None
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class CallRow:
    tool_name: str
    was_surfaced: int
    status: str | None = None
    duration_ms: int | None = None
    timestamp: float = field(default_factory=time.time)


class DecisionLog:
    """Async-friendly SQLite logger. Never raises on write errors."""

    def __init__(self, db_path: Path | None = None, session_id: str | None = None):
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.session_id = session_id or mint_session_id()
        self._decision_idx = 0
        self._last_surfaced: set[str] = set()
        self._queue: asyncio.Queue[tuple[str, Any]] | None = None
        self._worker: asyncio.Task | None = None
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA_SQL)

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(str(self.db_path))
        try:
            yield con
            con.commit()
        finally:
            con.close()

    async def start(self) -> None:
        if self._worker is not None:
            return
        self._queue = asyncio.Queue()
        self._worker = asyncio.create_task(self._drain())

    async def stop(self) -> None:
        if self._queue is None or self._worker is None:
            return
        await self._queue.put(("__stop__", None))
        try:
            await asyncio.wait_for(self._worker, timeout=5.0)
        except asyncio.TimeoutError:
            self._worker.cancel()
        self._worker = None
        self._queue = None

    async def _drain(self) -> None:
        assert self._queue is not None
        while True:
            kind, payload = await self._queue.get()
            if kind == "__stop__":
                return
            try:
                if kind == "decision":
                    self._write_decision_sync(payload)
                elif kind == "call":
                    self._write_call_sync(payload)
            except Exception as e:
                print(f"[decision_log] write error: {e}", file=sys.stderr, flush=True)

    def log_decision(self, row: DecisionRow) -> int:
        """Log a gate decision. Returns the decision_idx assigned. Non-blocking if
        the async worker has started; otherwise writes synchronously."""
        idx = self._decision_idx
        self._decision_idx += 1
        self._last_surfaced = {t["name"] for t in row.selected}
        payload = (idx, row)
        if self._queue is not None:
            self._queue.put_nowait(("decision", payload))
        else:
            try:
                self._write_decision_sync(payload)
            except Exception as e:
                print(f"[decision_log] sync write error: {e}", file=sys.stderr, flush=True)
        return idx

    def log_call(self, row: CallRow) -> None:
        if self._queue is not None:
            self._queue.put_nowait(("call", row))
        else:
            try:
                self._write_call_sync(row)
            except Exception as e:
                print(f"[decision_log] sync write error: {e}", file=sys.stderr, flush=True)

    def was_surfaced(self, tool_name: str) -> int:
        return 1 if tool_name in self._last_surfaced else 0

    def current_decision_idx(self) -> int | None:
        if self._decision_idx == 0:
            return None
        return self._decision_idx - 1

    def _write_decision_sync(self, payload: tuple[int, DecisionRow]) -> None:
        idx, row = payload
        with self._connect() as con:
            con.execute(
                """INSERT INTO gate_decisions
                   (session_id, decision_idx, timestamp, signal_kind, signal_text,
                    catalog_hash, catalog_size, top_k, budget_tokens, selected_json,
                    encoder_ckpt, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, idx, row.timestamp, row.signal_kind, row.signal_text,
                    row.catalog_hash, row.catalog_size, row.top_k, row.budget_tokens,
                    json.dumps(row.selected), row.encoder_ckpt, row.latency_ms,
                ),
            )

    def _write_call_sync(self, row: CallRow) -> None:
        with self._connect() as con:
            con.execute(
                """INSERT INTO tool_calls
                   (session_id, decision_idx, timestamp, tool_name, was_surfaced,
                    status, duration_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.session_id, self.current_decision_idx(), row.timestamp,
                    row.tool_name, row.was_surfaced, row.status, row.duration_ms,
                ),
            )
