"""Cola de casos en DuckDB (archivo local ``data/cases.duckdb``).

Cada pago que el motor mandó a REVISAR o RECHAZAR se vuelve un caso. El analista humano lo
cierra como fraude o legítimo. El agente solo agrega un resumen y una recomendación: nunca
cambia el estado de un caso por su cuenta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    tx_id BIGINT PRIMARY KEY,
    ts BIGINT NOT NULL,
    customer_id BIGINT NOT NULL,
    merchant_id BIGINT NOT NULL,
    category VARCHAR,
    channel VARCHAR,
    merchant_city VARCHAR,
    amount_usd DOUBLE,
    action VARCHAR NOT NULL,
    score DOUBLE,
    rules VARCHAR,
    reasons VARCHAR,
    features VARCHAR,
    status VARCHAR NOT NULL DEFAULT 'abierto',
    analyst_note VARCHAR,
    agent_json VARCHAR,
    agent_model VARCHAR,
    agent_ms DOUBLE,
    agent_problems VARCHAR
)
"""


@dataclass
class Case:
    tx_id: int
    ts: int
    customer_id: int
    merchant_id: int
    category: str
    channel: str
    merchant_city: str
    amount_usd: float
    action: str
    score: float
    rules: list[str] = field(default_factory=list)
    reasons: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, float | None] = field(default_factory=dict)
    status: str = "abierto"
    agent: dict[str, Any] | None = None
    agent_model: str | None = None
    agent_ms: float = 0.0
    agent_problems: list[str] = field(default_factory=list)

    @property
    def model_view(self) -> str:
        """Lo que dice el modelo XGBoost + reglas, en el idioma del agente."""
        fraud = self.score >= 0.5 or "R001" in self.rules
        return "fraude_probable" if fraud else "legitimo_probable"


class CaseStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(path.as_posix())
        self.con.execute(_SCHEMA)

    def close(self) -> None:
        self.con.close()

    def add(self, decision: dict[str, Any]) -> bool:
        """Agrega un caso desde un mensaje de decisión. Ignora aprobados y duplicados."""
        if decision.get("action") == "APPROVE":
            return False
        exists = self.con.execute(
            "SELECT 1 FROM cases WHERE tx_id = ?", [decision["tx_id"]]
        ).fetchone()
        if exists:
            return False
        self.con.execute(
            """INSERT INTO cases (tx_id, ts, customer_id, merchant_id, category, channel,
               merchant_city, amount_usd, action, score, rules, reasons, features)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                decision["tx_id"],
                decision["ts"],
                decision["customer_id"],
                decision["merchant_id"],
                decision.get("category"),
                decision.get("channel"),
                decision.get("merchant_city"),
                decision["amount_usd"],
                decision["action"],
                decision["score"],
                json.dumps(decision.get("rules", [])),
                json.dumps(decision.get("reasons", [])),
                json.dumps(decision.get("features", {})),
            ],
        )
        return True

    def _row_to_case(self, row: tuple[Any, ...], names: list[str]) -> Case:
        d = dict(zip(names, row, strict=True))
        return Case(
            tx_id=int(d["tx_id"]),
            ts=int(d["ts"]),
            customer_id=int(d["customer_id"]),
            merchant_id=int(d["merchant_id"]),
            category=str(d["category"]),
            channel=str(d["channel"]),
            merchant_city=str(d["merchant_city"]),
            amount_usd=float(d["amount_usd"]),
            action=str(d["action"]),
            score=float(d["score"]),
            rules=json.loads(d["rules"] or "[]"),
            reasons=json.loads(d["reasons"] or "[]"),
            features=json.loads(d["features"] or "{}"),
            status=str(d["status"]),
            agent=json.loads(d["agent_json"]) if d["agent_json"] else None,
            agent_model=d["agent_model"],
            agent_ms=float(d["agent_ms"] or 0.0),
            agent_problems=json.loads(d["agent_problems"] or "[]"),
        )

    def _query(self, sql: str, params: list[Any]) -> list[Case]:
        cur = self.con.execute(sql, params)
        names = [c[0] for c in cur.description or []]
        return [self._row_to_case(r, names) for r in cur.fetchall()]

    def get(self, tx_id: int) -> Case | None:
        found = self._query("SELECT * FROM cases WHERE tx_id = ?", [tx_id])
        return found[0] if found else None

    def list_cases(self, status: str | None = "abierto", limit: int = 20,
             without_agent: bool = False) -> list[Case]:  # fmt: skip
        sql = "SELECT * FROM cases WHERE 1=1"
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        if without_agent:
            sql += " AND agent_json IS NULL"
        sql += " ORDER BY score DESC, ts LIMIT ?"
        params.append(limit)
        return self._query(sql, params)

    def all_ids(self) -> list[int]:
        return [int(r[0]) for r in self.con.execute("SELECT tx_id FROM cases").fetchall()]

    def save_agent(self, tx_id: int, result: dict[str, Any], model: str, ms: float,
                   problems: list[str]) -> None:  # fmt: skip
        self.con.execute(
            """UPDATE cases SET agent_json = ?, agent_model = ?, agent_ms = ?, agent_problems = ?
               WHERE tx_id = ?""",
            [json.dumps(result, ensure_ascii=False), model, ms, json.dumps(problems), tx_id],
        )

    def resolve(self, tx_id: int, status: str, note: str | None = None) -> None:
        if status not in {"fraude", "legitimo"}:
            raise ValueError("status debe ser 'fraude' o 'legitimo'")
        self.con.execute(
            "UPDATE cases SET status = ?, analyst_note = ? WHERE tx_id = ?", [status, note, tx_id]
        )

    def counts(self) -> dict[str, int]:
        rows = self.con.execute("SELECT status, count(*) FROM cases GROUP BY status").fetchall()
        return {str(s): int(n) for s, n in rows}
