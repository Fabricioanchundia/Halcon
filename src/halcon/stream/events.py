"""Mensajes que viajan por Redpanda.

Una transacción en vivo lleva lo que trae una autorización real (monto, comercio, MCC, canal,
ciudad del comercio). NUNCA lleva etiquetas de fraude: en vivo nadie sabe aún si es fraude.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class TxEvent:
    tx_id: int
    ts: int  # segundos Unix UTC (tiempo del evento, no del reloj de la máquina)
    customer_id: int
    merchant_id: int
    mcc: str
    category: str
    channel: str
    merchant_city: str
    amount_usd: float

    def encode(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def decode(cls, raw: bytes) -> TxEvent:
        return cls(**json.loads(raw))


def read_events(raw_dir: Path, start_ts: int | None = None, end_ts: int | None = None,
                limit: int | None = None) -> Iterator[TxEvent]:  # fmt: skip
    """Lee transacciones del simulador en orden de tiempo, sin columnas de etiquetas."""
    sql = """
        SELECT tx_id, epoch(ts)::BIGINT AS ts, customer_id, merchant_id, mcc, category, channel,
               merchant_city, amount_usd
        FROM read_parquet(?)
        WHERE epoch(ts) >= ? AND epoch(ts) < ?
        ORDER BY ts, tx_id
    """
    params: list[object] = [
        (raw_dir / "transactions.parquet").as_posix(),
        start_ts if start_ts is not None else -(2**62),
        end_ts if end_ts is not None else 2**62,
    ]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    con = duckdb.connect()
    cur = con.execute(sql, params)
    while batch := cur.fetchmany(10_000):
        for row in batch:
            yield TxEvent(*row)
    con.close()


def time_bounds(raw_dir: Path) -> tuple[int, int]:
    row = (
        duckdb.connect()
        .execute(
            "SELECT epoch(min(ts))::BIGINT, epoch(max(ts))::BIGINT FROM read_parquet(?)",
            [(raw_dir / "transactions.parquet").as_posix()],
        )
        .fetchone()
    )
    if row is None or row[0] is None:
        raise ValueError(f"No hay transacciones en {raw_dir}")
    return int(row[0]), int(row[1])


def load_home_cities(raw_dir: Path) -> dict[int, str]:
    rows = (
        duckdb.connect()
        .execute(
            "SELECT customer_id, home_city FROM read_parquet(?)",
            [(raw_dir / "customers.parquet").as_posix()],
        )
        .fetchall()
    )
    return {int(c): str(city) for c, city in rows}
