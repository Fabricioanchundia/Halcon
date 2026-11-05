"""Cómo llegan los casos a la cola.

* ``from_topic``: lee el topic ``decisions`` de Redpanda (el flujo real de la Fase 4).
* ``from_test_period``: decide los pagos de los últimos días directamente con el modelo, sin
  Docker. Sirve para trabajar el agente sin levantar la infraestructura; usa exactamente el
  mismo scorer y las mismas features (las del SQL offline, que tienen paridad con las online).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import duckdb
import numpy as np

from halcon.cases.store import CaseStore
from halcon.decision.policy import decide
from halcon.decision.rules import apply_rules
from halcon.decision.scorer import Scorer
from halcon.model.features import FEATURES

DAY = 86_400
Printer = Callable[[str], None]


def from_test_period(raw_dir: Path, features_path: Path, model_dir: Path, store: CaseStore,
                     days: int = 30, max_cases: int | None = None) -> int:  # fmt: skip
    cols = ", ".join(f"f.{name}" for name in FEATURES)
    sql = f"""
        WITH lim AS (SELECT max(ts) - to_days(CAST(? AS INTEGER)) AS since FROM read_parquet(?))
        SELECT f.tx_id, epoch(f.ts)::BIGINT AS ts, f.customer_id, f.merchant_id, t.category,
               t.channel, t.merchant_city, {cols}
        FROM read_parquet(?) f
        JOIN read_parquet(?) t USING (tx_id)
        WHERE f.ts >= (SELECT since FROM lim)
        ORDER BY f.ts, f.tx_id
    """  # noqa: S608 — solo se interpolan nombres de columnas constantes
    fpath = features_path.as_posix()
    data = (
        duckdb.connect()
        .execute(sql, [days, fpath, fpath, (raw_dir / "transactions.parquet").as_posix()])
        .fetchnumpy()
    )

    x = np.column_stack(
        [np.ma.filled(np.ma.asarray(data[f], dtype=float), np.nan) for f in FEATURES]
    )
    scorer = Scorer.load(model_dir)
    scores = scorer.booster.inplace_predict(x)
    rule_level, _ = apply_rules({f: x[:, i] for i, f in enumerate(FEATURES)})
    actions = decide(np.asarray(scores, dtype=float), rule_level, scorer.thresholds)
    flagged = np.flatnonzero(actions > 0)
    if max_cases is not None:
        flagged = flagged[:max_cases]

    added = 0
    for i in flagged:
        feats = {f: (None if np.isnan(x[i, j]) else float(x[i, j])) for j, f in enumerate(FEATURES)}
        d = scorer.score(feats, explain=True)
        added += store.add(
            {
                "tx_id": int(data["tx_id"][i]),
                "ts": int(data["ts"][i]),
                "customer_id": int(data["customer_id"][i]),
                "merchant_id": int(data["merchant_id"][i]),
                "category": str(data["category"][i]),
                "channel": str(data["channel"][i]),
                "merchant_city": str(data["merchant_city"][i]),
                "amount_usd": float(data["amount_usd"][i]),
                "action": d.action.name,
                "score": d.score,
                "rules": d.rules,
                "reasons": [r.__dict__ for r in d.reasons],
                "features": feats,
            }
        )
    return added


def from_topic(bootstrap: str, store: CaseStore, idle_seconds: float = 5.0,
               log: Printer = print) -> int:  # fmt: skip
    from confluent_kafka import Consumer

    from halcon.stream.service import DECISION_TOPIC

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"halcon-cases-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "allow.auto.create.topics": False,
        }
    )
    consumer.subscribe([DECISION_TOPIC])
    added = seen = 0
    last = time.perf_counter()
    while time.perf_counter() - last < idle_seconds:
        msg = consumer.poll(0.5)
        if msg is None:
            continue
        value = msg.value()
        if msg.error() or value is None:
            continue
        last = time.perf_counter()
        seen += 1
        added += store.add(json.loads(value))
    consumer.close()
    log(f"Leídas {seen:,} decisiones; {added} casos nuevos en la cola.")
    return added
