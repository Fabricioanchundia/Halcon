"""Exporta el resultado de la simulación a Parquet (tres tablas, como en un banco real).

* ``transactions.parquet``: lo que llega en cada autorización (monto, comercio, MCC, canal,
  ciudad del comercio) más las etiquetas de fraude.
* ``customers.parquet``: datos del cliente que el banco conoce (ciudad de residencia).
* ``merchants.parquet``: catálogo de comercios.

Las propiedades ocultas de los clientes (gasto promedio, frecuencia) NO se exportan: el modelo
debe aprenderlas del historial, igual que en la vida real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from halcon.simulator.catalog import CATEGORIES, CITIES, ONLINE_CITY
from halcon.simulator.engine import SCENARIOS, SimResult

_CITY_NAMES = np.array([c.name for c in CITIES] + [ONLINE_CITY])  # índice -1 → ONLINE
_CAT_NAMES = np.array([c.name for c in CATEGORIES])
_CAT_MCC = np.array([c.mcc for c in CATEGORIES])
_CAT_CHANNEL = np.array([c.channel for c in CATEGORIES])
_SCENARIO_NAMES = np.array([None, *SCENARIOS[1:]], dtype=object)
_TS = pa.timestamp("s", tz="UTC")


def _timestamps(seconds: np.ndarray, null_mask: np.ndarray | None = None) -> pa.Array:
    return pa.array(seconds.astype("datetime64[s]"), type=_TS, mask=null_mask)


def write_parquet(result: SimResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tx, cust, merch = result.transactions, result.customers, result.merchants
    m_cat = merch.category[tx.merchant]

    transactions = pa.table(
        {
            "tx_id": pa.array(np.arange(len(tx), dtype=np.int64)),
            "ts": _timestamps(tx.ts),
            "customer_id": pa.array(tx.customer.astype(np.int32)),
            "merchant_id": pa.array(tx.merchant.astype(np.int32)),
            "mcc": pa.array(_CAT_MCC[m_cat]),
            "category": pa.array(_CAT_NAMES[m_cat]),
            "channel": pa.array(_CAT_CHANNEL[m_cat]),
            "merchant_city": pa.array(_CITY_NAMES[merch.city[tx.merchant]]),
            "amount_usd": pa.array(tx.amount),
            "is_fraud": pa.array(tx.scenario > 0),
            "fraud_scenario": pa.array(_SCENARIO_NAMES[tx.scenario], type=pa.string()),
            "fraud_reported_at": _timestamps(tx.reported_at, null_mask=tx.reported_at < 0),
        }
    )
    customers = pa.table(
        {
            "customer_id": pa.array(cust.customer_id.astype(np.int32)),
            "home_city": pa.array(_CITY_NAMES[cust.home_city]),
        }
    )
    merchants = pa.table(
        {
            "merchant_id": pa.array(merch.merchant_id.astype(np.int32)),
            "category": pa.array(_CAT_NAMES[merch.category]),
            "mcc": pa.array(_CAT_MCC[merch.category]),
            "channel": pa.array(_CAT_CHANNEL[merch.category]),
            "city": pa.array(_CITY_NAMES[merch.city]),
        }
    )
    paths = {
        "transactions": out_dir / "transactions.parquet",
        "customers": out_dir / "customers.parquet",
        "merchants": out_dir / "merchants.parquet",
    }
    pq.write_table(transactions, paths["transactions"], compression="zstd")
    pq.write_table(customers, paths["customers"], compression="zstd")
    pq.write_table(merchants, paths["merchants"], compression="zstd")
    return paths
