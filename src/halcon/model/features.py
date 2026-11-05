"""Ingeniería de features con DuckDB.

Regla de oro: cada feature de una transacción usa SOLO información disponible en el instante de
la autorización (la transacción actual y las anteriores). Nada del futuro y nada de etiquetas.
Eso evita la "fuga de datos" (*data leakage*), el error más común en proyectos de fraude: un
modelo que en el notebook parece perfecto y en producción falla.

Las ventanas de tiempo son las mismas que se calcularán en vivo en la Fase 4 (streaming), donde
una prueba de paridad comprobará que ambas versiones dan los mismos números.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

# Orden fijo de columnas que entran al modelo.
FEATURES: tuple[str, ...] = (
    "amount_usd",
    "hour_local",
    "is_night",
    "day_of_week",
    "is_online",
    "category_code",
    "is_other_city",
    "secs_since_prev_tx",
    "tx_count_10m",
    "tx_count_1h",
    "tx_count_24h",
    "tx_count_7d",
    "amount_sum_1h",
    "amount_sum_24h",
    "small_tx_count_1h",
    "avg_amount_30d",
    "amount_ratio_30d",
    "distinct_merchants_1h",
)

LABEL_COLUMNS: tuple[str, ...] = ("is_fraud", "fraud_scenario", "fraud_reported_at")

# Las ventanas usan RANGE sobre el tiempo; "CURRENT ROW" incluye la transacción actual, que el
# banco ya conoce al momento de decidir. El promedio de 30 días EXCLUYE la actual para que el
# ratio compare contra el comportamiento previo del cliente.
_SQL = """
WITH base AS (
    SELECT
        t.*,
        c.home_city,
        epoch(t.ts)::BIGINT AS ts_s
    FROM read_parquet('{raw}/transactions.parquet') t
    JOIN read_parquet('{raw}/customers.parquet') c USING (customer_id)
),
w AS (
    SELECT
        *,
        count(*) OVER (cust RANGE BETWEEN 600 PRECEDING AND CURRENT ROW) AS tx_count_10m,
        count(*) OVER (cust RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) AS tx_count_1h,
        count(*) OVER (cust RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW) AS tx_count_24h,
        count(*) OVER (cust RANGE BETWEEN 604800 PRECEDING AND CURRENT ROW) AS tx_count_7d,
        sum(amount_usd) OVER (cust RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW)
            AS amount_sum_1h,
        sum(amount_usd) OVER (cust RANGE BETWEEN 86400 PRECEDING AND CURRENT ROW)
            AS amount_sum_24h,
        count(*) FILTER (WHERE amount_usd <= 2.0)
            OVER (cust RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW) AS small_tx_count_1h,
        avg(amount_usd) OVER (cust RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING)
            AS avg_amount_30d,
        ts_s - lag(ts_s) OVER (PARTITION BY customer_id ORDER BY ts_s, tx_id)
            AS secs_since_prev_tx
    FROM base
    WINDOW cust AS (PARTITION BY customer_id ORDER BY ts_s)
),
m AS (
    -- Comercios distintos en la última hora (DuckDB no admite DISTINCT en ventanas RANGE).
    SELECT a.tx_id, count(DISTINCT b.merchant_id) AS distinct_merchants_1h
    FROM base a
    JOIN base b
      ON a.customer_id = b.customer_id
     AND b.ts_s BETWEEN a.ts_s - 3600 AND a.ts_s
    GROUP BY a.tx_id
)
SELECT
    w.tx_id,
    w.ts,
    w.customer_id,
    w.merchant_id,
    w.category,
    w.amount_usd,
    hour(w.ts AT TIME ZONE 'America/Guayaquil') AS hour_local,
    (hour(w.ts AT TIME ZONE 'America/Guayaquil') < 6)::INTEGER AS is_night,
    isodow(w.ts AT TIME ZONE 'America/Guayaquil') AS day_of_week,
    (w.channel = 'online')::INTEGER AS is_online,
    list_position({categories}, w.category) AS category_code,
    (w.channel = 'pos' AND w.merchant_city <> w.home_city)::INTEGER AS is_other_city,
    coalesce(w.secs_since_prev_tx, 2592000) AS secs_since_prev_tx,
    w.tx_count_10m,
    w.tx_count_1h,
    w.tx_count_24h,
    w.tx_count_7d,
    w.amount_sum_1h,
    w.amount_sum_24h,
    w.small_tx_count_1h,
    w.avg_amount_30d,
    w.amount_usd / nullif(w.avg_amount_30d, 0) AS amount_ratio_30d,
    m.distinct_merchants_1h,
    w.is_fraud,
    w.fraud_scenario,
    w.fraud_reported_at
FROM w JOIN m USING (tx_id)
ORDER BY w.ts, w.tx_id
"""


def build_features(raw_dir: Path, out_path: Path, threads: int | None = None) -> int:
    """Lee los Parquet del simulador y escribe un Parquet de features. Devuelve las filas."""
    from halcon.simulator.catalog import CATEGORIES

    categories = "[" + ", ".join(f"'{c.name}'" for c in CATEGORIES) + "]"
    sql = _SQL.format(raw=raw_dir.as_posix(), categories=categories)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit = '2GB'")
    con.execute("SET enable_progress_bar = false")
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    con.execute(f"COPY ({sql}) TO '{out_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    row = con.execute("SELECT count(*) FROM read_parquet(?)", [out_path.as_posix()]).fetchone()
    con.close()
    return int(row[0]) if row else 0
