"""Features en vivo: el mismo cálculo que ``model/features.py`` (SQL), pero de a una transacción.

Si las dos versiones difieren aunque sea un poco, el modelo recibe en producción números
distintos a los que vio al entrenar (*training-serving skew*) y falla en silencio. Por eso
``tests/test_stream.py`` compara ambas versiones fila por fila: deben dar exactamente lo mismo.
(También se verificó a mano sobre las 994 146 transacciones de la simulación por defecto.)

Convención compartida con el SQL: cada ventana incluye la transacción actual y las ANTERIORES
de su mismo cliente, sin contar otras del mismo segundo. El promedio de 30 días excluye la
actual.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from halcon.simulator.catalog import CATEGORIES
from halcon.stream.events import TxEvent
from halcon.stream.store import HistoryStore, PastTx

ECUADOR = timedelta(hours=-5)
SMALL_AMOUNT = 2.0
NO_PREVIOUS_TX = 2_592_000  # mismo valor por defecto que el SQL (30 días)
_CATEGORY_CODE = {c.name: i + 1 for i, c in enumerate(CATEGORIES)}  # list_position empieza en 1

FeatureRow = dict[str, float | None]


class OnlineFeatures:
    def __init__(self, store: HistoryStore, home_city: dict[int, str]) -> None:
        self.store = store
        self.home_city = home_city

    def compute(self, ev: TxEvent, history: list[PastTx], last_ts: int | None) -> FeatureRow:
        t, amount = ev.ts, ev.amount_usd
        prev = [p for p in history if p.ts < t]  # sin otras compras del mismo segundo

        def count(window: int) -> float:
            return 1.0 + sum(1 for p in prev if p.ts >= t - window)

        def total(window: int) -> float:
            return amount + sum(p.amount for p in prev if p.ts >= t - window)

        last_30d = [p.amount for p in prev if p.ts >= t - 2_592_000]
        avg_30d = sum(last_30d) / len(last_30d) if last_30d else None
        local = datetime.fromtimestamp(t, tz=UTC) + ECUADOR
        merchants_1h = {p.merchant_id for p in prev if p.ts >= t - 3_600} | {ev.merchant_id}
        home = self.home_city.get(ev.customer_id)

        row: FeatureRow = {
            "amount_usd": amount,
            "hour_local": float(local.hour),
            "is_night": float(local.hour < 6),
            "day_of_week": float(local.isoweekday()),
            "is_online": float(ev.channel == "online"),
            "category_code": float(_CATEGORY_CODE[ev.category]),
            "is_other_city": float(ev.channel == "pos" and ev.merchant_city != home),
            "secs_since_prev_tx": float(t - last_ts) if last_ts is not None else NO_PREVIOUS_TX,
            "tx_count_10m": count(600),
            "tx_count_1h": count(3_600),
            "tx_count_24h": count(86_400),
            "tx_count_7d": count(604_800),
            "amount_sum_1h": total(3_600),
            "amount_sum_24h": total(86_400),
            "small_tx_count_1h": float(amount <= SMALL_AMOUNT)
            + sum(1 for p in prev if p.ts >= t - 3_600 and p.amount <= SMALL_AMOUNT),
            "avg_amount_30d": avg_30d,
            "amount_ratio_30d": amount / avg_30d if avg_30d else None,
            "distinct_merchants_1h": float(len(merchants_1h)),
        }
        return row

    def process(self, ev: TxEvent) -> FeatureRow:
        """Lee el historial, calcula las features y guarda la transacción (2 viajes a Valkey)."""
        history, last_ts = self.store.read(ev.customer_id, ev.ts)
        row = self.compute(ev, history, last_ts)
        self.store.write(ev.customer_id, PastTx(ev.ts, ev.tx_id, ev.amount_usd, ev.merchant_id))
        return row
