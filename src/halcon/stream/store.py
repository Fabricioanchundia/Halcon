"""Almacén de historial por cliente para calcular features en vivo.

Guarda las compras de los últimos 30 días de cada cliente en un *sorted set* de Valkey
(puntaje = segundo de la compra). Así cada transacción nueva necesita UNA lectura y UNA
escritura, ambas en un solo viaje de red (pipeline).

``InMemoryStore`` hace lo mismo en un diccionario: sirve para pruebas y para medir el costo
del cálculo sin la red.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol

RETENTION_S = 30 * 86_400


@dataclass(frozen=True)
class PastTx:
    ts: int
    tx_id: int
    amount: float
    merchant_id: int


class HistoryStore(Protocol):
    def read(self, customer_id: int, now_ts: int) -> tuple[list[PastTx], int | None]:
        """Compras del cliente con ts >= now - 30 días, y el ts de su última compra (si hubo)."""
        ...

    def write(self, customer_id: int, tx: PastTx) -> None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._hist: dict[int, deque[PastTx]] = defaultdict(deque)
        self._last: dict[int, int] = {}

    def read(self, customer_id: int, now_ts: int) -> tuple[list[PastTx], int | None]:
        hist = self._hist.get(customer_id)
        if hist is None:
            return [], self._last.get(customer_id)
        floor = now_ts - RETENTION_S
        return [p for p in hist if p.ts >= floor], self._last.get(customer_id)

    def write(self, customer_id: int, tx: PastTx) -> None:
        hist = self._hist[customer_id]
        hist.append(tx)
        floor = tx.ts - RETENTION_S
        while hist and hist[0].ts < floor:
            hist.popleft()
        self._last[customer_id] = tx.ts


class ValkeyStore:
    """Historial en Valkey. ``client`` es un ``valkey.Valkey`` (o un falso compatible en tests)."""

    def __init__(self, client: Any, prefix: str = "halcon") -> None:
        self.client = client
        self.prefix = prefix

    def _keys(self, customer_id: int) -> tuple[str, str]:
        return f"{self.prefix}:hist:{customer_id}", f"{self.prefix}:last:{customer_id}"

    def read(self, customer_id: int, now_ts: int) -> tuple[list[PastTx], int | None]:
        hist_key, last_key = self._keys(customer_id)
        pipe = self.client.pipeline(transaction=False)
        pipe.zrangebyscore(hist_key, now_ts - RETENTION_S, "+inf", withscores=True)
        pipe.get(last_key)
        members, last = pipe.execute()
        past = []
        for member, score in members:
            tx_id, amount, merchant = (
                member.decode() if isinstance(member, bytes) else member
            ).split(":")
            past.append(PastTx(int(score), int(tx_id), float(amount), int(merchant)))
        return past, int(last) if last is not None else None

    def write(self, customer_id: int, tx: PastTx) -> None:
        hist_key, last_key = self._keys(customer_id)
        pipe = self.client.pipeline(transaction=False)
        pipe.zadd(hist_key, {f"{tx.tx_id}:{tx.amount!r}:{tx.merchant_id}": tx.ts})
        pipe.zremrangebyscore(hist_key, "-inf", f"({tx.ts - RETENTION_S}")
        pipe.set(last_key, tx.ts)
        pipe.execute()

    def bulk_write(self, rows: list[tuple[int, PastTx]]) -> None:
        """Carga masiva (calentamiento): muchas compras en un solo viaje de red."""
        pipe = self.client.pipeline(transaction=False)
        for customer_id, tx in rows:
            hist_key, last_key = self._keys(customer_id)
            pipe.zadd(hist_key, {f"{tx.tx_id}:{tx.amount!r}:{tx.merchant_id}": tx.ts})
            pipe.set(last_key, tx.ts)
        pipe.execute()

    def flush(self) -> int:
        keys = list(self.client.scan_iter(match=f"{self.prefix}:*", count=10_000))
        for i in range(0, len(keys), 5_000):
            self.client.delete(*keys[i : i + 5_000])
        return len(keys)
