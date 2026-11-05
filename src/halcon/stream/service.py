"""Flujo en tiempo real: productor → Redpanda → scorer → Redpanda.

::

    simulador ──► topic "transactions" ──► scorer (features en Valkey + modelo + reglas)
                  (clave = cliente)                         │
                                                            ▼
                                                 topic "decisions" ──► cola de casos (Fase 5)

* La clave del mensaje es el cliente: Redpanda garantiza el orden dentro de una partición, y
  todas las compras de un cliente caen en la misma. Las features de una tarjeta solo dependen
  de su propio historial, así que el orden que importa se conserva.
* El tiempo de las features es el del evento (``ts``), no el reloj de la máquina. Por eso se
  puede reproducir un mes en minutos y obtener exactamente las mismas features que offline.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from halcon.decision.scorer import Scorer
from halcon.stream.events import TxEvent, load_home_cities, read_events, time_bounds
from halcon.stream.online import OnlineFeatures
from halcon.stream.store import RETENTION_S, InMemoryStore, PastTx, ValkeyStore

TX_TOPIC = "transactions"
DECISION_TOPIC = "decisions"
DAY = 86_400

Printer = Callable[[str], None]


def connect_valkey(url: str) -> Any:
    import valkey

    client = valkey.from_url(url)  # type: ignore[no-untyped-call]
    client.ping()
    return client


def default_stream_start(raw_dir: Path, test_days: int = 30) -> int:
    """Por defecto se transmiten los últimos 30 días: el periodo que el modelo nunca vio."""
    lo, hi = time_bounds(raw_dir)
    start_day = lo // DAY * DAY
    total_days = (hi - start_day) // DAY + 1
    return start_day + (total_days - test_days) * DAY


# --------------------------------------------------------------------------- preparación


def init_topics(bootstrap: str, reset: bool = False, partitions: int = 3) -> list[str]:
    from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

    admin = AdminClient({"bootstrap.servers": bootstrap})
    if reset:
        existing = set(admin.list_topics(timeout=10).topics)
        to_delete = [t for t in (TX_TOPIC, DECISION_TOPIC) if t in existing]
        if to_delete:
            for fut in admin.delete_topics(to_delete).values():
                fut.result()
            time.sleep(2)  # el borrado de topics es asíncrono en el broker
    if reset:
        try:
            for fut in admin.delete_consumer_groups(["halcon-scorer"]).values():
                fut.result()
        except Exception:  # noqa: BLE001, S110 — si el grupo no existe, no hay nada que borrar
            pass
    created = []
    new = [NewTopic(TX_TOPIC, partitions, 1), NewTopic(DECISION_TOPIC, 1, 1)]
    for topic, fut in admin.create_topics(new).items():
        try:
            fut.result()
            created.append(topic)
        except Exception as exc:  # noqa: BLE001 — "ya existe" no es un error aquí
            if "TOPIC_ALREADY_EXISTS" not in str(exc):
                raise
    return created


def warmup(raw_dir: Path, store: ValkeyStore, start_ts: int, reset: bool = True) -> int:
    """Carga en Valkey el historial previo al inicio del flujo (*backfill* del feature store).

    En un banco real el feature store ya tendría meses de historia cuando se enciende el
    scorer. Aquí lo reconstruimos: 30 días de compras por cliente + la fecha de su última compra.
    """
    if reset:
        store.flush()
    con = duckdb.connect()
    path = (raw_dir / "transactions.parquet").as_posix()
    rows = con.execute(
        """SELECT customer_id, epoch(ts)::BIGINT, tx_id, amount_usd, merchant_id
           FROM read_parquet(?) WHERE epoch(ts) >= ? AND epoch(ts) < ? ORDER BY ts, tx_id""",
        [path, start_ts - RETENTION_S, start_ts],
    ).fetchall()
    batch: list[tuple[int, PastTx]] = []
    for cust, ts, tx_id, amount, merchant in rows:
        batch.append((int(cust), PastTx(int(ts), int(tx_id), float(amount), int(merchant))))
        if len(batch) >= 20_000:
            store.bulk_write(batch)
            batch.clear()
    if batch:
        store.bulk_write(batch)
    # Última compra de cada cliente, aunque haya sido hace más de 30 días.
    last = con.execute(
        """SELECT customer_id, epoch(max(ts))::BIGINT FROM read_parquet(?)
           WHERE epoch(ts) < ? GROUP BY customer_id""",
        [path, start_ts],
    ).fetchall()
    pipe = store.client.pipeline(transaction=False)
    for cust, ts in last:
        pipe.set(f"{store.prefix}:last:{int(cust)}", int(ts))
    pipe.execute()
    return len(rows)


# --------------------------------------------------------------------------- productor


def produce(raw_dir: Path, bootstrap: str, start_ts: int, limit: int | None, rate: float,
            log: Printer = print) -> int:  # fmt: skip
    """Publica transacciones en orden de tiempo a ``rate`` mensajes por segundo (0 = sin freno)."""
    from confluent_kafka import Producer

    producer = Producer(
        {"bootstrap.servers": bootstrap, "linger.ms": 5, "enable.idempotence": True}
    )
    t0 = time.perf_counter()
    sent = 0
    for ev in read_events(raw_dir, start_ts=start_ts, limit=limit):
        while True:
            try:
                producer.produce(
                    TX_TOPIC,
                    key=str(ev.customer_id),
                    value=ev.encode(),
                    headers={"sent_at_ns": str(time.time_ns())},
                )
                break
            except BufferError:  # cola local llena: esperar a que el broker confirme
                producer.poll(0.05)
        sent += 1
        producer.poll(0)
        if rate > 0 and sent % 10 == 0:  # ráfagas chicas: el flujo se parece más al real
            ahead = sent / rate - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)
        if sent % 10_000 == 0:
            log(f"  enviadas {sent:,} ({sent / (time.perf_counter() - t0):,.0f}/s)")
    producer.flush(30)
    elapsed = time.perf_counter() - t0
    log(f"Productor: {sent:,} transacciones en {elapsed:.1f} s ({sent / elapsed:,.0f}/s)")
    return sent


# --------------------------------------------------------------------------- scorer en vivo


@dataclass
class StreamStats:
    """Métricas del scorer.

    Las líneas de progreso muestran la ÚLTIMA ventana de 5 s, no el acumulado: un promedio
    acumulado esconde una pausa de 30 s dentro de un minuto bueno. El resumen final sí es global.
    """

    processed: int = 0
    actions: Counter[str] = field(default_factory=Counter)
    proc_ms: list[float] = field(default_factory=list)
    e2e_ms: list[float] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)
    last_at: float = field(default_factory=time.perf_counter)
    _mark: tuple[float, int, int, int] = (0.0, 0, 0, 0)  # (tiempo, procesados, idx proc, idx e2e)

    def record(self, action: str, proc_ms: float, e2e_ms: float | None = None) -> None:
        if self.processed == 0:
            self.started = time.perf_counter()
            self._mark = (self.started, 0, 0, 0)
        self.processed += 1
        self.actions[action] += 1
        self.proc_ms.append(proc_ms)
        if e2e_ms is not None:
            self.e2e_ms.append(e2e_ms)
        self.last_at = time.perf_counter()

    @staticmethod
    def _fmt(n: int, seconds: float, proc: list[float], e2e: list[float]) -> str:
        p = np.percentile
        text = f"{n / max(1e-9, seconds):>6,.0f} tx/s · "
        if proc:
            text += f"proceso p50 {p(proc, 50):.2f} / p99 {p(proc, 99):.2f} ms · "
        if e2e:
            text += f"extremo a extremo p50 {p(e2e, 50):.1f} / p99 {p(e2e, 99):.1f} ms · "
        return text

    def window_line(self) -> str:
        """Últimos segundos desde la línea anterior."""
        now = time.perf_counter()
        t0, n0, i0, j0 = self._mark
        text = f"{self.processed:>8,} tx · últimos {now - t0:.0f} s: " + self._fmt(
            self.processed - n0, now - t0, self.proc_ms[i0:], self.e2e_ms[j0:]
        )
        self._mark = (now, self.processed, len(self.proc_ms), len(self.e2e_ms))
        return text.rstrip(" ·")

    def line(self) -> str:
        """Resumen global (sin contar la espera final sin mensajes)."""
        busy = self.last_at - self.started
        return (
            f"{self.processed:>8,} tx · "
            + self._fmt(self.processed, busy, self.proc_ms, self.e2e_ms)
            + " ".join(f"{k}={v}" for k, v in sorted(self.actions.items()))
        )

    def to_dict(self) -> dict[str, float | int | dict[str, int]]:
        busy = max(1e-9, self.last_at - self.started)
        proc = self.proc_ms or [0.0]
        e2e = self.e2e_ms or [0.0]
        return {
            "processed": self.processed,
            "tx_per_s": round(self.processed / busy, 1),
            "proc_p50_ms": round(float(np.percentile(proc, 50)), 3),
            "proc_p99_ms": round(float(np.percentile(proc, 99)), 3),
            "e2e_p50_ms": round(float(np.percentile(e2e, 50)), 1),
            "e2e_p99_ms": round(float(np.percentile(e2e, 99)), 1),
            "actions": dict(self.actions),
        }


class StreamScorer:
    """Procesa un evento: features online → decisión → mensaje de salida. Sin red propia."""

    def __init__(self, online: OnlineFeatures, scorer: Scorer) -> None:
        self.online = online
        self.scorer = scorer

    def handle(self, ev: TxEvent) -> tuple[dict[str, Any], float]:
        t0 = time.perf_counter()
        feats = self.online.process(ev)
        d = self.scorer.score(feats)
        out: dict[str, Any] = {
            "tx_id": ev.tx_id,
            "ts": ev.ts,
            "customer_id": ev.customer_id,
            "merchant_id": ev.merchant_id,
            "category": ev.category,
            "channel": ev.channel,
            "merchant_city": ev.merchant_city,
            "amount_usd": ev.amount_usd,
            "action": d.action.name,
            "score": d.score,
            "rules": d.rules,
            "reasons": [r.__dict__ for r in d.reasons],
        }
        if d.action.name != "APPROVE":
            out["features"] = feats  # el analista (y el agente de la Fase 5) las necesita
        return out, (time.perf_counter() - t0) * 1000


def run_scorer(bootstrap: str, valkey_url: str, raw_dir: Path, model_dir: Path,
               max_messages: int | None, idle_seconds: float, log: Printer = print,
               group: str = "halcon-scorer") -> StreamStats:  # fmt: skip
    from confluent_kafka import Consumer, Producer

    store = ValkeyStore(connect_valkey(valkey_url))
    handler = StreamScorer(OnlineFeatures(store, load_home_cities(raw_dir)),
                           Scorer.load(model_dir))  # fmt: skip
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
        }
    )
    producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 5})
    consumer.subscribe([TX_TOPIC])
    stats = StreamStats()
    last_msg = time.perf_counter()
    last_log = time.perf_counter()
    log("Scorer escuchando… (Ctrl+C para parar)")
    try:
        while max_messages is None or stats.processed < max_messages:
            # poll() entrega cada mensaje apenas llega. consume(n) esperaría a juntar n mensajes
            # (o a su timeout) y eso sumaría cientos de milisegundos a la latencia.
            msg = consumer.poll(0.2)
            now = time.perf_counter()
            if now - last_log > 5 and stats.processed:
                log("  " + stats.window_line())
                last_log = now
            if msg is None:
                if stats.processed and now - last_msg > idle_seconds:
                    break
                continue
            last_msg = now
            if msg.error():
                log(f"  error del broker: {msg.error()}")
                continue
            raw = msg.value()
            if raw is None:
                continue
            ev = TxEvent.decode(raw)
            out, proc_ms = handler.handle(ev)
            producer.produce(DECISION_TOPIC, key=str(ev.customer_id),
                             value=json.dumps(out, separators=(",", ":")).encode())  # fmt: skip
            producer.poll(0)
            headers = dict(msg.headers() or [])
            sent = headers.get("sent_at_ns")  # mismo reloj: productor y scorer en la laptop
            e2e = (time.time_ns() - int(sent)) / 1e6 if isinstance(sent, bytes | str) else None
            stats.record(out["action"], proc_ms, e2e)
    except KeyboardInterrupt:
        log("Detenido por el usuario.")
    finally:
        producer.flush(10)
        consumer.close()
    log("Final: " + stats.line())
    return stats


def bench(raw_dir: Path, model_dir: Path, n: int, valkey_url: str | None,
          log: Printer = print) -> StreamStats:  # fmt: skip
    """Mide el scorer sin Redpanda: cuánto cuesta cada transacción en CPU (+ Valkey si se pide)."""
    start = default_stream_start(raw_dir)
    if valkey_url:
        vstore = ValkeyStore(connect_valkey(valkey_url), prefix="halcon-bench")
        warmup(raw_dir, vstore, start)
        store: InMemoryStore | ValkeyStore = vstore
    else:
        mem = InMemoryStore()
        for ev in read_events(raw_dir, start_ts=start - RETENTION_S, end_ts=start):
            mem.write(ev.customer_id, PastTx(ev.ts, ev.tx_id, ev.amount_usd, ev.merchant_id))
        store = mem
    handler = StreamScorer(OnlineFeatures(store, load_home_cities(raw_dir)),
                           Scorer.load(model_dir))  # fmt: skip
    stats = StreamStats()
    for ev in read_events(raw_dir, start_ts=start, limit=n):
        out, proc_ms = handler.handle(ev)
        stats.record(out["action"], proc_ms)
    if isinstance(store, ValkeyStore):
        store.flush()
    log("Bench: " + stats.line())
    return stats


# --------------------------------------------------------------------------- resumen


def summarize(bootstrap: str, raw_dir: Path, idle_seconds: float = 5.0,
              log: Printer = print) -> dict[str, Any]:  # fmt: skip
    """Lee todas las decisiones y las cruza con la verdad del simulador (solo para la demo)."""
    import uuid

    from confluent_kafka import Consumer

    consumer = Consumer({"bootstrap.servers": bootstrap, "group.id": f"summary-{uuid.uuid4()}",
                         "auto.offset.reset": "earliest", "enable.auto.commit": False})  # fmt: skip
    consumer.subscribe([DECISION_TOPIC])
    rows: list[tuple[int, str, float]] = []
    last = time.perf_counter()
    while time.perf_counter() - last < idle_seconds:
        for msg in consumer.consume(num_messages=1000, timeout=0.5):
            value = msg.value()
            if msg.error() or value is None:
                continue
            d = json.loads(value)
            rows.append((int(d["tx_id"]), str(d["action"]), float(d["amount_usd"])))
            last = time.perf_counter()
    consumer.close()
    if not rows:
        log("No hay decisiones todavía.")
        return {}
    con = duckdb.connect()
    con.execute("CREATE TABLE d(tx_id BIGINT, action VARCHAR, amount DOUBLE)")
    con.executemany("INSERT INTO d VALUES (?, ?, ?)", rows)
    result = con.execute(
        """SELECT d.action,
                  count(*) AS n,
                  count(*) FILTER (WHERE t.is_fraud) AS frauds,
                  round(sum(d.amount) FILTER (WHERE t.is_fraud), 2) AS fraud_usd
           FROM d JOIN read_parquet(?) t USING (tx_id)
           GROUP BY d.action ORDER BY d.action""",
        [(raw_dir / "transactions.parquet").as_posix()],
    ).fetchall()
    total_fraud = sum(r[2] for r in result) or 1
    log(f"{len(rows):,} decisiones en el topic '{DECISION_TOPIC}':")
    log(f"  {'acción':<10}{'transacciones':>15}{'fraudes':>10}{'$ fraude':>14}")
    for action, n, frauds, usd in result:
        log(f"  {action:<10}{n:>15,}{frauds:>10}{(usd or 0):>14,.2f}")
    caught = sum(r[2] for r in result if r[0] != "APPROVE")
    log(f"  Fraudes detenidos o en revisión: {caught} de {total_fraud} "
        f"({100 * caught / total_fraud:.1f} %)")  # fmt: skip
    return {r[0]: {"n": r[1], "frauds": r[2]} for r in result}
