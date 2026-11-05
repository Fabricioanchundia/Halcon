"""Evidencia de un caso: herramientas de SOLO LECTURA que arman los hechos para el agente.

Reglas que se cumplen aquí, no en el prompt:

* **Nada del futuro:** el historial del cliente y del comercio se lee solo ANTES del pago.
* **Nada de etiquetas:** ninguna consulta toca ``is_fraud``, ``fraud_scenario`` ni
  ``fraud_reported_at``. El agente ve lo mismo que vería un analista el día del pago.
* **Solo lectura:** DuckDB abre los Parquet en modo lectura; nada aquí escribe.

Cada pieza de evidencia lleva un código (``E1``…``E7``) para que el agente la cite y podamos
verificar que no inventa. ``E7`` son comparaciones ya calculadas por el código ("vive en X,
compró en Y", "N veces su promedio"): un modelo de 7B redacta bien, pero compara mal. Mejor
que el código haga la aritmética y el modelo explique.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb

from halcon.cases.store import Case
from halcon.decision.rules import RULES

ECUADOR = timedelta(hours=-5)
_DAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
_RULE_TEXT = {r.code: r.description for r in RULES}

# Cómo se lee cada pista del modelo en lenguaje de analista.
FEATURE_TEXT: dict[str, str] = {
    "amount_usd": "monto de la compra (USD)",
    "hour_local": "hora local",
    "is_night": "compra de madrugada (1 = sí)",
    "day_of_week": "día de la semana (1 = lunes)",
    "is_online": "compra en línea (1 = sí)",
    "category_code": "categoría del comercio (código)",
    "is_other_city": "compra presencial fuera de su ciudad (1 = sí)",
    "secs_since_prev_tx": "segundos desde su compra anterior",
    "tx_count_10m": "compras en los últimos 10 minutos",
    "tx_count_1h": "compras en la última hora",
    "tx_count_24h": "compras en las últimas 24 horas",
    "tx_count_7d": "compras en los últimos 7 días",
    "amount_sum_1h": "gasto en la última hora (USD)",
    "amount_sum_24h": "gasto en las últimas 24 horas (USD)",
    "small_tx_count_1h": "compras de $2 o menos en la última hora",
    "avg_amount_30d": "gasto promedio en 30 días (USD)",
    "amount_ratio_30d": "veces que el monto supera su promedio",
    "distinct_merchants_1h": "comercios distintos en la última hora",
}

# Columnas que el agente NUNCA debe ver.
FORBIDDEN_COLUMNS = ("is_fraud", "fraud_scenario", "fraud_reported_at")


@dataclass(frozen=True)
class Evidence:
    code: str
    title: str
    text: str

    def render(self) -> str:
        return f"[{self.code}] {self.title}\n{self.text}"


def _local(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC) + ECUADOR


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _num(x: float) -> str:
    return f"{x:,.2f}" if abs(x - round(x)) > 1e-9 else f"{int(round(x)):,}"


class EvidenceBuilder:
    def __init__(self, raw_dir: Path) -> None:
        self.tx = (raw_dir / "transactions.parquet").as_posix()
        self.customers = (raw_dir / "customers.parquet").as_posix()
        self.con = duckdb.connect()
        # Vista sin etiquetas: es lo único que las consultas de abajo pueden leer.
        self.con.execute(
            f"""CREATE VIEW tx AS SELECT tx_id, epoch(ts)::BIGINT AS ts, customer_id, merchant_id,
                       category, channel, merchant_city, amount_usd
                FROM read_parquet('{self.tx}')"""  # noqa: S608 — ruta interna del proyecto
        )

    def build(self, case: Case) -> list[Evidence]:
        t = case.ts
        local = _local(t)
        ev: list[Evidence] = []

        ev.append(Evidence("E1", "Transacción", "\n".join([
            f"- Monto: {_money(case.amount_usd)}",
            f"- Categoría: {case.category} · canal: {case.channel} · ciudad del comercio: "
            f"{case.merchant_city}",
            f"- Fecha y hora local: {local:%Y-%m-%d %H:%M} ({_DAYS[local.weekday()]})",
        ])))  # fmt: skip

        rules = [f"- {code}: {_RULE_TEXT.get(code, code)}" for code in case.rules] or [
            "- Ninguna regla se disparó."
        ]
        ev.append(Evidence("E2", "Decisión del sistema", "\n".join([
            f"- Acción: {case.action} · puntaje de riesgo del modelo: {case.score:.4f} (0 a 1)",
            *rules,
        ])))  # fmt: skip

        def show(feature: str, value: float) -> str:
            if feature == "category_code":
                return f"categoría del comercio = {case.category}"
            return f"{FEATURE_TEXT.get(feature, feature)} = {_num(value)}"

        reasons = [
            f"- {show(r['feature'], float(r['value']))} "
            f"(empuja el riesgo +{float(r['contribution']):.2f})"
            for r in case.reasons
        ] or ["- Sin pistas destacadas."]
        shown = {r["feature"] for r in case.reasons}
        f = case.features
        context = [
            f"- {show(k, float(v))}"
            for k in (
                "tx_count_10m",
                "small_tx_count_1h",
                "amount_sum_1h",
                "amount_ratio_30d",
                "is_other_city",
            )  # fmt: skip
            if k not in shown and (v := f.get(k)) is not None
        ]
        ev.append(Evidence("E3", "Pistas del modelo (SHAP) y contexto",
                           "\n".join([*reasons, *context])))  # fmt: skip

        prof = self.con.execute(
            """SELECT count(*), avg(amount_usd), max(amount_usd),
                      avg((channel = 'online')::INT), count(DISTINCT merchant_city)
               FROM tx WHERE customer_id = ? AND ts < ? AND ts >= ? - 2592000""",
            [case.customer_id, t, t],
        ).fetchone()
        home = self.con.execute(
            "SELECT home_city FROM read_parquet(?) WHERE customer_id = ?",
            [self.customers, case.customer_id],
        ).fetchone()
        cats = self.con.execute(
            """SELECT category, count(*) n FROM tx
               WHERE customer_id = ? AND ts < ? AND ts >= ? - 2592000
               GROUP BY 1 ORDER BY n DESC, category LIMIT 3""",
            [case.customer_id, t, t],
        ).fetchall()
        n, avg, mx, online, ncities = prof if prof else (0, None, None, None, 0)
        lines = [f"- Ciudad de residencia: {home[0] if home else 'desconocida'}"]
        if n:
            lines += [
                f"- Compras en los 30 días previos: {n}",
                f"- Monto promedio: {_money(avg)} · monto máximo: {_money(mx)}",
                f"- Porcentaje de compras en línea: {round(100 * online)} %",
                f"- Categorías más frecuentes: {', '.join(c for c, _ in cats)}",
            ]
        else:
            lines.append("- Sin compras en los 30 días previos (cliente nuevo o inactivo).")
        ev.append(Evidence("E4", "Perfil del cliente (30 días antes del pago)", "\n".join(lines)))

        recent = self.con.execute(
            """SELECT ts, amount_usd, category, merchant_city FROM tx
               WHERE customer_id = ? AND ts >= ? - 86400
                 AND (ts < ? OR (ts = ? AND tx_id < ?))  -- solo lo que ya había pasado
               ORDER BY ts DESC, tx_id DESC LIMIT 8""",
            [case.customer_id, t, t, t, case.tx_id],
        ).fetchall()
        rec_lines = [
            f"- hace {self._ago(t - int(ts))}: {_money(float(a))} en {c} ({city})"
            for ts, a, c, city in recent
        ] or ["- Ninguna compra en las 24 horas previas."]
        title = "Compras del cliente en las 24 horas previas (más recientes primero)"
        ev.append(Evidence("E5", title, "\n".join(rec_lines)))

        merch = self.con.execute(
            """SELECT count(*), count(DISTINCT customer_id), avg(amount_usd)
               FROM tx WHERE merchant_id = ? AND ts < ? AND ts >= ? - 2592000""",
            [case.merchant_id, t, t],
        ).fetchone()
        mn, mc, mavg = merch if merch else (0, 0, None)
        ev.append(Evidence("E6", "Comercio (30 días antes del pago)", "\n".join([
            f"- Categoría: {case.category} · ciudad: {case.merchant_city}",
            f"- Ventas: {mn} a {mc} clientes distintos"
            + (f" · ticket promedio {_money(mavg)}" if mavg is not None else ""),
        ])))  # fmt: skip

        home_city = home[0] if home else None
        signals: list[str] = []
        ratio, avg30 = f.get("amount_ratio_30d"), f.get("avg_amount_30d")
        if ratio is not None and avg30 is not None:
            signals.append(f"- El monto es {ratio:.1f} veces su promedio de 30 días "
                           f"({_money(float(avg30))}).")  # fmt: skip
        else:
            signals.append("- Sin historial de 30 días para comparar el monto.")
        if n and mx is not None and case.amount_usd > float(mx):
            signals.append(f"- Es mayor que su compra más grande de los 30 días previos "
                           f"({_money(float(mx))}).")  # fmt: skip
        if case.channel == "online":
            signals.append("- Compra en línea (sin ciudad física).")
        elif home_city and case.merchant_city != home_city:
            signals.append(f"- Compra presencial en {case.merchant_city}, distinta de su ciudad "
                           f"de residencia ({home_city}).")  # fmt: skip
        else:
            signals.append(f"- Compra presencial en su misma ciudad de residencia ({home_city}).")
        signals.append(f"- Hora {local:%H:%M}: " + ("de madrugada." if local.hour < 6
                                                     else "horario habitual."))  # fmt: skip
        small = int(f.get("small_tx_count_1h") or 0)
        burst = int(f.get("tx_count_10m") or 1)
        signals.append(f"- Compras de $2 o menos en la última hora, incluida esta: {small}.")
        signals.append(f"- Compras en los últimos 10 minutos, incluida esta: {burst}.")
        ev.append(Evidence("E7", "Comparaciones clave (calculadas por el sistema)",
                           "\n".join(signals)))  # fmt: skip
        return ev

    @staticmethod
    def _ago(seconds: int) -> str:
        if seconds < 120:
            return f"{seconds} s"
        if seconds < 7200:
            return f"{seconds // 60} min"
        return f"{seconds // 3600} h"
