"""Drift y reentrenamiento: ¿cuánto tarda Halcón en aprender un fraude nuevo SOLO con el modelo?

Historia que se simula:

1. Hasta el día ``drift_day`` el card testing es el de siempre (pruebas de $0,50–$2).
2. Ese día los atacantes descubren R001 (pruebas de $2 o menos) y pasan a $2,01–$3. Como les
   funciona, escalan el ataque (4 veces más víctimas).
3. El banco **no tiene R005** (todavía nadie escribió esa regla) y reentrena el modelo cada
   semana con las etiquetas que ya le llegaron. Los reclamos tardan ~15 días en promedio.

Se compara, semana a semana:

* **congelado**: el modelo del día del ataque, sin reentrenar nunca.
* **estándar**: modelo nuevo cada semana con la receta de la Fase 2: un hueco de 14 días para
  que maduren las etiquetas y 14 días de validación para calibrar los umbrales.
* **hueco corto**: la misma receta con un hueco de 3 días. Ve antes el ataque nuevo, pero la
  mayoría de esos fraudes AÚN no tiene reclamo: entran al entrenamiento como "legítimos".
* **+ reclamos frescos**: la receta estándar más SOLO los fraudes ya confirmados del hueco.
  Aprovecha lo nuevo sin enseñarle al modelo que el fraude sin reportar es legítimo.

Y un monitoreo sin etiquetas (``monitor.py``) que dice qué alarma suena primero.

Pérdida: se cuenta lo aprobado (y el 15 % de lo revisado, que se escapa) HASTA que una compra
de la tarjeta se rechaza; ahí el banco bloquea la tarjeta y el resto del ataque no ocurre.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt

from halcon.decision.policy import CostModel, Thresholds, calibrate, decide
from halcon.decision.rules import RULES, Action
from halcon.drift.monitor import MONITORS, DayStatus, daily_monitor, first_alarm
from halcon.model.features import FEATURES, build_features
from halcon.model.train import _fit, _load, _matrix, pick_device
from halcon.simulator import SimConfig, simulate, write_parquet

DAY = 86_400
Printer = Callable[[str], None]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class Recipe:
    """Ventanas de reentrenamiento, contadas hacia atrás desde el día en que se reentrena."""

    name: str
    gap_days: int = 14
    valid_days: int = 14
    train_days: int = 56
    fresh_frauds: bool = False  # suma los fraudes YA reportados del hueco (solo positivos)


STANDARD = Recipe("estándar")
SHORT_GAP = Recipe("hueco corto", gap_days=3)
FRESH = Recipe("+ reclamos frescos", fresh_frauds=True)
RECIPES = (STANDARD, SHORT_GAP, FRESH)
FROZEN = "congelado"


@dataclass(frozen=True)
class WeekResult:
    week: int  # semana desde el inicio del ataque (1 = la primera)
    start_day: int
    attacks: int  # tarjetas con cobro grande esa semana
    new_attack_labels: dict[str, int]  # pagos del ataque nuevo que el modelo vio como fraude
    blocked_pct: dict[str, float]  # tarjetas bloqueadas antes del cobro grande
    usd_lost: dict[str, float]  # $ perdido en card testing
    false_declines: dict[str, int]  # clientes honestos rechazados esa semana


@dataclass
class DriftReport:
    device: str
    drift_day: int
    weeks: list[WeekResult] = field(default_factory=list)
    days: list[DayStatus] = field(default_factory=list)
    first_alarm_day: dict[str, int | None] = field(default_factory=dict)
    false_alarm_days: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def variants(self) -> list[str]:
        return [FROZEN, *(r.name for r in RECIPES)]

    def learned_week(self, variant: str, threshold: float = 50.0) -> int | None:
        """Primera semana en que bloquea al menos ``threshold`` % de las tarjetas atacadas."""
        for w in self.weeks:
            if w.attacks and w.blocked_pct[variant] >= threshold:
                return w.week
        return None


def _rule_levels(cols: dict[str, npt.NDArray[np.float64]], exclude: frozenset[str]) -> IntArray:
    """Como ``apply_rules`` pero sin algunas reglas: el banco de este experimento no tiene R005."""
    level = np.zeros(len(cols["amount_usd"]), dtype=np.int64)
    for rule in RULES:
        if rule.code in exclude:
            continue
        hit = np.asarray(rule.condition(cols), dtype=bool)
        level = np.where(hit, np.maximum(level, int(rule.action)), level)
    return level


class _World:
    """Los datos de la simulación ya cargados, con helpers para cortar por día."""

    def __init__(self, features_path: Path) -> None:
        data = _load(features_path)
        self.x = _matrix(data)
        self.ts = np.asarray(data["ts_s"], dtype=np.int64)
        self.start = int(self.ts.min()) // DAY * DAY
        self.day = (self.ts - self.start) // DAY
        self.y = np.asarray(data["is_fraud"], dtype=bool)
        self.reported = np.asarray(data["reported_s"], dtype=np.int64)
        self.scenario = np.asarray(data["fraud_scenario"]).astype(str)
        self.amount = np.asarray(data["amount_usd"], dtype=np.float64)
        self.cols = {name: self.x[:, i] for i, name in enumerate(FEATURES)}
        self.rules = _rule_levels(self.cols, frozenset({"R005"}))
        self.r001 = self.cols["small_tx_count_1h"] >= 4
        con = duckdb.connect()
        self.customer = np.asarray(
            con.execute(
                "SELECT customer_id FROM read_parquet(?) ORDER BY ts, tx_id",  # mismo orden
                [features_path.as_posix()],
            ).fetchnumpy()["customer_id"],
            dtype=np.int64,
        )
        con.close()

    def between(self, lo: int, hi: int) -> BoolArray:
        return (self.day >= lo) & (self.day < hi)

    def known_at(self, day: int) -> BoolArray:
        cutoff = self.start + day * DAY
        return self.y & (self.reported >= 0) & (self.reported < cutoff)


@dataclass
class _Model:
    booster: object
    thresholds: Thresholds
    train_rows: BoolArray
    labels: BoolArray

    def scores(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        proba = self.booster.predict_proba(x)  # type: ignore[attr-defined]
        return np.asarray(proba[:, 1], dtype=np.float64)


def _train_at(w: _World, day: int, recipe: Recipe, device: str, warmup: int = 7) -> _Model:
    valid_hi = day - recipe.gap_days
    train_hi = valid_hi - recipe.valid_days
    train_lo = max(warmup, train_hi - recipe.train_days)
    if train_hi - train_lo < 7:
        raise ValueError(f"Muy pocos días para reentrenar el día {day} ({recipe.name}).")
    known = w.known_at(day)
    tr, va = w.between(train_lo, train_hi), w.between(train_hi, valid_hi)
    if recipe.fresh_frauds:
        tr = tr | (w.between(valid_hi, day) & known)
    model = _fit(w.x[tr], known[tr], device)
    s = np.asarray(model.predict_proba(w.x[va])[:, 1], dtype=np.float64)
    th = calibrate(s, w.rules[va], known[va], w.amount[va], w.ts[va], CostModel(), 50)
    return _Model(model, th, tr, known)


def _attack_outcome(w: _World, span: BoolArray, action: IntArray,
                    review_catch_rate: float = 0.85) -> tuple[int, float, float]:  # fmt: skip
    """(tarjetas atacadas, % bloqueadas antes del cobro grande, $ perdido) en ``span``.

    ``action`` cubre todas las filas del mundo (fuera de ``span`` no se usa)."""
    idx = np.flatnonzero(span & (w.scenario == "card_testing"))
    act = action[idx]
    cust, amount = w.customer[idx], w.amount[idx]
    attacked = blocked = 0
    lost = 0.0
    for c in np.unique(cust):
        mine = cust == c  # ya vienen en orden de tiempo
        a, m = act[mine], amount[mine]
        stop = int(np.argmax(a == Action.DECLINE)) if (a == Action.DECLINE).any() else len(a)
        lost += float(m[:stop][a[:stop] == Action.APPROVE].sum())
        lost += (1 - review_catch_rate) * float(m[:stop][a[:stop] == Action.REVIEW].sum())
        cash = m >= 150
        if cash.any():
            attacked += 1
            blocked += int(stop < int(np.argmax(cash)))
    pct = round(100 * blocked / attacked, 1) if attacked else 0.0
    return attacked, pct, round(lost, 2)


def run_drift(work_dir: Path, cfg: SimConfig | None = None, weeks: int = 8, device: str = "auto",
              log: Printer = print) -> DriftReport:  # fmt: skip
    cfg = cfg or SimConfig(days=150, drift_day=80)
    if cfg.drift_day is None:
        raise ValueError("La configuración necesita drift_day.")
    d0 = cfg.drift_day
    weeks = min(weeks, (cfg.days - d0) // 7)
    device = pick_device(device)
    raw, feats = work_dir / "raw", work_dir / "features.parquet"
    log(f"  simulando {cfg.days} días (el ataque cambia el día {d0})…")
    write_parquet(simulate(cfg), raw)
    log("  calculando features…")
    build_features(raw, feats)
    w = _World(feats)
    report = DriftReport(device=device, drift_day=d0)

    log(f"  entrenando el modelo del día {d0} (el que está en producción) en {device}…")
    frozen = _train_at(w, d0, STANDARD, device)

    # Monitoreo sin etiquetas con el modelo en producción. Referencia: 2 semanas que el modelo
    # no vio al entrenar; las 2 semanas siguientes (antes del ataque) miden falsas alarmas.
    ref = (d0 - 28, d0 - 14)
    watch = w.between(ref[0], cfg.days)
    scores = np.zeros(len(w.ts))
    scores[watch] = frozen.scores(w.x[watch])
    report.days = daily_monitor(w.cols, FEATURES, scores, w.day, w.r001, ref, (ref[1], cfg.days))
    for name in MONITORS:
        report.first_alarm_day[name] = first_alarm(report.days, name, since=d0)
        report.false_alarm_days[name] = sum(1 for s in report.days
                                            if s.day < d0 and name in s.alarms)  # fmt: skip

    new_attack = (w.scenario == "card_testing") & (w.day >= d0)
    honest = ~w.y
    for k in range(weeks):
        day = d0 + 7 * k
        log(f"  semana {k + 1}/{weeks}: reentrenando con lo que se sabía el día {day}…")
        models = {FROZEN: frozen}
        for recipe in RECIPES:
            models[recipe.name] = frozen if k == 0 else _train_at(w, day, recipe, device)
        span = w.between(day, day + 7)
        labels, blocked, lost, false_dec = {}, {}, {}, {}
        attacks = 0
        for name, m in models.items():
            labels[name] = int((m.train_rows & m.labels & new_attack).sum())
            action = np.zeros(len(w.ts), dtype=np.int64)
            action[span] = decide(m.scores(w.x[span]), w.rules[span], m.thresholds)
            attacks, blocked[name], lost[name] = _attack_outcome(w, span, action)
            declined_honest = span & honest & (action == Action.DECLINE)
            false_dec[name] = int(len(np.unique(w.customer[declined_honest])))
        report.weeks.append(WeekResult(k + 1, day, attacks, labels, blocked, lost, false_dec))
    return report
