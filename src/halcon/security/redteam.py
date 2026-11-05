"""Red team: ¿qué pasa cuando el atacante conoce nuestras reglas?

La regla R001 rechaza el *card testing* cuando hay 4 o más compras de $2 o menos en una hora.
Un atacante que la descubre (por prueba y error, o por una filtración) sube sus pruebas a
$2,01–$3. Este módulo simula exactamente eso:

* Misma semilla y mismos clientes, comercios y pagos legítimos. Lo ÚNICO que cambia es el monto
  de las compras de prueba. Así la diferencia se le atribuye al ataque y a nada más.
* Se decide con el modelo y la política YA entrenados (sin reentrenar), como pasaría en
  producción el día del ataque.

Métrica clave: **tarjetas bloqueadas antes del cobro grande**. Si alguna compra de prueba se
rechaza, el banco bloquea la tarjeta y el ladrón nunca llega a la compra de $150–$900.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np

from halcon.decision.policy import decide
from halcon.decision.rules import Action, apply_rules
from halcon.decision.scorer import Scorer
from halcon.model.features import FEATURES, build_features
from halcon.simulator import SimConfig, simulate, write_parquet

Printer = Callable[[str], None]


@dataclass(frozen=True)
class AttackResult:
    name: str
    probes: int
    probes_declined_pct: float
    probes_reviewed_pct: float
    probes_approved_pct: float
    cards_blocked_before_cashout_pct: float
    cashouts_stopped_pct: float
    card_testing_usd_lost: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _score_period(raw: Path, features: Path, model_dir: Path, days: int) -> dict[str, np.ndarray]:
    cols = ", ".join(f"f.{c}" for c in FEATURES)
    sql = f"""
        WITH lim AS (SELECT max(ts) - to_days(CAST(? AS INTEGER)) AS since FROM read_parquet(?))
        SELECT f.tx_id, f.customer_id, epoch(f.ts)::BIGINT AS ts, f.amount_usd,
               coalesce(t.fraud_scenario, 'legit') AS scenario, {cols}
        FROM read_parquet(?) f JOIN read_parquet(?) t USING (tx_id)
        WHERE f.ts >= (SELECT since FROM lim)
        ORDER BY f.ts, f.tx_id
    """  # noqa: S608 — solo nombres de columnas constantes
    fpath = features.as_posix()
    data: dict[str, np.ndarray] = (
        duckdb.connect()
        .execute(sql, [days, fpath, fpath, (raw / "transactions.parquet").as_posix()])
        .fetchnumpy()
    )
    x = np.column_stack(
        [np.ma.filled(np.ma.asarray(data[c], dtype=float), np.nan) for c in FEATURES]
    )
    scorer = Scorer.load(model_dir)
    scores = np.asarray(scorer.booster.inplace_predict(x), dtype=float)
    levels, _ = apply_rules({c: x[:, i] for i, c in enumerate(FEATURES)})
    data["action"] = decide(scores, levels, scorer.thresholds)
    return data


def _measure(name: str, d: dict[str, np.ndarray], review_catch_rate: float = 0.85) -> AttackResult:
    ct = np.asarray(d["scenario"]).astype(str) == "card_testing"
    amount = np.asarray(d["amount_usd"], dtype=float)
    action = np.asarray(d["action"])
    probe = ct & (amount < 10)
    cash = ct & (amount >= 150)

    def pct(mask: np.ndarray, act: int) -> float:
        return round(100 * float((action[mask] == act).mean()), 1) if mask.any() else 0.0

    # ¿Se bloqueó la tarjeta (alguna prueba RECHAZADA) antes de su primer cobro grande?
    cust, ts = np.asarray(d["customer_id"]), np.asarray(d["ts"])
    blocked = stopped = attacks = 0
    for c in np.unique(cust[cash]):
        first_cash = ts[cash & (cust == c)].min()
        mine = probe & (cust == c) & (ts < first_cash)
        attacks += 1
        if (action[mine] == Action.DECLINE).any():
            blocked += 1
            stopped += 1
        elif (action[cash & (cust == c)] > Action.APPROVE).any():
            stopped += 1  # el cobro mismo quedó en revisión o rechazado

    # Dinero perdido en card testing: lo aprobado, más lo revisado que se escapa.
    approved = ct & (action == Action.APPROVE)
    reviewed = ct & (action == Action.REVIEW)
    lost = amount[approved].sum() + (1 - review_catch_rate) * amount[reviewed].sum()
    return AttackResult(
        name=name,
        probes=int(probe.sum()),
        probes_declined_pct=pct(probe, Action.DECLINE),
        probes_reviewed_pct=pct(probe, Action.REVIEW),
        probes_approved_pct=pct(probe, Action.APPROVE),
        cards_blocked_before_cashout_pct=round(100 * blocked / max(1, attacks), 1),
        cashouts_stopped_pct=round(100 * stopped / max(1, attacks), 1),
        card_testing_usd_lost=round(float(lost), 2),
    )


def run_redteam(work_dir: Path, model_dir: Path, probe_min: float = 2.01, probe_max: float = 3.0,
                days: int = 30, base: SimConfig | None = None,
                log: Printer = print) -> tuple[AttackResult, AttackResult]:  # fmt: skip
    base = base or SimConfig()
    results = []
    variants = (
        ("ataque de siempre ($0,50–$2)", base),
        (f"atacante adaptado (${probe_min:.2f}–${probe_max:.2f})".replace(".", ","),
         SimConfig(**{**asdict(base), "probe_min": probe_min, "probe_max": probe_max})),
    )  # fmt: skip
    for i, (name, cfg) in enumerate(variants):
        raw = work_dir / f"v{i}" / "raw"
        feats = work_dir / f"v{i}" / "features.parquet"
        log(f"  simulando y calculando features: {name}…")
        write_parquet(simulate(cfg), raw)
        build_features(raw, feats)
        results.append(_measure(name, _score_period(raw, feats, model_dir, days)))
    return results[0], results[1]
