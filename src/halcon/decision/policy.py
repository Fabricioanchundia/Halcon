"""Política de decisión: puntaje del modelo + reglas → APROBAR, REVISAR o RECHAZAR.

Los umbrales no se eligen a ojo: se buscan los que **minimizan el costo total en dólares**,
sin pasarse de la capacidad diaria del equipo de analistas.

Costos por resultado (supuestos documentados, ajustables en ``CostModel``):

==================  ===========================================  =================================
                    Fraude                                       Legítima
==================  ===========================================  =================================
APROBAR             monto + cargo por contracargo                0
REVISAR             revisión + 15 % del monto + contracargo      costo de revisión
                    (el analista no siempre llega a tiempo)
RECHAZAR            0                                            costo fijo + % del monto (cliente
                                                                 molesto, venta perdida)
==================  ===========================================  =================================

Los umbrales se calibran en un periodo de **validación**, nunca en el de test. Si se eligieran
mirando el test, el resultado reportado sería optimista y no se repetiría en producción.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from halcon.decision.rules import Action

DAY = 86_400
NEVER = 1.01  # los puntajes van de 0 a 1: un umbral de 1.01 nunca se alcanza
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class CostModel:
    chargeback_fee: float = 15.0
    review_cost: float = 3.0
    review_catch_rate: float = 0.85  # el analista detiene el 85 % de los fraudes que revisa
    false_decline_fixed: float = 10.0
    false_decline_pct: float = 0.10


@dataclass(frozen=True)
class Thresholds:
    review: float
    decline: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyResult:
    name: str
    total_cost: float
    fraud_loss: float
    review_cost: float
    false_decline_cost: float
    reviews_per_day: float
    false_declines: int
    frauds_stopped_pct: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(scores: FloatArray, rule_level: IntArray, thresholds: Thresholds) -> IntArray:
    """Gana la acción más severa entre la del puntaje y la de las reglas."""
    by_score = np.where(
        scores >= thresholds.decline,
        int(Action.DECLINE),
        np.where(scores >= thresholds.review, int(Action.REVIEW), int(Action.APPROVE)),
    )
    return np.maximum(by_score, rule_level).astype(np.int64)


def evaluate_policy(
    name: str,
    action: IntArray,
    y: BoolArray,
    amount: FloatArray,
    ts_s: IntArray,
    costs: CostModel,
) -> PolicyResult:
    approve, review, decline = (action == a for a in Action)
    missed = y & approve
    reviewed_fraud = y & review
    escape = 1.0 - costs.review_catch_rate
    fraud_loss = float((amount[missed] + costs.chargeback_fee).sum()) + escape * float(
        (amount[reviewed_fraud] + costs.chargeback_fee).sum()
    )
    review_total = float(review.sum() * costs.review_cost)
    fd = ~y & decline
    fd_cost = float((costs.false_decline_fixed + costs.false_decline_pct * amount[fd]).sum())
    days = max(1, int((ts_s.max() - ts_s.min()) // DAY) + 1)
    return PolicyResult(
        name=name,
        total_cost=round(fraud_loss + review_total + fd_cost, 2),
        fraud_loss=round(fraud_loss, 2),
        review_cost=round(review_total, 2),
        false_decline_cost=round(fd_cost, 2),
        reviews_per_day=round(int(review.sum()) / days, 1),
        false_declines=int(fd.sum()),
        frauds_stopped_pct=round(
            100
            * (int(decline[y].sum()) + costs.review_catch_rate * int(reviewed_fraud.sum()))
            / max(1, int(y.sum())),
            1,
        ),
    )


def calibrate(
    scores: FloatArray,
    rule_level: IntArray,
    y: BoolArray,
    amount: FloatArray,
    ts_s: IntArray,
    costs: CostModel,
    reviews_per_day: int,
    grid_size: int = 60,
) -> Thresholds:
    """Busca (umbral de revisión, umbral de rechazo) con el menor costo dentro de la capacidad."""
    quantiles = np.concatenate([np.linspace(0.95, 0.999, grid_size), [0.9995, 0.9999]])
    candidates = np.unique(np.quantile(scores, quantiles))
    lowest = float(candidates.min())

    # Las filas por debajo del umbral más bajo y sin reglas siempre se aprueban: su costo es fijo.
    active = (scores >= lowest) | (rule_level > 0)
    passive = ~active
    fixed_cost = float((amount[passive & y] + costs.chargeback_fee).sum())
    s, r, yy, a, t = scores[active], rule_level[active], y[active], amount[active], ts_s[active]
    days = max(1, int((ts_s.max() - ts_s.min()) // DAY) + 1)

    best: tuple[float, Thresholds] | None = None
    decline_options = [*candidates.tolist(), NEVER]
    for t_review in candidates:
        for t_decline in decline_options:
            if t_decline < t_review:
                continue
            th = Thresholds(review=float(t_review), decline=float(t_decline))
            act = decide(s, r, th)
            if int((act == Action.REVIEW).sum()) / days > reviews_per_day:
                continue
            result = evaluate_policy("calib", act, yy, a, t, costs)
            total = result.total_cost + fixed_cost
            if best is None or total < best[0]:
                best = (total, th)
    if best is None:
        # Ni el umbral más alto cabe en la capacidad: todo lo dudoso se rechaza.
        return Thresholds(review=NEVER, decline=float(candidates.max()))
    return best[1]
