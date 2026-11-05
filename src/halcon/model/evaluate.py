"""Métricas de fraude que le importan a un banco, no solo a Kaggle.

* ``pr_auc``: área bajo la curva precisión-recall. Con 0,2 % de fraude, el ROC-AUC engaña
  (un modelo mediocre puede sacar 0,95); el PR-AUC no.
* **Presupuesto de alertas**: el equipo de analistas solo puede revisar ``k`` casos por día.
  Cada día se toman las ``k`` transacciones con mayor riesgo y se mide cuánto fraude cae ahí:
  precisión (qué fracción de alertas era fraude), recall (qué fracción del fraude se atrapó) y
  dólares atrapados, que es lo que mira el negocio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import average_precision_score, roc_auc_score

DAY = 86_400


@dataclass(frozen=True)
class BudgetResult:
    alerts_per_day: int
    precision: float
    recall: float
    amount_recall: float
    recall_by_scenario: dict[str, float]


@dataclass(frozen=True)
class Evaluation:
    name: str
    pr_auc: float
    roc_auc: float
    budget: BudgetResult

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def alert_budget(
    scores: npt.NDArray[np.float64],
    y: npt.NDArray[np.bool_],
    ts_s: npt.NDArray[np.int64],
    amount: npt.NDArray[np.float64],
    scenario: npt.NDArray[np.str_],
    alerts_per_day: int,
) -> BudgetResult:
    day = (ts_s - ts_s.min()) // DAY
    flagged = np.zeros(len(scores), dtype=bool)
    for d in np.unique(day):
        idx = np.flatnonzero(day == d)
        top = idx[np.argsort(-scores[idx], kind="stable")[:alerts_per_day]]
        flagged[top] = True

    caught = flagged & y
    by_scenario: dict[str, float] = {}
    for s in sorted(set(scenario[y].tolist())):
        in_s = scenario == s
        by_scenario[str(s)] = round(int(caught[in_s].sum()) / max(1, int(in_s.sum())), 4)
    return BudgetResult(
        alerts_per_day=alerts_per_day,
        precision=round(int(caught.sum()) / max(1, int(flagged.sum())), 4),
        recall=round(int(caught.sum()) / max(1, int(y.sum())), 4),
        amount_recall=round(float(amount[caught].sum()) / max(1e-9, float(amount[y].sum())), 4),
        recall_by_scenario=by_scenario,
    )


def evaluate(
    name: str,
    scores: npt.NDArray[np.float64],
    y: npt.NDArray[np.bool_],
    ts_s: npt.NDArray[np.int64],
    amount: npt.NDArray[np.float64],
    scenario: npt.NDArray[np.str_],
    alerts_per_day: int,
) -> Evaluation:
    return Evaluation(
        name=name,
        pr_auc=round(float(average_precision_score(y, scores)), 4),
        roc_auc=round(float(roc_auc_score(y, scores)), 4),
        budget=alert_budget(scores, y, ts_s, amount, scenario, alerts_per_day),
    )
