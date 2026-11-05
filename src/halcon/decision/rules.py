"""Reglas duras: rápidas, explicables y con código de razón.

Las reglas no compiten con el modelo, lo complementan:

* Cubren patrones obvios con certeza casi total (card testing) y permiten **rechazar** sin
  depender de un puntaje.
* Se pueden cambiar en minutos cuando aparece un ataque nuevo, mientras el modelo se reentrena.
* Su código de razón (``R001``…) llega al analista y al cliente. Es lo que exige un regulador.

Cada regla se probó contra los datos simulados antes de entrar. Por ejemplo, "compra en línea
8 veces el promedio" se descartó: atrapaba 84 fraudes molestando a más de 4 000 clientes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt

BoolArray = npt.NDArray[np.bool_]
Columns = Mapping[str, npt.NDArray[np.float64]]


class Action(IntEnum):
    """Orden de severidad: si dos fuentes opinan distinto, gana la más severa."""

    APPROVE = 0
    REVIEW = 1
    DECLINE = 2


@dataclass(frozen=True)
class Rule:
    code: str
    description: str
    action: Action
    condition: Callable[[Columns], BoolArray]


def _nan_safe(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.nan_to_num(values, nan=0.0)


RULES: tuple[Rule, ...] = (
    Rule(
        "R001",
        "Card testing: 4 o más compras de $2 o menos en la última hora",
        Action.DECLINE,
        lambda f: f["small_tx_count_1h"] >= 4,
    ),
    Rule(
        "R002",
        "Velocidad: 5 o más compras en 10 minutos",
        Action.REVIEW,
        lambda f: f["tx_count_10m"] >= 5,
    ),
    Rule(
        "R003",
        "Compra presencial en otra ciudad, 6 veces su promedio y de $200 o más",
        Action.REVIEW,
        lambda f: (
            (f["is_other_city"] == 1)
            & (_nan_safe(f["amount_ratio_30d"]) >= 6)
            & (f["amount_usd"] >= 200)
        ),
    ),
    Rule(
        "R004",
        "Monto muy alto: $2 000 o más",
        Action.REVIEW,
        lambda f: f["amount_usd"] >= 2000,
    ),
)

RULE_FEATURES: frozenset[str] = frozenset(
    {"small_tx_count_1h", "tx_count_10m", "is_other_city", "amount_ratio_30d", "amount_usd"}
)


def apply_rules(columns: Columns) -> tuple[npt.NDArray[np.int64], dict[str, BoolArray]]:
    """Evalúa todas las reglas a la vez. Devuelve la acción más severa por fila y qué disparó."""
    n = len(next(iter(columns.values())))
    level = np.zeros(n, dtype=np.int64)
    hits: dict[str, BoolArray] = {}
    for rule in RULES:
        hit = np.asarray(rule.condition(columns), dtype=bool)
        hits[rule.code] = hit
        level = np.where(hit, np.maximum(level, int(rule.action)), level)
    return level, hits
