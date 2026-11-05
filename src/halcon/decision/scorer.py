"""Scorer: decide UNA transacción y explica por qué.

Es la pieza que en la Fase 4 se conectará al flujo en tiempo real y en la Fase 5 alimentará
al agente investigador. Devuelve:

* la acción (APROBAR, REVISAR o RECHAZAR) y el puntaje del modelo;
* las reglas que se dispararon, con su código;
* las pistas que más empujaron el puntaje, calculadas con valores SHAP (``pred_contribs`` de
  XGBoost). Así el analista no recibe solo un número, sino un "por qué".
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from halcon.decision.policy import Thresholds, decide
from halcon.decision.rules import RULES, Action, apply_rules
from halcon.model.features import FEATURES


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float
    contribution: float  # en log-odds: positivo empuja hacia fraude


@dataclass(frozen=True)
class Decision:
    action: Action
    score: float
    rules: list[str]
    reasons: list[Reason]
    latency_ms: float


class Scorer:
    def __init__(self, booster: xgb.Booster, thresholds: Thresholds) -> None:
        self.booster = booster
        self.thresholds = thresholds
        self._rule_text = {r.code: r.description for r in RULES}

    @classmethod
    def load(cls, model_dir: Path) -> Scorer:
        booster = xgb.Booster()
        booster.load_model(model_dir / "xgb.json")
        booster.set_param({"device": "cpu", "nthread": 1})  # una transacción: 1 hilo es más rápido
        policy = json.loads((model_dir / "policy.json").read_text(encoding="utf-8"))
        return cls(booster, Thresholds(**policy["thresholds"]))

    def rule_description(self, code: str) -> str:
        return self._rule_text[code]

    def score(self, features: Mapping[str, float | None], top_k: int = 3) -> Decision:
        t0 = time.perf_counter()
        values = [features.get(f) for f in FEATURES]  # una feature ausente se trata como NaN
        row = np.array([[math.nan if v is None else float(v) for v in values]], dtype=np.float64)

        dm = xgb.DMatrix(row, feature_names=list(FEATURES))
        contribs = self.booster.predict(dm, pred_contribs=True)[0]
        margin = float(contribs.sum())
        score = 1.0 / (1.0 + math.exp(-margin))

        cols = {name: row[:, i] for i, name in enumerate(FEATURES)}
        rule_level, hits = apply_rules(cols)
        action = decide(np.array([score]), rule_level, self.thresholds)[0]

        order = np.argsort(-contribs[:-1])  # la última columna es el sesgo (bias)
        reasons = [
            Reason(FEATURES[i], float(row[0, i]), round(float(contribs[i]), 3))
            for i in order[:top_k]
            if contribs[i] > 0
        ]
        return Decision(
            action=Action(int(action)),
            score=round(score, 4),
            rules=[code for code, hit in hits.items() if hit[0]],
            reasons=reasons,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
