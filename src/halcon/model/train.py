"""Entrenamiento del modelo XGBoost con validación temporal realista.

Línea de tiempo (con 120 días simulados)::

    día 0 ─ 7 ────────── train ────────── 62 ─ validación ─ 76 ─ hueco ─ 90 ──── test ──── 120
            └ calentamiento: los clientes aún no tienen historial

* **Corte** (día 90): el día en que "entrenamos" el modelo. Solo conocemos las etiquetas de
  fraude reportadas ANTES de ese día. Un fraude ocurrido el día 75 y reportado el día 95 se ve
  como legítimo en el entrenamiento, igual que en un banco real.
* **Hueco** (14 días antes del corte): sus transacciones aún tienen muchas etiquetas
  pendientes, así que no se usan para entrenar.
* **Validación** (14 días antes del hueco): no se entrena con ella; sirve para elegir los
  umbrales de la política de decisión (Fase 3) sin mirar el test.
* **Test** (después del corte): se evalúa con la verdad completa, como si esperáramos a que
  lleguen todos los reclamos.

Para mostrar cuánto cuesta esperar las etiquetas, se entrena también un modelo "oráculo" que
hace trampa y conoce todas las etiquetas del periodo de entrenamiento.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np
import numpy.typing as npt
import xgboost as xgb

from halcon.decision.policy import (
    CostModel,
    PolicyResult,
    Thresholds,
    calibrate,
    decide,
    evaluate_policy,
)
from halcon.decision.rules import RULES, apply_rules
from halcon.model.evaluate import Evaluation, evaluate
from halcon.model.features import FEATURES

DAY = 86_400


@dataclass(frozen=True)
class SplitConfig:
    warmup_days: int = 7
    valid_days: int = 14
    delay_gap_days: int = 14
    test_days: int = 30
    alerts_per_day: int = 50


@dataclass(frozen=True)
class TrainReport:
    device: str
    train_rows: int
    train_frauds_known: int
    train_frauds_hidden: int
    test_rows: int
    test_frauds: int
    train_seconds: float
    model: Evaluation
    oracle: Evaluation
    baseline: Evaluation
    feature_importance: dict[str, float]
    thresholds: Thresholds
    policies: list[PolicyResult]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def pick_device(requested: str) -> str:
    """'auto' usa la GPU si hay una NVIDIA visible (nvidia-smi), si no la CPU."""
    if requested != "auto":
        return requested
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def _load(path: Path) -> dict[str, npt.NDArray[np.generic]]:
    cols = ", ".join(FEATURES)
    sql = f"""
        SELECT {cols},
               epoch(ts)::BIGINT AS ts_s,
               is_fraud,
               coalesce(fraud_scenario, 'legit') AS fraud_scenario,
               coalesce(epoch(fraud_reported_at)::BIGINT, -1) AS reported_s
        FROM read_parquet(?)
        ORDER BY ts, tx_id
    """  # noqa: S608 — las columnas vienen de la constante FEATURES, la ruta va como parámetro
    con = duckdb.connect()
    data: dict[str, npt.NDArray[np.generic]] = con.execute(sql, [path.as_posix()]).fetchnumpy()
    con.close()
    return data


def _matrix(data: dict[str, npt.NDArray[np.generic]]) -> npt.NDArray[np.float64]:
    cols = []
    for name in FEATURES:
        col = data[name]
        if np.ma.isMaskedArray(col):
            col = np.ma.filled(col.astype(np.float64), np.nan)
        cols.append(np.asarray(col, dtype=np.float64))
    return np.column_stack(cols)


def _fit(x: npt.NDArray[np.float64], y: npt.NDArray[np.bool_], device: str) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(x, y)
    model.set_params(device="cpu")  # predecir en CPU evita copiar datos a la GPU
    return model


def train(
    features_path: Path,
    model_dir: Path,
    split: SplitConfig | None = None,
    device: str = "auto",
) -> TrainReport:
    split = split or SplitConfig()
    device = pick_device(device)
    data = _load(features_path)
    x = _matrix(data)
    ts_s = np.asarray(data["ts_s"], dtype=np.int64)
    is_fraud = np.asarray(data["is_fraud"], dtype=bool)
    reported = np.asarray(data["reported_s"], dtype=np.int64)
    scenario = np.asarray(data["fraud_scenario"]).astype(str)
    amount = np.asarray(data["amount_usd"], dtype=np.float64)

    start = int(ts_s.min()) // DAY * DAY
    total_days = int((ts_s.max() - start) // DAY) + 1
    cutoff = start + (total_days - split.test_days) * DAY
    train_lo = start + split.warmup_days * DAY
    valid_hi = cutoff - split.delay_gap_days * DAY
    train_hi = valid_hi - split.valid_days * DAY
    if train_hi <= train_lo:
        raise ValueError(f"Muy pocos días ({total_days}) para el split temporal pedido.")

    in_train = (ts_s >= train_lo) & (ts_s < train_hi)
    in_valid = (ts_s >= train_hi) & (ts_s < valid_hi)
    in_test = ts_s >= cutoff
    known = is_fraud & (reported >= 0) & (reported < cutoff)  # lo que el banco sabía al corte

    y_train = known[in_train]
    t0 = time.perf_counter()
    model = _fit(x[in_train], y_train, device)
    train_seconds = round(time.perf_counter() - t0, 2)
    oracle = _fit(x[in_train], is_fraud[in_train], device)

    args = (is_fraud[in_test], ts_s[in_test], amount[in_test], scenario[in_test])
    k = split.alerts_per_day
    model_eval = evaluate("xgboost", model.predict_proba(x[in_test])[:, 1], *args, k)
    oracle_eval = evaluate("xgboost_oracle", oracle.predict_proba(x[in_test])[:, 1], *args, k)

    # Baseline ingenuo: "cuanto más se aleja del gasto normal, más sospechoso".
    ratio = x[in_test, FEATURES.index("amount_ratio_30d")]
    fallback = x[in_test, FEATURES.index("amount_usd")] / 25.0
    baseline_scores = np.where(np.isnan(ratio), fallback, ratio)
    baseline_eval = evaluate("baseline_amount_ratio", baseline_scores, *args, k)

    # --- Fase 3: política de decisión calibrada en validación, evaluada en test ---
    costs = CostModel()
    cols = {name: x[:, i] for i, name in enumerate(FEATURES)}
    rule_level, _ = apply_rules(cols)
    scores_all = np.zeros(len(ts_s))
    scores_all[in_valid] = model.predict_proba(x[in_valid])[:, 1]
    scores_all[in_test] = model.predict_proba(x[in_test])[:, 1]
    no_rules = np.zeros_like(rule_level)

    def calib(levels: npt.NDArray[np.int64]) -> Thresholds:
        v = in_valid
        return calibrate(
            scores_all[v], levels[v], known[v], amount[v], ts_s[v], costs, k
        )  # etiquetas conocidas al corte, como en la vida real

    thresholds = calib(rule_level)
    model_only = calib(no_rules)
    t = in_test
    y_t, a_t, ts_t = is_fraud[t], amount[t], ts_s[t]
    policies = [
        evaluate_policy(
            "aprobar_todo", np.zeros(int(t.sum()), dtype=np.int64), y_t, a_t, ts_t, costs
        ),
        evaluate_policy("solo_reglas", rule_level[t], y_t, a_t, ts_t, costs),
        evaluate_policy(
            "solo_modelo", decide(scores_all[t], no_rules[t], model_only), y_t, a_t, ts_t, costs
        ),
        evaluate_policy(
            "halcon", decide(scores_all[t], rule_level[t], thresholds), y_t, a_t, ts_t, costs
        ),
    ]

    raw_gain = model.get_booster().get_score(importance_type="gain")
    gain = {key: float(np.sum(value)) for key, value in raw_gain.items()}
    total_gain = sum(gain.values()) or 1.0
    importance = {
        FEATURES[int(key[1:])]: round(value / total_gain, 4)
        for key, value in sorted(gain.items(), key=lambda kv: -kv[1])
    }

    report = TrainReport(
        device=device,
        train_rows=int(in_train.sum()),
        train_frauds_known=int(y_train.sum()),
        train_frauds_hidden=int((is_fraud[in_train] & ~known[in_train]).sum()),
        test_rows=int(in_test.sum()),
        test_frauds=int(is_fraud[in_test].sum()),
        train_seconds=train_seconds,
        model=model_eval,
        oracle=oracle_eval,
        baseline=baseline_eval,
        feature_importance=importance,
        thresholds=thresholds,
        policies=policies,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().feature_names = list(FEATURES)
    model.save_model(model_dir / "xgb.json")
    policy_doc = {
        "thresholds": thresholds.to_dict(),
        "costs": asdict(costs),
        "reviews_per_day_capacity": k,
        "rules": [
            {"code": r.code, "description": r.description, "action": r.action.name} for r in RULES
        ],
    }
    (model_dir / "policy.json").write_text(
        json.dumps(policy_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report
