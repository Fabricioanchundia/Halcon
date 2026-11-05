"""Pruebas del motor de decisión: reglas, política por costos y scorer explicable."""

from pathlib import Path

import numpy as np
import pytest

from halcon.cli import run
from halcon.decision.policy import CostModel, Thresholds, calibrate, decide, evaluate_policy
from halcon.decision.rules import RULES, Action, apply_rules
from halcon.decision.scorer import Scorer
from halcon.model.features import FEATURES, build_features
from halcon.model.train import SplitConfig, train
from halcon.simulator import SimConfig, simulate, write_parquet

DAY = 86_400


def _cols(**overrides: float) -> dict[str, np.ndarray]:
    base = {name: 0.0 for name in FEATURES} | {"amount_usd": 20.0, "amount_ratio_30d": 1.0}
    return {k: np.array([v]) for k, v in (base | overrides).items()}


def test_rule_codes_are_unique() -> None:
    codes = [r.code for r in RULES]
    assert len(codes) == len(set(codes))


def test_card_testing_rule_declines() -> None:
    level, hits = apply_rules(_cols(small_tx_count_1h=4, amount_usd=1.0))
    assert level[0] == Action.DECLINE
    assert hits["R001"][0]


def test_normal_purchase_triggers_nothing() -> None:
    level, hits = apply_rules(_cols())
    assert level[0] == Action.APPROVE
    assert not any(h[0] for h in hits.values())


def test_missing_ratio_does_not_trigger_rule() -> None:
    level, _ = apply_rules(_cols(is_other_city=1, amount_ratio_30d=float("nan"), amount_usd=500))
    assert level[0] == Action.APPROVE


def test_most_severe_action_wins() -> None:
    scores = np.array([0.1, 0.1, 0.9])
    rules = np.array([0, 2, 0], dtype=np.int64)
    action = decide(scores, rules, Thresholds(review=0.05, decline=0.8))
    assert action.tolist() == [Action.REVIEW, Action.DECLINE, Action.DECLINE]


def test_policy_costs_arithmetic() -> None:
    costs = CostModel(chargeback_fee=10, review_cost=2, review_catch_rate=1.0,
                      false_decline_fixed=5, false_decline_pct=0.0)  # fmt: skip
    action = np.array([0, 1, 2, 2], dtype=np.int64)
    y = np.array([True, True, True, False])
    amount = np.array([100.0, 50.0, 70.0, 30.0])
    ts = np.array([0, 10, 20, 30], dtype=np.int64)
    r = evaluate_policy("t", action, y, amount, ts, costs)
    assert r.fraud_loss == 110.0  # aprobado: 100 + 10 de contracargo
    assert r.review_cost == 2.0
    assert r.false_decline_cost == 5.0
    assert r.total_cost == 117.0
    assert r.frauds_stopped_pct == pytest.approx(66.7)


def test_calibration_respects_review_capacity() -> None:
    rng = np.random.default_rng(0)
    n = 20 * 1000
    ts = np.repeat(np.arange(20) * DAY, 1000).astype(np.int64)
    y = rng.random(n) < 0.01
    scores = np.clip(rng.normal(0.1, 0.05, n) + 0.6 * y, 0, 1)
    amount = rng.uniform(10, 500, n)
    rules = np.zeros(n, dtype=np.int64)
    th = calibrate(scores, rules, y, amount, ts, CostModel(), reviews_per_day=5)
    reviews = (decide(scores, rules, th) == Action.REVIEW).sum() / 20
    assert reviews <= 5
    assert th.decline >= th.review


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("pipeline")
    write_parquet(simulate(SimConfig(n_customers=1_000, n_merchants=300, days=70, seed=5)), root)
    feats = root / "features.parquet"
    build_features(root, feats)
    split = SplitConfig(warmup_days=5, valid_days=10, delay_gap_days=10, test_days=20,
                        alerts_per_day=20)  # fmt: skip
    report = train(feats, root / "models", split, device="cpu")
    assert [p.name for p in report.policies] == [
        "aprobar_todo", "solo_reglas", "solo_modelo", "halcon"
    ]  # fmt: skip
    halcon = report.policies[-1]
    assert halcon.total_cost < report.policies[0].total_cost  # mejor que no hacer nada
    return feats, root / "models"


def test_scorer_explains_card_testing(trained: tuple[Path, Path]) -> None:
    scorer = Scorer.load(trained[1])
    features = {name: 0.0 for name in FEATURES} | {
        "amount_usd": 1.1, "small_tx_count_1h": 6, "tx_count_10m": 6, "is_online": 1,
        "secs_since_prev_tx": 20, "avg_amount_30d": 25.0, "amount_ratio_30d": 0.04,
    }  # fmt: skip
    decision = scorer.score(features)
    assert decision.action == Action.DECLINE
    assert "R001" in decision.rules
    assert decision.reasons  # siempre explica por qué
    assert decision.latency_ms < 50  # meta de latencia por transacción


def test_scorer_handles_missing_history(trained: tuple[Path, Path]) -> None:
    scorer = Scorer.load(trained[1])
    decision = scorer.score({"amount_usd": 12.0, "avg_amount_30d": None})
    assert 0.0 <= decision.score <= 1.0


def test_cli_score(trained: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
    feats, models = trained
    code = run(["score", "--sample", "card_testing", "--features", str(feats),
                "--model-dir", str(models)])  # fmt: skip
    assert code == 0
    assert "Decisión" in capsys.readouterr().out
