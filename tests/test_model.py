"""Pruebas del modelo: sin fuga de datos, split temporal correcto y métricas coherentes."""

from pathlib import Path

import duckdb
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from halcon.cli import run
from halcon.model.evaluate import alert_budget
from halcon.model.features import FEATURES, LABEL_COLUMNS, build_features
from halcon.model.train import SplitConfig, train
from halcon.simulator import SimConfig, simulate, write_parquet

CFG = SimConfig(n_customers=1_000, n_merchants=300, days=70, seed=3)


@pytest.fixture(scope="module")
def raw(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("raw")
    write_parquet(simulate(CFG), out)
    return out


@pytest.fixture(scope="module")
def features(raw: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("feat") / "features.parquet"
    build_features(raw, out)
    return out


def test_labels_are_not_features() -> None:
    assert not set(FEATURES) & set(LABEL_COLUMNS)
    assert not any("fraud" in name for name in FEATURES)


def test_one_feature_row_per_transaction(raw: Path, features: Path) -> None:
    n_raw = pq.read_metadata(raw / "transactions.parquet").num_rows
    assert pq.read_metadata(features).num_rows == n_raw


def test_features_do_not_look_into_the_future(raw: Path, features: Path, tmp_path: Path) -> None:
    """Si borro todo lo que pasa después del día 30, los features anteriores no cambian."""
    tx = pq.read_table(raw / "transactions.parquet")
    cut = pc.min(tx["ts"]).as_py().timestamp() + 30 * 86_400
    epoch_s = pc.cast(pc.cast(tx["ts"], "int64"), "int64")
    past_only = tx.filter(pc.less(epoch_s, int(cut)))
    trunc = tmp_path / "raw"
    trunc.mkdir()
    pq.write_table(past_only, trunc / "transactions.parquet")
    for name in ("customers", "merchants"):
        pq.write_table(pq.read_table(raw / f"{name}.parquet"), trunc / f"{name}.parquet")
    trunc_feat = tmp_path / "features.parquet"
    build_features(trunc, trunc_feat)

    query = f"SELECT tx_id, {', '.join(FEATURES)} FROM read_parquet(?) ORDER BY tx_id"  # noqa: S608
    con = duckdb.connect()
    full = con.execute(query, [features.as_posix()]).fetchnumpy()
    part = con.execute(query, [trunc_feat.as_posix()]).fetchnumpy()
    ids = part["tx_id"]
    keep = np.isin(full["tx_id"], ids)
    for name in FEATURES:
        a = np.ma.filled(np.ma.asarray(full[name][keep], dtype=float), np.nan)
        b = np.ma.filled(np.ma.asarray(part[name], dtype=float), np.nan)
        np.testing.assert_allclose(a, b, equal_nan=True, err_msg=name)


def test_card_testing_shows_many_small_payments(features: Path) -> None:
    row = (
        duckdb.connect()
        .execute(
            """SELECT avg(small_tx_count_1h) FILTER (WHERE fraud_scenario = 'card_testing'),
                      avg(small_tx_count_1h) FILTER (WHERE NOT is_fraud)
               FROM read_parquet(?)""",
            [features.as_posix()],
        )
        .fetchone()
    )
    assert row is not None
    assert row[0] > 3 > row[1]


def test_alert_budget_counts() -> None:
    day = 86_400
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([True, False, False, True])
    ts = np.array([0, 10, day, day + 10], dtype=np.int64)
    amount = np.array([100.0, 5.0, 7.0, 50.0])
    scenario = np.array(["skimming", "legit", "legit", "card_testing"])
    result = alert_budget(scores, y, ts, amount, scenario, alerts_per_day=1)
    assert result.precision == 0.5  # 1 de 2 alertas era fraude
    assert result.recall == 0.5  # se atrapó 1 de 2 fraudes
    assert result.amount_recall == round(100 / 150, 4)
    assert result.recall_by_scenario == {"card_testing": 0.0, "skimming": 1.0}


def test_train_respects_feedback_delay_and_beats_baseline(features: Path, tmp_path: Path) -> None:
    split = SplitConfig(warmup_days=5, delay_gap_days=10, test_days=20, alerts_per_day=20)
    report = train(features, tmp_path, split, device="cpu")
    assert report.train_frauds_known > 0
    assert report.model.pr_auc > report.baseline.pr_auc
    assert (tmp_path / "xgb.json").exists()
    assert (tmp_path / "metrics.json").exists()


def test_cli_features_and_train(
    raw: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    feat = tmp_path / "f.parquet"
    assert run(["features", "--raw", str(raw), "--out", str(feat)]) == 0
    assert run(["train", "--features", str(feat), "--model-dir", str(tmp_path / "m"),
                "--device", "cpu"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "PR-AUC" in out and "xgboost" in out
