"""Pruebas del simulador: reproducible, coherente y con fraude realista."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from halcon.cli import run
from halcon.simulator import SimConfig, SimResult, simulate, summary
from halcon.simulator.engine import ACCOUNT_TAKEOVER, CARD_TESTING, LEGIT, SKIMMING

SMALL = SimConfig(n_customers=500, n_merchants=300, days=60, seed=7)


@pytest.fixture(scope="module")
def result() -> SimResult:
    return simulate(SMALL)


def test_same_seed_same_data(result: SimResult) -> None:
    again = simulate(SMALL)
    np.testing.assert_array_equal(result.transactions.ts, again.transactions.ts)
    np.testing.assert_array_equal(result.transactions.amount, again.transactions.amount)


def test_different_seed_different_data(result: SimResult) -> None:
    other = simulate(SimConfig(n_customers=500, n_merchants=300, days=60, seed=8))
    assert len(other.transactions) != len(result.transactions)


def test_sorted_and_inside_period(result: SimResult) -> None:
    ts = result.transactions.ts
    assert np.all(np.diff(ts) >= 0)
    assert ts.min() >= SMALL.start_ts
    assert ts.max() < SMALL.end_ts


def test_amounts_positive_with_cents(result: SimResult) -> None:
    amount = result.transactions.amount
    assert amount.min() >= 0.5
    np.testing.assert_allclose(amount, np.round(amount, 2))


def test_all_scenarios_present_and_fraud_is_rare(result: SimResult) -> None:
    scenario = result.transactions.scenario
    for s in (LEGIT, CARD_TESTING, ACCOUNT_TAKEOVER, SKIMMING):
        assert (scenario == s).any()
    assert 0.0005 < (scenario > 0).mean() < 0.02


def test_labels_arrive_after_the_fraud(result: SimResult) -> None:
    tx = result.transactions
    fraud = tx.scenario > 0
    assert np.all(tx.reported_at[fraud] > tx.ts[fraud])
    assert np.all(tx.reported_at[~fraud] == -1)


def test_card_testing_probes_are_tiny(result: SimResult) -> None:
    tx = result.transactions
    amounts = tx.amount[tx.scenario == CARD_TESTING]
    assert (amounts <= 2.0).mean() > 0.6


def test_fraud_merchants_match_channel(result: SimResult) -> None:
    tx, merch = result.transactions, result.merchants
    city = merch.city[tx.merchant]
    assert np.all(city[tx.scenario == CARD_TESTING] == -1)  # en línea
    assert np.all(city[tx.scenario == ACCOUNT_TAKEOVER] == -1)  # en línea
    assert np.all(city[tx.scenario == SKIMMING] >= 0)  # presencial


def test_summary_counts_add_up(result: SimResult) -> None:
    stats = summary(result)
    parts = stats["legit"] + stats["card_testing"] + stats["account_takeover"] + stats["skimming"]
    assert parts == stats["transactions"]


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="n_customers"):
        SimConfig(n_customers=1)


def test_cli_writes_parquet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(
        ["simulate", "--customers", "200", "--merchants", "150", "--days", "30",
         "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    table = pq.read_table(tmp_path / "transactions.parquet")
    assert {"ts", "amount_usd", "mcc", "channel", "is_fraud", "fraud_reported_at"} <= set(
        table.column_names
    )
    assert (tmp_path / "customers.parquet").exists()
    # Las propiedades ocultas del cliente nunca se exportan.
    cust_cols = pq.read_table(tmp_path / "customers.parquet").column_names
    assert "mean_amount" not in cust_cols
    assert "Simulación lista" in capsys.readouterr().out
