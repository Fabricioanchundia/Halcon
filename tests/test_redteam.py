"""Red team: el atacante que conoce R001 y la regla R005 que lo frena."""

from pathlib import Path

import numpy as np
import pytest

from halcon.decision.rules import Action, apply_rules
from halcon.model.features import FEATURES, build_features
from halcon.model.train import SplitConfig, train
from halcon.security.redteam import run_redteam
from halcon.simulator import SimConfig, simulate, write_parquet

BASE = SimConfig(n_customers=1_500, n_merchants=300, days=70, seed=21, card_testing_rate=0.01)


def _row(**values: float) -> dict[str, np.ndarray]:
    base = {name: 0.0 for name in FEATURES} | {"amount_usd": 20.0, "amount_ratio_30d": 1.0}
    return {k: np.array([v]) for k, v in (base | values).items()}


def test_adapted_probes_evade_r001_but_not_r005() -> None:
    probe = _row(amount_usd=2.5, small_tx_count_1h=0, tx_count_10m=4, is_online=1)
    level, hits = apply_rules(probe)
    assert not hits["R001"][0]  # $2,50 esquiva la regla de "$2 o menos"
    assert hits["R005"][0] and level[0] == Action.DECLINE


def test_r005_ignores_normal_shopping() -> None:
    # Tres compras en línea seguidas, o una compra grande: no es una ráfaga de pruebas.
    for row in (_row(tx_count_10m=3, amount_usd=5, is_online=1),
                _row(tx_count_10m=5, amount_usd=45, is_online=1),
                _row(tx_count_10m=6, amount_usd=5, is_online=0)):  # fmt: skip
        _, hits = apply_rules(row)
        assert not hits["R005"][0]


def test_simulator_probe_amounts_follow_config() -> None:
    res = simulate(SimConfig(n_customers=800, n_merchants=200, days=40, seed=3,
                             card_testing_rate=0.02, probe_min=2.01, probe_max=3.0))  # fmt: skip
    tx = res.transactions
    probes = tx.amount[(tx.scenario == 1) & (tx.amount < 10)]
    assert probes.size and probes.min() >= 2.01 and probes.max() <= 3.0


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("rt")
    write_parquet(simulate(BASE), root)
    build_features(root, root / "features.parquet")
    split = SplitConfig(warmup_days=5, valid_days=10, delay_gap_days=10, test_days=20,
                        alerts_per_day=20)  # fmt: skip
    train(root / "features.parquet", root / "models", split, device="cpu")
    return root / "models"


def test_redteam_runs_and_r005_keeps_blocking(model_dir: Path, tmp_path: Path) -> None:
    normal, adapted = run_redteam(tmp_path, model_dir, days=20, base=BASE, log=lambda _: None)
    assert normal.probes > 0 and adapted.probes > 0
    assert normal.probes_declined_pct > 50
    assert adapted.probes_declined_pct > 30  # gracias a R005, aunque R001 ya no sirva
