"""Fase 8: drift del fraude, monitoreo sin etiquetas y reentrenamiento."""

from pathlib import Path

import numpy as np

from halcon.drift.monitor import CountAlarm, psi
from halcon.drift.retrain import run_drift
from halcon.simulator import SimConfig, simulate
from halcon.simulator.engine import CARD_TESTING

SMALL = SimConfig(n_customers=1_000, n_merchants=200, days=60, seed=5, card_testing_rate=0.01)


def test_psi_is_zero_for_same_data_and_large_for_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20_000)
    assert psi(ref, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(ref, rng.normal(1.5, 1, 20_000)) > 0.25


def test_psi_counts_missing_values_as_their_own_bin() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0] * 100)
    cur = np.array([1.0, 2.0, np.nan, np.nan] * 100)  # la mitad sin historial
    assert psi(ref, cur) > 0.25


def test_count_alarm_uses_a_poisson_floor() -> None:
    alarm = CountAlarm.fit(np.array([4.0, 4.0, 4.0, 4.0]))  # varianza 0, pero es un conteo
    assert alarm.std == 2.0
    assert not alarm.fires(10) and alarm.fires(13)


def test_drift_keeps_the_past_identical() -> None:
    base = simulate(SMALL).transactions
    cfg = SimConfig(**{**SMALL.__dict__, "drift_day": 30})
    drift = simulate(cfg).transactions
    cut = cfg.drift_ts
    assert cut is not None
    old, new = base.ts < cut, drift.ts < cut
    for a, b in ((base.ts, drift.ts), (base.amount, drift.amount),
                 (base.scenario, drift.scenario), (base.customer, drift.customer)):  # fmt: skip
        np.testing.assert_array_equal(a[old], b[new])


def test_after_drift_probes_dodge_r001_and_attacks_scale_up() -> None:
    cfg = SimConfig(**{**SMALL.__dict__, "drift_day": 25})
    tx = simulate(cfg).transactions
    cut = cfg.drift_ts
    assert cut is not None
    probe = (tx.scenario == CARD_TESTING) & (tx.amount < 10)
    after = probe & (tx.ts >= cut + 1_000)  # las pruebas de un ataque duran < 15 min
    assert after.any() and tx.amount[after].min() >= 2.01
    victims = (tx.scenario == CARD_TESTING) & (tx.amount >= 150)
    before_rate = len(np.unique(tx.customer[victims & (tx.ts < cut)])) / 25
    after_rate = len(np.unique(tx.customer[victims & (tx.ts >= cut)])) / 35
    assert after_rate > 2 * before_rate


def test_run_drift_end_to_end(tmp_path: Path) -> None:
    cfg = SimConfig(**{**SMALL.__dict__, "drift_day": 45})
    report = run_drift(tmp_path, cfg, weeks=2, device="cpu", log=lambda _: None)
    assert len(report.weeks) == 2
    first, second = report.weeks
    # Semana 1: todos usan el modelo del día del ataque.
    assert len(set(first.blocked_pct.values())) == 1
    # Sumar los reclamos frescos nunca le da al modelo MENOS ejemplos del ataque nuevo.
    labels = second.new_attack_labels
    assert labels["+ reclamos frescos"] >= labels["estándar"]
    assert labels["congelado"] == 0  # el modelo del día del ataque no sabía nada de él
    assert {d.day for d in report.days} >= {45, 50}


def test_coverage_alarm_fires_when_bursts_dodge_r001() -> None:
    from halcon.drift.monitor import daily_monitor

    # 28 días, 3 ráfagas diarias. Las 2 primeras semanas R001 las atrapa; después, ninguna.
    day = np.repeat(np.arange(28), 3).astype(np.int64)
    n = len(day)
    cols = {
        "tx_count_10m": np.full(n, 5.0),
        "amount_usd": np.full(n, 2.5),
        "is_online": np.ones(n),
        "hour_local": np.full(n, 12.0),
    }
    r001 = day < 14
    days = daily_monitor(cols, ("hour_local",), np.zeros(n), day, r001, (0, 14), (14, 28))
    by_day = {s.day: s for s in days}
    assert "cobertura R001" not in by_day[14].alarms  # la semana aún tiene ráfagas atrapadas
    assert "cobertura R001" in by_day[18].alarms  # 4 de 7 días sin atrapar: cobertura < 50 %
    assert not any("ráfagas" in s.alarms for s in days)  # el volumen no cambió
    assert not any("PSI features" in s.alarms for s in days)
