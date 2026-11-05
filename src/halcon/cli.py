"""Línea de comandos de Halcón."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from halcon import __version__


def _cmd_simulate(args: argparse.Namespace) -> int:
    from halcon.simulator import SimConfig, simulate, summary, write_parquet

    cfg = SimConfig(
        n_customers=args.customers,
        n_merchants=args.merchants,
        days=args.days,
        start=date.fromisoformat(args.start),
        seed=args.seed,
    )
    t0 = time.perf_counter()
    result = simulate(cfg)
    paths = write_parquet(result, Path(args.out))
    elapsed = time.perf_counter() - t0

    stats = summary(result)
    print(f"Simulación lista en {elapsed:.1f} s (semilla {cfg.seed})")
    for key, value in stats.items():
        print(f"  {key:<18} {value:>12,}")
    for name, path in paths.items():
        print(f"  → {name:<13} {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    from halcon.model.features import build_features

    t0 = time.perf_counter()
    out = Path(args.out)
    rows = build_features(Path(args.raw), out)
    print(f"Features listos en {time.perf_counter() - t0:.1f} s: {rows:,} filas → {out}")
    return 0


def _pct(value: float) -> str:
    return f"{100 * value:5.1f} %"


def _cmd_train(args: argparse.Namespace) -> int:
    from halcon.model.train import SplitConfig, train

    split = SplitConfig(alerts_per_day=args.alerts_per_day)
    report = train(Path(args.features), Path(args.model_dir), split, device=args.device)
    print(f"Entrenado en {report.train_seconds} s con {report.device.upper()}")
    print(
        f"  train: {report.train_rows:,} filas, {report.train_frauds_known} fraudes conocidos, "
        f"{report.train_frauds_hidden} aún sin reportar al corte"
    )
    print(f"  test:  {report.test_rows:,} filas, {report.test_frauds} fraudes")
    print(f"\n  Con {args.alerts_per_day} alertas por día:")
    print(f"  {'modelo':<24}{'PR-AUC':>8}{'precisión':>11}{'recall':>9}{'$ atrapado':>12}")
    for ev in (report.baseline, report.model, report.oracle):
        b = ev.budget
        print(
            f"  {ev.name:<24}{ev.pr_auc:>8.3f}{_pct(b.precision):>11}"
            f"{_pct(b.recall):>9}{_pct(b.amount_recall):>12}"
        )
    print("\n  Recall por escenario (xgboost):")
    for name, value in report.model.budget.recall_by_scenario.items():
        print(f"    {name:<20}{_pct(value)}")
    print("\n  Features más importantes:")
    for name, value in list(report.feature_importance.items())[:5]:
        print(f"    {name:<24}{_pct(value)}")
    print(f"\n  → {Path(args.model_dir) / 'xgb.json'} y metrics.json")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="halcon", description="Detección de fraude en pagos.")
    parser.add_argument("--version", action="version", version=f"halcon {__version__}")
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", help="Genera transacciones sintéticas con fraude.")
    sim.add_argument("--customers", type=int, default=5_000)
    sim.add_argument("--merchants", type=int, default=2_000)
    sim.add_argument("--days", type=int, default=120)
    sim.add_argument("--start", default="2026-01-01", help="Primer día (AAAA-MM-DD).")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--out", default="data/raw", help="Carpeta de salida.")
    sim.set_defaults(func=_cmd_simulate)

    feat = sub.add_parser("features", help="Calcula features (solo información del pasado).")
    feat.add_argument("--raw", default="data/raw", help="Carpeta con los Parquet del simulador.")
    feat.add_argument("--out", default="data/features/features.parquet")
    feat.set_defaults(func=_cmd_features)

    tr = sub.add_parser("train", help="Entrena XGBoost con split temporal y feedback delay.")
    tr.add_argument("--features", default="data/features/features.parquet")
    tr.add_argument("--model-dir", default="models")
    tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    tr.add_argument("--alerts-per-day", type=int, default=50)
    tr.set_defaults(func=_cmd_train)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    code: int = args.func(args)
    return code
