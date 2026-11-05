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
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    code: int = args.func(args)
    return code
