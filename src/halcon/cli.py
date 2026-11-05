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
    th = report.thresholds
    print(
        f"\n  Política calibrada en validación: revisar si puntaje ≥ {th.review:.3f}, "
        f"rechazar si ≥ {th.decline:.3f} (o si lo pide una regla)"
    )
    print(
        f"  {'política':<16}{'costo total':>13}{'pérdida fraude':>16}{'revisiones/día':>16}"
        f"{'rechazos injustos':>19}{'fraude detenido':>17}"
    )
    for p in report.policies:
        print(
            f"  {p.name:<16}{p.total_cost:>13,.0f}{p.fraud_loss:>16,.0f}{p.reviews_per_day:>16}"
            f"{p.false_declines:>19}{p.frauds_stopped_pct:>16} %"
        )
    print(f"\n  → {Path(args.model_dir)}/xgb.json, policy.json y metrics.json")
    return 0


_ACTION_LABEL = {"APPROVE": "✅ APROBAR", "REVIEW": "🔍 REVISAR", "DECLINE": "⛔ RECHAZAR"}


def _cmd_score(args: argparse.Namespace) -> int:
    import duckdb

    from halcon.decision.scorer import Scorer
    from halcon.model.features import FEATURES

    con = duckdb.connect()
    path = Path(args.features).as_posix()
    cols = ", ".join(FEATURES)
    if args.tx_id is not None:
        where, params = "tx_id = ?", [path, args.tx_id]
    else:
        # Ejemplos del periodo de test (últimos 30 días): el modelo nunca los vio.
        recent = "ts >= (SELECT max(ts) - INTERVAL 30 DAY FROM read_parquet(?))"
        if args.sample == "legit":
            where, params = f"fraud_scenario IS NULL AND {recent}", [path, path]
        else:
            where, params = f"fraud_scenario = ? AND {recent}", [path, args.sample, path]
    sql = f"""
        SELECT tx_id, category, coalesce(fraud_scenario, 'legit') AS truth, {cols},
               strftime(ts AT TIME ZONE 'America/Guayaquil', '%Y-%m-%d %H:%M') AS ts_local
        FROM read_parquet(?)
        WHERE {where}
        ORDER BY hash(tx_id + {int(args.seed)})
        LIMIT 1
    """  # noqa: S608 — solo se interpolan constantes; los valores van como parámetros
    found = con.execute(sql, params).fetchone()
    if found is None:
        print("No se encontró esa transacción.")
        return 1
    names = [d[0] for d in con.description or []]
    row = dict(zip(names, found, strict=True))

    scorer = Scorer.load(Path(args.model_dir))
    decision = scorer.score({f: row[f] for f in FEATURES})
    print(
        f"Transacción {row['tx_id']} · {row['ts_local']} (Ecuador) · {row['category']} · "
        f"${row['amount_usd']:,.2f}"
    )
    print(
        f"  Decisión: {_ACTION_LABEL[decision.action.name]}   puntaje {decision.score:.4f}   "
        f"({decision.latency_ms} ms)"
    )
    for code in decision.rules:
        print(f"  Regla {code}: {scorer.rule_description(code)}")
    if decision.reasons and decision.action.name != "APPROVE":
        print("  Pistas que más subieron el riesgo:")
        for r in decision.reasons:
            print(f"    {r.feature:<24} = {r.value:>10.2f}   (+{r.contribution:.2f})")
    print(f"  Verdad (solo para la demo): {row['truth']}")
    return 0


def _cmd_stream(args: argparse.Namespace) -> int:
    from halcon.stream import service as svc

    raw = Path(args.raw)
    start = svc.default_stream_start(raw)
    if args.action == "init":
        parts = svc.init_topics(args.bootstrap, reset=args.reset)
        print("Topics listos: " + ", ".join(f"{t} ({n} particiones)" for t, n in parts.items()))
        from halcon.stream.store import ValkeyStore

        store = ValkeyStore(svc.connect_valkey(args.valkey))
        t0 = time.perf_counter()
        n = svc.warmup(raw, store, start)
        print(f"Valkey precargado con {n:,} compras de los 30 días previos "
              f"({time.perf_counter() - t0:.1f} s).")  # fmt: skip
    elif args.action == "produce":
        svc.produce(raw, args.bootstrap, start, args.limit, args.rate)
    elif args.action == "score":
        svc.run_scorer(args.bootstrap, args.valkey, raw, Path(args.model_dir), args.limit,
                       args.idle, metrics_port=args.metrics_port)  # fmt: skip
    elif args.action == "summary":
        svc.summarize(args.bootstrap, raw)
    elif args.action == "bench":
        svc.bench(raw, Path(args.model_dir), args.limit or 20_000,
                  args.valkey if args.with_valkey else None)  # fmt: skip
    return 0


def _print_agent(case_id: int, output: dict[str, object], problems: list[str], ms: float) -> None:
    rec = {"fraude_probable": "⛔ FRAUDE PROBABLE", "legitimo_probable": "✅ LEGÍTIMO PROBABLE",
           "incierto": "❔ INCIERTO"}.get(str(output.get("recomendacion")), "—")  # fmt: skip
    print(f"\n  Agente sobre el caso {case_id} ({ms / 1000:.1f} s)")
    print(f"  Recomendación: {rec} · confianza {output.get('confianza', '—')} · "
          f"patrón: {output.get('patron', '—')}")  # fmt: skip
    print(f"  Resumen: {output.get('resumen', '')}")
    hallazgos = output.get("hallazgos")
    for h in hallazgos if isinstance(hallazgos, list) else []:
        print(f"    • {h.get('texto', '')} {h.get('evidencia', [])}")
    print(f"  Siguiente paso: {output.get('siguiente_paso', '')}")
    print("  Barandas: " + ("✓ citas y números verificados" if not problems
                            else "⚠ " + "; ".join(problems)))  # fmt: skip


def _make_llm(args: argparse.Namespace) -> object:
    from halcon.cases.agent import HeuristicLLM, OllamaClient

    if args.llm == "heuristica":
        return HeuristicLLM()
    return OllamaClient(model=args.model, base_url=args.ollama)


def _cmd_cases(args: argparse.Namespace) -> int:
    from halcon.cases.store import CaseStore

    store = CaseStore(Path(args.db))
    raw = Path(args.raw)
    try:
        if args.action == "ingest":
            if args.source == "topic":
                from halcon.cases.ingest import from_topic

                from_topic(args.bootstrap, store)
            else:
                from halcon.cases.ingest import from_test_period

                n = from_test_period(raw, Path(args.features), Path(args.model_dir), store,
                                     days=args.days, max_cases=args.limit)  # fmt: skip
                print(f"{n} casos nuevos en la cola (últimos {args.days} días, sin Docker).")
            print("Estado de la cola: " + ", ".join(f"{k}={v}" for k, v in store.counts().items()))
        elif args.action == "list":
            for c in store.list_cases(limit=args.limit or 15):
                flag = "🤖" if c.agent else "  "
                print(f"{flag} {c.tx_id:>8}  {c.action:<8} puntaje {c.score:.3f}  "
                      f"${c.amount_usd:>9,.2f}  {c.category:<14} {c.channel:<6} "
                      f"{','.join(c.rules) or '-'}")  # fmt: skip
        elif args.action == "investigate":
            from halcon.cases.agent import investigate
            from halcon.cases.evidence import EvidenceBuilder

            llm = _make_llm(args)
            builder = EvidenceBuilder(raw)
            if args.tx_id is not None:
                targets = [store.get(args.tx_id)]
            else:
                targets = list(store.list_cases(limit=args.limit or 1, without_agent=True))
            for case in targets:
                if case is None:
                    print("No existe ese caso.")
                    return 1
                evidence = builder.build(case)
                if args.show_evidence:
                    print("\n".join(e.render() for e in evidence))
                res = investigate(llm, case.tx_id, evidence)  # type: ignore[arg-type]
                store.save_agent(case.tx_id, res.output, getattr(llm, "name", "?"), res.ms,
                                 res.problems)  # fmt: skip
                _print_agent(case.tx_id, res.output, res.problems, res.ms)
                model_says = case.model_view
                print(f"  Modelo XGBoost: puntaje {case.score:.4f} → {model_says}")
                if res.output.get("recomendacion") not in (model_says, "incierto"):
                    print("  ⚠ DESACUERDO: el agente contradice al modelo. Revisar con atención.")
        elif args.action == "resolve":
            if args.tx_id is None or args.as_ is None:
                print("Usa: halcon cases resolve --tx-id N --as fraude|legitimo")
                return 1
            store.resolve(args.tx_id, args.as_, args.note)
            print(f"Caso {args.tx_id} cerrado como {args.as_}.")
        elif args.action == "evaluate":
            from halcon.cases.evaluate import evaluate

            ev = evaluate(store, raw, _make_llm(args), n=args.limit or 40)  # type: ignore[arg-type]
            print(f"\nEvaluación del agente ({ev.model}) sobre {ev.cases} casos "
                  f"({ev.frauds} fraudes reales):")  # fmt: skip
            print(f"  Acierto de la recomendación: {ev.accuracy} % (se anima en el "
                  f"{ev.coverage} % de los casos)")  # fmt: skip
            print(f"    fraudes reconocidos: {ev.fraud_recall} % · legítimos reconocidos: "
                  f"{ev.legit_recall} %")  # fmt: skip
            print(f"  Acierto del patrón: {ev.pattern_accuracy} %")
            print(f"  Respuestas sin problemas: {ev.grounded_pct} %")
            print(f"    con números inventados: {ev.invented_numbers_pct} % · "
                  f"con citas inválidas: {ev.bad_citations_pct} %")  # fmt: skip
            print(f"  Latencia por caso: p50 {ev.latency_p50_ms / 1000:.1f} s · "
                  f"p95 {ev.latency_p95_ms / 1000:.1f} s")  # fmt: skip
            print(
                f"  Comparación: heurística sin IA (puntaje ≥ 0,5) acierta {ev.baseline_accuracy} %"
            )
            print(f"  Desacuerdos agente vs. modelo: {ev.disagreements} · el agente tenía razón "
                  f"en {ev.agent_right_in_disagreements}")  # fmt: skip
    finally:
        store.close()
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

    sc = sub.add_parser("score", help="Decide una transacción y explica por qué.")
    group = sc.add_mutually_exclusive_group()
    group.add_argument("--tx-id", type=int)
    group.add_argument(
        "--sample",
        choices=["legit", "card_testing", "account_takeover", "skimming"],
        default="card_testing",
        help="Elige una transacción de ejemplo de ese tipo.",
    )
    sc.add_argument("--seed", type=int, default=0, help="Cambia el ejemplo elegido.")
    sc.add_argument("--features", default="data/features/features.parquet")
    sc.add_argument("--model-dir", default="models")
    sc.set_defaults(func=_cmd_score)

    st = sub.add_parser("stream", help="Tiempo real con Redpanda y Valkey.")
    st.add_argument(
        "action",
        choices=["init", "produce", "score", "summary", "bench"],
        help="init: topics y precarga · produce: envía pagos · score: decide en vivo · "
        "summary: cruza con la verdad · bench: mide sin broker",
    )
    st.add_argument("--bootstrap", default="localhost:19092")
    st.add_argument("--valkey", default="valkey://localhost:6379/0")
    st.add_argument("--raw", default="data/raw")
    st.add_argument("--model-dir", default="models")
    st.add_argument("--rate", type=float, default=500, help="Pagos por segundo (0 = sin freno).")
    st.add_argument("--limit", type=int, default=None, help="Máximo de mensajes.")
    st.add_argument(
        "--idle", type=float, default=20, help="Segundos sin mensajes para terminar (0 = nunca)."
    )
    st.add_argument("--metrics-port", type=int, default=None,
                    help="score: expone /metrics para Prometheus en este puerto.")  # fmt: skip
    st.add_argument("--reset", action="store_true", help="init: borra topics y vuelve a crearlos.")
    st.add_argument("--with-valkey", action="store_true", help="bench: incluye Valkey.")
    st.set_defaults(func=_cmd_stream)

    cs = sub.add_parser("cases", help="Cola de casos y agente investigador.")
    cs.add_argument("action", choices=["ingest", "list", "investigate", "resolve", "evaluate"])
    cs.add_argument("--db", default="data/cases.duckdb")
    cs.add_argument("--raw", default="data/raw")
    cs.add_argument("--features", default="data/features/features.parquet")
    cs.add_argument("--model-dir", default="models")
    cs.add_argument(
        "--source",
        choices=["test", "topic"],
        default="test",
        help="ingest: 'test' decide los últimos días sin Docker; 'topic' lee Redpanda",
    )
    cs.add_argument("--bootstrap", default="localhost:19092")
    cs.add_argument("--days", type=int, default=30)
    cs.add_argument("--limit", type=int, default=None)
    cs.add_argument("--tx-id", type=int, default=None)
    cs.add_argument("--as", dest="as_", choices=["fraude", "legitimo"], default=None)
    cs.add_argument("--note", default=None)
    cs.add_argument("--llm", choices=["ollama", "heuristica"], default="ollama")
    cs.add_argument("--model", default="qwen2.5:7b")
    cs.add_argument("--ollama", default="http://localhost:11434")
    cs.add_argument("--show-evidence", action="store_true")
    cs.set_defaults(func=_cmd_cases)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    code: int = args.func(args)
    return code
