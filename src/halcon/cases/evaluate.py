"""¿El agente ayuda o solo redacta bonito? Se mide contra la verdad del simulador.

La verdad (``is_fraud``, ``fraud_scenario``) se lee AQUÍ y solo aquí, después de que el agente
respondió. El agente nunca la ve.

Métricas:

* **Acierto de la recomendación** (sin contar "incierto") y **cobertura** (cuántas veces se
  anima a recomendar).
* **Acierto del patrón**: ¿reconoce card testing, toma de cuenta, tarjeta clonada?
* **Barandas**: % de respuestas sin problemas, con citas inválidas o con números inventados.
* **Latencia** por caso en tu GPU.
* **Comparación** con la heurística sin IA (puntaje ≥ 0,5): si el LLM no le gana, su valor está
  en ahorrar tiempo de lectura al analista, no en decidir mejor. Hay que decirlo así.
* **Desacuerdos:** cuando el agente contradice al modelo, ¿quién tiene razón? Si el agente casi
  nunca acierta en esos casos, su "opinión" no aporta: solo debe redactar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import numpy as np

from halcon.cases.agent import LLM, AgentResult, HeuristicLLM, investigate
from halcon.cases.evidence import EvidenceBuilder
from halcon.cases.store import CaseStore

PATTERN_OF_SCENARIO = {
    "card_testing": "card_testing",
    "account_takeover": "toma_de_cuenta",
    "skimming": "tarjeta_clonada",
    "legit": "ninguno",
}
Printer = Callable[[str], None]


@dataclass
class AgentEval:
    model: str
    cases: int
    frauds: int
    coverage: float
    accuracy: float
    fraud_recall: float
    legit_recall: float
    pattern_accuracy: float
    grounded_pct: float
    invented_numbers_pct: float
    bad_citations_pct: float
    latency_p50_ms: float
    latency_p95_ms: float
    baseline_accuracy: float
    disagreements: int
    agent_right_in_disagreements: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _truth(raw_dir: Path, ids: list[int]) -> dict[int, str]:
    rows = (
        duckdb.connect()
        .execute(
            """SELECT tx_id, coalesce(fraud_scenario, 'legit') FROM read_parquet(?)
           WHERE tx_id IN (SELECT unnest(?::BIGINT[]))""",
            [(raw_dir / "transactions.parquet").as_posix(), ids],
        )
        .fetchall()
    )
    return {int(t): str(s) for t, s in rows}


def sample_cases(store: CaseStore, raw_dir: Path, n: int, seed: int = 7) -> list[int]:
    """Muestra mitad fraude y mitad legítimos (si hay), para medir ambos lados."""
    ids = store.all_ids()
    truth = _truth(raw_dir, ids)
    rng = np.random.default_rng(seed)
    fraud = [i for i in ids if truth.get(i, "legit") != "legit"]
    legit = [i for i in ids if truth.get(i, "legit") == "legit"]
    k_f = min(len(fraud), n // 2)
    k_l = min(len(legit), n - k_f)
    picked = list(rng.choice(fraud, k_f, replace=False)) + list(
        rng.choice(legit, k_l, replace=False)
    )
    return sorted(int(i) for i in picked)


def evaluate(store: CaseStore, raw_dir: Path, llm: LLM, n: int = 40, seed: int = 7,
             log: Printer = print, reuse: bool = True) -> AgentEval:  # fmt: skip
    """``reuse=True`` aprovecha respuestas ya guardadas del mismo modelo (no repite la GPU)."""
    ids = sample_cases(store, raw_dir, n, seed)
    truth = _truth(raw_dir, ids)
    builder = EvidenceBuilder(raw_dir)
    baseline = HeuristicLLM()
    rows = []
    for k, tx_id in enumerate(ids, 1):
        case = store.get(tx_id)
        if case is None:
            continue
        evidence = builder.build(case)
        if reuse and case.agent and case.agent_model == llm.name:
            res = AgentResult(case.agent, case.agent_problems, case.agent_ms)
        else:
            res = investigate(llm, tx_id, evidence)
            store.save_agent(tx_id, res.output, llm.name, res.ms, res.problems)
        base = investigate(baseline, tx_id, evidence)
        rows.append((tx_id, truth[tx_id], res, base.output.get("recomendacion")))
        warn = " ⚠ " + "; ".join(res.problems) if res.problems else ""
        rec = res.output.get("recomendacion", "—")
        log(f"  [{k}/{len(ids)}] caso {tx_id}: {rec} ({res.ms / 1000:.1f} s){warn}")

    def is_fraud(s: str) -> bool:
        return s != "legit"

    decided = [(t, r) for _, t, r, _ in rows if r.output.get("recomendacion") != "incierto"]
    correct = [
        (r.output.get("recomendacion") == "fraude_probable") == is_fraud(t) for t, r in decided
    ]
    fraud_rows = [(t, r) for _, t, r, _ in rows if is_fraud(t)]
    legit_rows = [(t, r) for _, t, r, _ in rows if not is_fraud(t)]
    lat = [r.ms for _, _, r, _ in rows] or [0.0]

    def pct(part: int, total: int) -> float:
        return round(100 * part / total, 1) if total else 0.0

    disagree = [
        (t, r) for _, t, r, b in rows if r.output.get("recomendacion") not in (b, "incierto")
    ]
    agent_right = sum(
        (r.output.get("recomendacion") == "fraude_probable") == is_fraud(t) for t, r in disagree
    )

    result = AgentEval(
        model=llm.name,
        cases=len(rows),
        frauds=len(fraud_rows),
        coverage=pct(len(decided), len(rows)),
        accuracy=pct(sum(correct), len(decided)),
        fraud_recall=pct(
            sum(r.output.get("recomendacion") == "fraude_probable" for _, r in fraud_rows),
            len(fraud_rows),
        ),  # fmt: skip
        legit_recall=pct(
            sum(r.output.get("recomendacion") == "legitimo_probable" for _, r in legit_rows),
            len(legit_rows),
        ),  # fmt: skip
        pattern_accuracy=pct(
            sum(r.output.get("patron") == PATTERN_OF_SCENARIO[t] for _, t, r, _ in rows), len(rows)
        ),  # fmt: skip
        grounded_pct=pct(sum(r.ok for _, _, r, _ in rows), len(rows)),
        invented_numbers_pct=pct(
            sum(any(p.startswith("números") for p in r.problems) for _, _, r, _ in rows), len(rows)
        ),  # fmt: skip
        bad_citations_pct=pct(
            sum(any("cita" in p for p in r.problems) for _, _, r, _ in rows), len(rows)
        ),  # fmt: skip
        latency_p50_ms=round(float(np.percentile(lat, 50)), 1),
        latency_p95_ms=round(float(np.percentile(lat, 95)), 1),
        disagreements=len(disagree),
        agent_right_in_disagreements=int(agent_right),
        baseline_accuracy=pct(
            sum((b == "fraude_probable") == is_fraud(t) for _, t, _, b in rows), len(rows)
        ),  # fmt: skip
    )
    return result
