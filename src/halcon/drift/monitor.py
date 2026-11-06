"""Monitoreo de drift: ¿nos damos cuenta de que el fraude cambió ANTES de que lleguen los reclamos?

Las etiquetas llegan tarde (en promedio ~15 días). Si solo miramos "fraude atrapado", nos
enteramos de un ataque nuevo semanas después. Por eso en producción se vigilan señales que
NO necesitan etiquetas:

* **PSI por feature** (*Population Stability Index*): compara la distribución de hoy con la de
  un periodo de referencia. Regla usada en la industria: < 0,10 estable, 0,10–0,25 vigilar,
  > 0,25 cambio fuerte. (Se excluye ``day_of_week``: un solo día siempre tiene un solo día de
  la semana, y el PSI saldría enorme sin que nada haya cambiado.)
* **PSI del puntaje** del modelo: si el modelo empieza a "ver" otro mundo, cambia su salida.
* **Ráfagas** (suma de 7 días): compras en línea de menos de $20 con 4 o más compras del
  cliente en 10 minutos. Mira la COLA, no el promedio.
* **Cobertura de R001** (7 días): de esas ráfagas, ¿qué parte atrapa la regla R001? Si las
  ráfagas siguen pero R001 deja de verlas, alguien aprendió a esquivarla.

Lección que este módulo deja medida: el PSI de toda la población no se mueve con un ataque
dirigido a 1 de cada 1 000 clientes. Los monitores de cola y de reglas sí lo ven.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

PSI_ALERT = 0.25
WINDOW_DAYS = 7
COVERAGE_ALERT = 0.5  # menos de la mitad de las ráfagas atrapadas por R001
MIN_BURSTS = 10  # con menos ráfagas en la semana, la cobertura no es confiable
MONITORS = ("PSI features", "PSI puntaje", "ráfagas", "cobertura R001")


def psi(reference: FloatArray, current: FloatArray, bins: int = 10) -> float:
    """PSI con cortes en los cuantiles de la referencia. Los NaN van en su propia caja."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    ref_ok, cur_ok = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    cuts = np.linspace(0, 1, bins + 1)[1:-1]
    edges: FloatArray = np.unique(np.quantile(ref_ok, cuts)) if ref_ok.size else np.empty(0)

    def shares(values: FloatArray, n_nan: int, total: int) -> FloatArray:
        counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
        return np.append(counts, n_nan).astype(np.float64) / total

    p = shares(ref_ok, ref.size - ref_ok.size, ref.size)
    q = shares(cur_ok, cur.size - cur_ok.size, cur.size)
    eps = 1e-4  # evita log(0) cuando una caja queda vacía
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


@dataclass(frozen=True)
class CountAlarm:
    """Se dispara si un conteo se sale de media ± ``k`` desviaciones de la referencia."""

    mean: float
    std: float
    k: float = 3.0

    @classmethod
    def fit(cls, counts: FloatArray, k: float = 3.0) -> CountAlarm:
        c = np.asarray(counts, dtype=np.float64)
        # Piso de Poisson: con conteos chicos, la desviación nunca es menor que √media (ni que 1).
        std = max(float(c.std(ddof=1)) if c.size > 1 else 0.0, float(np.sqrt(c.mean())), 1.0)
        return cls(float(c.mean()), std, k)

    def fires(self, value: float) -> bool:
        return abs(value - self.mean) > self.k * self.std


@dataclass(frozen=True)
class DayStatus:
    day: int
    max_feature_psi: float
    worst_feature: str
    score_psi: float
    bursts_7d: int
    r001_coverage_7d: float | None
    alarms: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rolling(daily: FloatArray) -> FloatArray:
    """Suma de los últimos ``WINDOW_DAYS`` días (incluido el actual)."""
    c = np.concatenate([[0.0], np.cumsum(daily)])
    idx = np.arange(len(daily))
    out: FloatArray = c[idx + 1] - c[np.maximum(0, idx + 1 - WINDOW_DAYS)]
    return out


def daily_monitor(
    columns: dict[str, FloatArray],
    feature_names: tuple[str, ...],
    scores: FloatArray,
    day: IntArray,
    r001: BoolArray,
    ref_days: tuple[int, int],
    watch_days: tuple[int, int],
) -> list[DayStatus]:
    """Compara cada día de ``watch_days`` contra la referencia ``ref_days`` (ambos [desde, hasta)).

    No usa etiquetas: solo lo que el banco ve en el momento de autorizar.
    """
    c = columns
    burst = (c["tx_count_10m"] >= 4) & (c["amount_usd"] < 20) & (c["is_online"] == 1)
    n_days = int(day.max()) + 1
    bursts = np.bincount(day[burst], minlength=n_days).astype(np.float64)
    caught = np.bincount(day[burst & r001], minlength=n_days).astype(np.float64)
    bursts_7d, caught_7d = _rolling(bursts), _rolling(caught)

    # Referencia: semanas enteras dentro de ref_days.
    full_weeks = np.arange(ref_days[0] + WINDOW_DAYS - 1, ref_days[1])
    burst_alarm = CountAlarm.fit(bursts_7d[full_weeks])
    ref = (day >= ref_days[0]) & (day < ref_days[1])
    monitored = tuple(f for f in feature_names if f != "day_of_week")

    out = []
    for d in range(*watch_days):
        today = day == d
        if not today.any():
            continue
        feature_psi = {f: psi(c[f][ref], c[f][today]) for f in monitored}
        worst = max(feature_psi, key=lambda f: feature_psi[f])
        s_psi = psi(scores[ref], scores[today])
        n7 = int(bursts_7d[d])
        coverage = round(float(caught_7d[d] / n7), 3) if n7 else None
        alarms = []
        if feature_psi[worst] > PSI_ALERT:
            alarms.append("PSI features")
        if s_psi > PSI_ALERT:
            alarms.append("PSI puntaje")
        if burst_alarm.fires(n7):
            alarms.append("ráfagas")
        if coverage is not None and n7 >= MIN_BURSTS and coverage < COVERAGE_ALERT:
            alarms.append("cobertura R001")
        out.append(DayStatus(d, round(feature_psi[worst], 4), worst, round(s_psi, 4), n7,
                             coverage, tuple(alarms)))  # fmt: skip
    return out


def first_alarm(days: list[DayStatus], name: str, since: int) -> int | None:
    """Primer día >= ``since`` en que suena la alarma ``name``."""
    for s in days:
        if s.day >= since and name in s.alarms:
            return s.day
    return None
