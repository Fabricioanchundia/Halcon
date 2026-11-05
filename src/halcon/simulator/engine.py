"""Motor del simulador de transacciones con tarjeta.

Diseño propio, inspirado en las ideas del simulador del libro *Reproducible Machine Learning
for Credit Card Fraud Detection — Practical Handbook* (ULB / Worldline): clientes y comercios
con propiedades fijas, transacciones legítimas generadas a partir de esas propiedades y
escenarios de fraude inyectados encima. No contiene código de ese libro.

Tres escenarios de fraude, cada uno deja una "huella" distinta que el modelo deberá aprender:

* ``card_testing``: ráfaga de compras diminutas en un comercio digital (el ladrón prueba si
  la tarjeta funciona) seguida de compras grandes en línea.
* ``account_takeover``: varias compras en línea muy por encima del gasto normal del cliente
  en pocas horas.
* ``skimming``: un datáfono (POS) es comprometido durante dos semanas; días después, las
  tarjetas clonadas se usan en persona en otra ciudad.

Cada fraude trae ``fraud_reported_at``: el momento en que el banco se entera (reclamo del
cliente o contracargo). Antes de esa fecha la etiqueta NO existe; la evaluación del modelo
debe respetarlo (*feedback delay*).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
import numpy.typing as npt

from halcon.simulator.catalog import CATEGORIES, CITIES

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

DAY = 86_400
HOUR = 3_600
ECUADOR_UTC_OFFSET_HOURS = 5  # Ecuador continental: UTC-5, sin horario de verano

SCENARIOS: tuple[str, ...] = ("legit", "card_testing", "account_takeover", "skimming")
LEGIT, CARD_TESTING, ACCOUNT_TAKEOVER, SKIMMING = range(4)

# Probabilidad de comprar a cada hora (hora local). Poca actividad de madrugada.
_HOUR_WEIGHTS = np.array(
    [
        1,
        0.5,
        0.3,
        0.2,
        0.2,
        0.4,
        1,
        2,
        3.5,
        4.5,
        5,
        5.5,
        6.5,
        6.5,
        5.5,
        5,
        5,
        5.5,
        6,
        6,
        5,
        4,
        3,
        2,
    ],
    dtype=np.float64,
)
HOUR_PROBS: FloatArray = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()

TRAVEL_PROB = 0.07  # probabilidad de que una compra presencial ocurra fuera de la ciudad de origen
SKIMMING_OTHER_CITY_PROB = 0.7  # tarjetas clonadas usadas fuera de la ciudad de la víctima


@dataclass(frozen=True)
class SimConfig:
    """Parámetros de la simulación. Las tasas de fraude son por cada 30 días."""

    n_customers: int = 5_000
    n_merchants: int = 2_000
    days: int = 120
    start: date = date(2026, 1, 1)
    seed: int = 42
    card_testing_rate: float = 0.002  # víctimas por cliente cada 30 días
    takeover_rate: float = 0.003  # víctimas por cliente cada 30 días
    skimming_rate: float = 0.004  # datáfonos comprometidos por comercio POS cada 30 días

    def __post_init__(self) -> None:
        if self.n_customers < 10:
            raise ValueError("n_customers debe ser al menos 10")
        if self.n_merchants < 100:
            raise ValueError("n_merchants debe ser al menos 100")
        if self.days < 15:
            raise ValueError("days debe ser al menos 15 (el skimming usa ventanas de 14 días)")

    @property
    def start_ts(self) -> int:
        """Medianoche (hora de Ecuador) del primer día, en segundos Unix UTC."""
        midnight = datetime(self.start.year, self.start.month, self.start.day, tzinfo=UTC)
        return int((midnight + timedelta(hours=ECUADOR_UTC_OFFSET_HOURS)).timestamp())

    @property
    def end_ts(self) -> int:
        return self.start_ts + self.days * DAY


@dataclass
class Customers:
    customer_id: IntArray
    home_city: IntArray
    # Propiedades OCULTAS: generan el comportamiento pero NO se exportan (serían trampa).
    mean_amount: FloatArray
    tx_per_day: FloatArray
    online_share: FloatArray


@dataclass
class Merchants:
    merchant_id: IntArray
    category: IntArray
    city: IntArray  # -1 para comercios en línea


@dataclass
class Transactions:
    ts: IntArray
    customer: IntArray
    merchant: IntArray
    amount: FloatArray
    scenario: IntArray
    reported_at: IntArray  # -1 si es legítima

    @classmethod
    def empty(cls) -> Transactions:
        e = np.empty(0, dtype=np.int64)
        return cls(e, e, e, np.empty(0, dtype=np.float64), e, e)

    def __len__(self) -> int:
        return len(self.ts)


@dataclass
class SimResult:
    config: SimConfig
    customers: Customers
    merchants: Merchants
    transactions: Transactions
    compromised_merchants: list[int] = field(default_factory=list)


_CAT_NAMES = [c.name for c in CATEGORIES]
_CAT_CHANNEL = np.array([c.channel == "online" for c in CATEGORIES])
_CAT_MULT = np.array([c.amount_multiplier for c in CATEGORIES], dtype=np.float64)
_CITY_PROBS = np.array([c.weight for c in CITIES], dtype=np.float64)
_CITY_PROBS /= _CITY_PROBS.sum()


def _cat(name: str) -> int:
    return _CAT_NAMES.index(name)


def build_customers(cfg: SimConfig, rng: np.random.Generator) -> Customers:
    n = cfg.n_customers
    return Customers(
        customer_id=np.arange(n, dtype=np.int64),
        home_city=rng.choice(len(CITIES), size=n, p=_CITY_PROBS).astype(np.int64),
        mean_amount=rng.lognormal(mean=np.log(25.0), sigma=0.5, size=n),
        tx_per_day=rng.uniform(0.3, 3.0, size=n),
        online_share=rng.beta(2.0, 5.0, size=n),
    )


def build_merchants(cfg: SimConfig, rng: np.random.Generator) -> Merchants:
    n = cfg.n_merchants
    cat_probs = np.array([c.merchant_weight for c in CATEGORIES], dtype=np.float64)
    category = rng.choice(len(CATEGORIES), size=n, p=cat_probs / cat_probs.sum()).astype(np.int64)
    # Garantiza al menos un comercio de cada categoría.
    category[: len(CATEGORIES)] = np.arange(len(CATEGORIES))

    city = np.full(n, -1, dtype=np.int64)
    pos_idx = np.flatnonzero(~_CAT_CHANNEL[category])
    # Cada ciudad recibe al menos 3 comercios presenciales; el resto según población.
    min_per_city = 3
    n_fixed = min(len(pos_idx), min_per_city * len(CITIES))
    city[pos_idx[:n_fixed]] = np.arange(n_fixed) % len(CITIES)
    rest = pos_idx[n_fixed:]
    city[rest] = rng.choice(len(CITIES), size=len(rest), p=_CITY_PROBS)
    return Merchants(merchant_id=np.arange(n, dtype=np.int64), category=category, city=city)


def generate_legit(
    cfg: SimConfig, rng: np.random.Generator, cust: Customers, merch: Merchants
) -> Transactions:
    counts = rng.poisson(cust.tx_per_day * cfg.days)
    who = np.repeat(cust.customer_id, counts)
    n = len(who)

    day = rng.integers(0, cfg.days, size=n)
    hour = rng.choice(24, size=n, p=HOUR_PROBS)
    ts = cfg.start_ts + day * DAY + hour * HOUR + rng.integers(0, HOUR, size=n)

    online = rng.random(n) < cust.online_share[who]
    traveling = rng.random(n) < TRAVEL_PROB
    tx_city = np.where(
        traveling, rng.choice(len(CITIES), size=n, p=_CITY_PROBS), cust.home_city[who]
    )

    merchant = np.empty(n, dtype=np.int64)
    online_pool = merch.merchant_id[merch.city == -1]
    mask = online
    merchant[mask] = online_pool[rng.integers(0, len(online_pool), size=int(mask.sum()))]
    for c in range(len(CITIES)):
        pool = merch.merchant_id[merch.city == c]
        mask = ~online & (tx_city == c)
        merchant[mask] = pool[rng.integers(0, len(pool), size=int(mask.sum()))]

    amount = cust.mean_amount[who] * _CAT_MULT[merch.category[merchant]]
    amount = np.round(np.clip(amount * rng.lognormal(0.0, 0.6, size=n), 0.5, None), 2)

    return Transactions(
        ts=ts.astype(np.int64),
        customer=who.astype(np.int64),
        merchant=merchant,
        amount=amount,
        scenario=np.zeros(n, dtype=np.int64),
        reported_at=np.full(n, -1, dtype=np.int64),
    )


class _FraudBuilder:
    """Acumula filas de fraude y las convierte en arrays al final."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.rows: dict[str, list[IntArray | FloatArray]] = {
            k: [] for k in ("ts", "customer", "merchant", "amount", "scenario", "reported_at")
        }

    def start_time(self, night_bias: float) -> int:
        """Inicio de un ataque. Con prob. ``night_bias`` la hora es uniforme (bots de
        madrugada); si no, sigue el horario normal de compras para pasar desapercibido."""
        day = int(self.rng.integers(0, self.cfg.days - 1))
        if self.rng.random() < night_bias:
            hour = int(self.rng.integers(0, 24))
        else:
            hour = int(self.rng.choice(24, p=HOUR_PROBS))
        return self.cfg.start_ts + day * DAY + hour * HOUR + int(self.rng.integers(0, HOUR))

    def report_delay(self) -> int:
        """Días hasta que el banco se entera: reclamo o contracargo (1 a ~45 días)."""
        return int((1.0 + self.rng.gamma(shape=2.0, scale=7.0)) * DAY)

    def add(
        self,
        scenario: int,
        customer: int,
        ts: IntArray,
        merchant: IntArray,
        amount: FloatArray,
    ) -> None:
        keep = ts < self.cfg.end_ts
        if not keep.any():
            return
        ts, merchant, amount = ts[keep], merchant[keep], amount[keep]
        n = len(ts)
        reported = int(ts.max()) + self.report_delay()
        self.rows["ts"].append(ts.astype(np.int64))
        self.rows["customer"].append(np.full(n, customer, dtype=np.int64))
        self.rows["merchant"].append(merchant.astype(np.int64))
        self.rows["amount"].append(np.round(amount, 2))
        self.rows["scenario"].append(np.full(n, scenario, dtype=np.int64))
        self.rows["reported_at"].append(np.full(n, reported, dtype=np.int64))

    def build(self) -> Transactions:
        if not self.rows["ts"]:
            return Transactions.empty()
        cols = {k: np.concatenate(v) for k, v in self.rows.items()}
        return Transactions(
            ts=cols["ts"].astype(np.int64),
            customer=cols["customer"].astype(np.int64),
            merchant=cols["merchant"].astype(np.int64),
            amount=cols["amount"].astype(np.float64),
            scenario=cols["scenario"].astype(np.int64),
            reported_at=cols["reported_at"].astype(np.int64),
        )


def _n_events(rng: np.random.Generator, rate: float, population: int, days: int) -> int:
    return min(population, int(rng.poisson(rate * population * days / 30)))


def inject_fraud(
    cfg: SimConfig,
    rng: np.random.Generator,
    cust: Customers,
    merch: Merchants,
    legit: Transactions,
) -> tuple[Transactions, list[int]]:
    fb = _FraudBuilder(cfg, rng)
    ids = merch.merchant_id
    cats = merch.category
    digital = ids[cats == _cat("digital_goods")]
    big_online = ids[np.isin(cats, [_cat("online_retail"), _cat("travel")])]
    any_online = ids[merch.city == -1]

    # 1) Card testing: pruebas de $0.50–$2 cada pocos segundos, luego compras grandes.
    n = _n_events(rng, cfg.card_testing_rate, cfg.n_customers, cfg.days)
    for v in rng.choice(cust.customer_id, size=n, replace=False):
        t0 = fb.start_time(night_bias=0.6)
        n_probe = int(rng.integers(5, 16))
        probe_ts = t0 + np.cumsum(rng.integers(10, 61, size=n_probe))
        probe_m = np.full(n_probe, rng.choice(digital))
        n_cash = int(rng.integers(1, 4))
        cash_ts = int(probe_ts[-1]) + np.sort(rng.integers(30 * 60, 3 * HOUR, size=n_cash))
        fb.add(
            CARD_TESTING,
            int(v),
            np.concatenate([probe_ts, cash_ts]),
            np.concatenate([probe_m, rng.choice(big_online, size=n_cash)]),
            np.concatenate([rng.uniform(0.5, 2.0, n_probe), rng.uniform(150.0, 900.0, n_cash)]),
        )

    # 2) Account takeover: compras en línea 4–12 veces el gasto normal, en ~3 horas.
    n = _n_events(rng, cfg.takeover_rate, cfg.n_customers, cfg.days)
    for v in rng.choice(cust.customer_id, size=n, replace=False):
        t0 = fb.start_time(night_bias=0.3)
        k = int(rng.integers(2, 7))
        amounts = np.clip(cust.mean_amount[v] * rng.uniform(4.0, 12.0, k), 50.0, None)
        fb.add(
            ACCOUNT_TAKEOVER,
            int(v),
            t0 + np.sort(rng.integers(0, 3 * HOUR, size=k)),
            rng.choice(any_online, size=k),
            amounts,
        )

    # 3) Skimming: datáfono comprometido 14 días; tarjetas clonadas usadas en otra ciudad.
    pos_ids = ids[merch.city >= 0]
    n = max(1, _n_events(rng, cfg.skimming_rate, len(pos_ids), cfg.days))
    compromised = rng.choice(pos_ids, size=n, replace=False)
    order = np.argsort(legit.ts)
    l_ts, l_cust, l_merch = legit.ts[order], legit.customer[order], legit.merchant[order]
    cloning_cats = [_cat("electronics"), _cat("fuel"), _cat("clothing")]
    for m in compromised:
        w0 = int(rng.integers(cfg.start_ts, cfg.end_ts - 14 * DAY))
        lo, hi = np.searchsorted(l_ts, [w0, w0 + 14 * DAY])
        hit = l_merch[lo:hi] == m
        victims, first_idx = np.unique(l_cust[lo:hi][hit], return_index=True)
        first_ts = l_ts[lo:hi][hit][first_idx]
        for v, t_v in zip(victims, first_ts, strict=True):
            if rng.random() > 0.5:  # no todas las tarjetas copiadas se usan
                continue
            # 70 %: la banda usa la tarjeta en otra ciudad; 30 %: en la misma ciudad.
            home = int(cust.home_city[v])
            if rng.random() < SKIMMING_OTHER_CITY_PROB:
                others = [c for c in range(len(CITIES)) if c != home]
                p = _CITY_PROBS[others] / _CITY_PROBS[others].sum()
                city = int(rng.choice(others, p=p))
            else:
                city = home
            in_city = (merch.city == city) & np.isin(cats, cloning_cats)
            pool = ids[in_city] if in_city.any() else ids[merch.city == city]
            hour = (
                int(rng.choice(24, p=HOUR_PROBS)) if rng.random() < 0.7 else int(rng.integers(24))
            )
            day_start = (int(t_v) - cfg.start_ts) // DAY * DAY + cfg.start_ts
            t_f = day_start + int(rng.integers(1, 11)) * DAY + hour * HOUR
            k = int(rng.integers(1, 5))
            fb.add(
                SKIMMING,
                int(v),
                t_f + np.sort(rng.integers(0, 6 * HOUR, size=k)),
                rng.choice(pool, size=k),
                rng.uniform(80.0, 600.0, k),
            )

    return fb.build(), [int(m) for m in compromised]


def simulate(cfg: SimConfig) -> SimResult:
    """Genera clientes, comercios y transacciones (legítimas + fraude), ordenadas por tiempo."""
    rng = np.random.default_rng(cfg.seed)
    cust = build_customers(cfg, rng)
    merch = build_merchants(cfg, rng)
    legit = generate_legit(cfg, rng, cust, merch)
    fraud, compromised = inject_fraud(cfg, rng, cust, merch, legit)

    def cat(a: IntArray, b: IntArray) -> IntArray:
        return np.concatenate([a, b])

    all_ts = cat(legit.ts, fraud.ts)
    order = np.argsort(all_ts, kind="stable")
    tx = Transactions(
        ts=all_ts[order],
        customer=cat(legit.customer, fraud.customer)[order],
        merchant=cat(legit.merchant, fraud.merchant)[order],
        amount=np.concatenate([legit.amount, fraud.amount])[order],
        scenario=cat(legit.scenario, fraud.scenario)[order],
        reported_at=cat(legit.reported_at, fraud.reported_at)[order],
    )
    return SimResult(cfg, cust, merch, tx, compromised)


def summary(result: SimResult) -> dict[str, float | int]:
    tx = result.transactions
    out: dict[str, float | int] = {"transactions": len(tx)}
    for i, name in enumerate(SCENARIOS):
        out[name] = int((tx.scenario == i).sum())
    out["fraud_rate_pct"] = round(100 * float((tx.scenario > 0).mean()), 3)
    out["fraud_amount_usd"] = round(float(tx.amount[tx.scenario > 0].sum()), 2)
    return out
