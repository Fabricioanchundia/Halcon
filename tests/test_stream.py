"""Pruebas del tiempo real: paridad online/offline, Valkey (falso) y el scorer de flujo.

Redpanda no se levanta en CI; el broker se prueba en la laptop con docker compose. Aquí se
prueba todo lo demás, que es donde están los errores sutiles.
"""

import json
from pathlib import Path

import duckdb
import fakeredis
import numpy as np
import pytest

from halcon.model.features import FEATURES, build_features
from halcon.model.train import SplitConfig, train
from halcon.simulator import SimConfig, simulate, write_parquet
from halcon.stream.events import TxEvent, load_home_cities, read_events
from halcon.stream.online import OnlineFeatures
from halcon.stream.service import StreamScorer, bench, default_stream_start, warmup
from halcon.stream.store import RETENTION_S, InMemoryStore, PastTx, ValkeyStore

CFG = SimConfig(n_customers=600, n_merchants=200, days=70, seed=11)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("stream")
    write_parquet(simulate(CFG), root)
    build_features(root, root / "features.parquet")
    split = SplitConfig(warmup_days=5, valid_days=10, delay_gap_days=10, test_days=20,
                        alerts_per_day=20)  # fmt: skip
    train(root / "features.parquet", root / "models", split, device="cpu")
    return root


def _offline(root: Path) -> dict[str, np.ndarray]:
    sql = f"SELECT tx_id, {', '.join(FEATURES)} FROM read_parquet(?) ORDER BY tx_id"  # noqa: S608
    data: dict[str, np.ndarray] = (
        duckdb.connect().execute(sql, [(root / "features.parquet").as_posix()]).fetchnumpy()
    )
    return data


def _assert_parity(
    online: dict[int, dict[str, float | None]], offline: dict[str, np.ndarray]
) -> None:
    ids = np.array(sorted(online))
    pos = np.searchsorted(offline["tx_id"], ids)
    for name in FEATURES:
        a = np.ma.filled(np.ma.asarray(offline[name][pos], dtype=float), np.nan)
        b = np.array([np.nan if online[i][name] is None else online[i][name] for i in ids])
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name)


def test_event_roundtrip() -> None:
    ev = TxEvent(1, 1_700_000_000, 7, 9, "5411", "groceries", "pos", "Manta", 12.5)
    assert TxEvent.decode(ev.encode()) == ev


def test_events_never_carry_labels(pipeline: Path) -> None:
    ev = next(read_events(pipeline, limit=1))
    payload = json.loads(ev.encode())
    assert not any("fraud" in key for key in payload)


def test_online_features_match_offline_exactly(pipeline: Path) -> None:
    """La prueba más importante de la fase: sin *training-serving skew*."""
    online = OnlineFeatures(InMemoryStore(), load_home_cities(pipeline))
    rows = {ev.tx_id: online.process(ev) for ev in read_events(pipeline)}
    _assert_parity(rows, _offline(pipeline))


def test_valkey_store_after_warmup_matches_offline(pipeline: Path) -> None:
    """Precargar Valkey y seguir en vivo da lo mismo que haber procesado todo desde el inicio."""
    store = ValkeyStore(fakeredis.FakeRedis(), prefix="t")
    start = default_stream_start(pipeline, test_days=20)
    warmup(pipeline, store, start)
    online = OnlineFeatures(store, load_home_cities(pipeline))
    rows = {
        ev.tx_id: online.process(ev) for ev in read_events(pipeline, start_ts=start, limit=3000)
    }
    _assert_parity(rows, _offline(pipeline))


def test_online_feature_order_matches_model() -> None:
    online = OnlineFeatures(InMemoryStore(), {1: "Manta"})
    ev = TxEvent(1, 1_700_000_000, 1, 2, "5411", "groceries", "pos", "Manta", 10.0)
    assert tuple(online.process(ev)) == FEATURES


def test_store_forgets_after_30_days() -> None:
    store = InMemoryStore()
    store.write(1, PastTx(0, 1, 10.0, 5))
    store.write(1, PastTx(RETENTION_S + 10, 2, 20.0, 5))
    history, last = store.read(1, RETENTION_S + 10)
    assert [p.tx_id for p in history] == [2]
    assert last == RETENTION_S + 10


def test_valkey_store_roundtrip() -> None:
    store = ValkeyStore(fakeredis.FakeRedis(), prefix="t")
    store.write(3, PastTx(100, 1, 12.34, 8))
    store.write(3, PastTx(100 + RETENTION_S + 1, 2, 5.0, 9))
    history, last = store.read(3, 100 + RETENTION_S + 1)
    assert history == [PastTx(100 + RETENTION_S + 1, 2, 5.0, 9)]
    assert last == 100 + RETENTION_S + 1
    assert store.flush() == 2


def test_stream_scorer_output(pipeline: Path) -> None:
    from halcon.decision.scorer import Scorer

    handler = StreamScorer(OnlineFeatures(InMemoryStore(), load_home_cities(pipeline)),
                           Scorer.load(pipeline / "models"))  # fmt: skip
    outputs = [handler.handle(ev)[0] for ev in read_events(pipeline, limit=5000)]
    actions = {o["action"] for o in outputs}
    assert "APPROVE" in actions
    flagged = [o for o in outputs if o["action"] != "APPROVE"]
    assert all("features" in o for o in flagged)  # el analista recibe las pistas
    assert all("features" not in o for o in outputs if o["action"] == "APPROVE")
    json.dumps(outputs)  # todo debe poder viajar por Redpanda


def test_bench_without_broker(pipeline: Path) -> None:
    stats = bench(pipeline, pipeline / "models", n=2000, valkey_url=None, log=lambda _: None)
    assert stats.processed == 2000
    p99 = stats.to_dict()["proc_p99_ms"]
    assert isinstance(p99, float) and p99 < 50
