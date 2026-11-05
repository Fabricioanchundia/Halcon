"""Pruebas de la parte operativa: métricas, docker-compose, tablero, alertas y runbook."""

import json
import re
from pathlib import Path

import yaml
from prometheus_client import generate_latest

from halcon.stream.metrics import BUCKETS, SLO_SECONDS, ScorerMetrics

ROOT = Path(__file__).resolve().parents[1]


def test_metrics_exposition_has_slo_bucket() -> None:
    m = ScorerMetrics()
    m.observe("APPROVE", 0.001, 0.01)
    m.observe("DECLINE", 0.003, 0.2)
    m.lag.labels(partition="0").set(7)
    text = generate_latest(m.registry).decode()
    assert 'halcon_decisions_total{action="APPROVE"} 1.0' in text
    assert 'halcon_end_to_end_seconds_bucket{le="0.05"} 1.0' in text  # una de dos bajo 50 ms
    assert 'halcon_consumer_lag_messages{partition="0"} 7.0' in text
    assert SLO_SECONDS in BUCKETS


def test_every_compose_service_has_a_memory_limit() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name, service in compose["services"].items():
        assert "mem_limit" in service, f"{name} sin mem_limit: la laptop tiene 6 GB para WSL"


def test_images_are_pinned() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    for name, service in compose["services"].items():
        image = service.get("image", "")
        if "build" not in service:
            assert ":" in image and not image.endswith(":latest"), f"{name} sin versión fija"


def test_dashboard_only_uses_known_metrics() -> None:
    dash = json.loads((ROOT / "ops/grafana/dashboards/halcon.json").read_text(encoding="utf-8"))
    exposed = generate_latest(ScorerMetrics().registry).decode()
    rules = (ROOT / "ops/prometheus/alerts.yml").read_text(encoding="utf-8")
    for panel in dash["panels"]:
        for target in panel["targets"]:
            for metric in re.findall(r"\b(halcon[:_][a-z0-9_:]+)", target["expr"]):
                base = re.sub(r"_(bucket|count|sum)$", "", metric)
                assert base in exposed or f"record: {metric}" in rules, metric


def test_every_alert_points_to_an_existing_runbook_section() -> None:
    alerts = yaml.safe_load((ROOT / "ops/prometheus/alerts.yml").read_text(encoding="utf-8"))
    runbook = (ROOT / "docs/runbook.md").read_text(encoding="utf-8").lower()
    for rule in alerts["groups"][0]["rules"]:
        if "alert" in rule:
            anchor = rule["annotations"]["runbook"].split("#")[1]
            assert f"## {anchor.replace('-', ' ')}" in runbook, anchor


def test_no_credentials_in_compose() -> None:
    """Nada de contraseñas escritas en el repo, ni de juguete (lección del PR #9)."""
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    risky = ("PASSWORD", "SECRET", "TOKEN", "API_KEY")
    for name, service in compose["services"].items():
        env = service.get("environment") or {}
        items = env.items() if isinstance(env, dict) else (e.split("=", 1) for e in env)
        for key, value in items:
            if any(word in str(key).upper() for word in risky):
                assert str(value).startswith("${"), f"{name}: {key} escrito en el código"
