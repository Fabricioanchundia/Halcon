"""Métricas del scorer para Prometheus.

Nombres y unidades siguen las convenciones de Prometheus (``_total``, ``_seconds``). Los
histogramas tienen un borde exacto en 50 ms porque ese es el objetivo del SLO: así la
proporción de decisiones "a tiempo" se calcula sin interpolar.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
SLO_SECONDS = 0.05


class ScorerMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.decisions = Counter(
            "halcon_decisions_total",
            "Decisiones tomadas por el scorer, por acción.",
            ["action"],
            registry=self.registry,
        )
        self.processing = Histogram(
            "halcon_processing_seconds",
            "Tiempo de proceso de un pago dentro del scorer (features + modelo + reglas).",
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.end_to_end = Histogram(
            "halcon_end_to_end_seconds",
            "Desde que el productor envía el pago hasta que sale la decisión.",
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.lag = Gauge(
            "halcon_consumer_lag_messages",
            "Mensajes pendientes en Redpanda por partición asignada a este scorer.",
            ["partition"],
            registry=self.registry,
        )
        self.errors = Counter(
            "halcon_errors_total",
            "Errores al procesar o leer mensajes.",
            ["kind"],
            registry=self.registry,
        )

    def serve(self, port: int) -> None:
        start_http_server(port, registry=self.registry)

    def observe(self, action: str, proc_seconds: float, e2e_seconds: float | None) -> None:
        self.decisions.labels(action=action).inc()
        self.processing.observe(proc_seconds)
        if e2e_seconds is not None:
            self.end_to_end.observe(e2e_seconds)
