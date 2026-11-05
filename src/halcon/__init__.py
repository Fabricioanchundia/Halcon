"""Halcón: detección de fraude en pagos en tiempo real."""

__version__ = "0.1.0"


def main() -> None:
    """Punto de entrada de la CLI (`halcon`)."""
    from halcon.cli import run

    raise SystemExit(run())
