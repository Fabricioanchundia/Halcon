"""Prueba de humo: el paquete se importa y la CLI responde."""

import pytest

import halcon


def test_version() -> None:
    assert halcon.__version__ == "0.1.0"


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    halcon.main()
    assert "Halcón" in capsys.readouterr().out
