"""Prueba de humo: el paquete se importa y la CLI responde."""

import pytest

import halcon
from halcon.cli import run


def test_version() -> None:
    assert halcon.__version__ == "0.1.0"


def test_cli_without_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert run([]) == 0
    assert "simulate" in capsys.readouterr().out
