"""A feature gate reads its value, not just its presence (audit C8).

`MLXK2_ENABLE_PIPES=0` and `MLXK2_ENABLE_ALPHA_FEATURES=0` used to *open* the gate: the
check was plain truthiness, and a non-empty string is truthy. Setting a switch to `0` is
how an operator turns it off, so a gate that opens on it is unusable as one.
"""

import os
import subprocess
import sys

import pytest

from mlxk2.cli import _feature_gate_open

GATE = "MLXK2_ENABLE_ALPHA_FEATURES"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_explicit_yes_opens_the_gate(monkeypatch, value):
    monkeypatch.setenv(GATE, value)
    assert _feature_gate_open(GATE) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "maybe"])
def test_everything_else_keeps_it_shut(monkeypatch, value):
    monkeypatch.setenv(GATE, value)
    assert _feature_gate_open(GATE) is False


def test_unset_keeps_it_shut(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)
    assert _feature_gate_open(GATE) is False


def test_zero_blocks_the_command_end_to_end():
    """The value reaches the CLI, not only the helper."""
    env = dict(os.environ, **{GATE: "0"})
    result = subprocess.run(
        [sys.executable, "-m", "mlxk2.cli", "embed", "some-model", "text"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 1
    assert GATE in (result.stdout + result.stderr)


# --- the same rule, for the switch that is not a feature gate ----------------

@pytest.mark.parametrize("value,expected", [("1", True), ("on", True), ("0", False),
                                            ("false", False), ("", False)])
def test_the_debug_switch_reads_its_value_too(monkeypatch, value, expected):
    """`MLXK2_DEBUG=0` used to turn streaming debug output on — same bug, third switch."""
    from mlxk2.core.server.streaming import _debug_enabled

    monkeypatch.setenv("MLXK2_DEBUG", value)
    assert _debug_enabled() is expected
