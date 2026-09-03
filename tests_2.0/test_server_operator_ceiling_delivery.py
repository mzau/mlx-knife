"""The operator ceiling must reach the process that answers requests (C1).

``mlxk serve`` supervises, and uvicorn imports ``server_base`` a second time under its
real name. Patching the module attribute — what ``test_server_token_limits_api.py``
does — proves the precedence rule, not the delivery: a ceiling can be set on one copy
of the module while another copy serves. These tests therefore import the module in a
fresh process, the way uvicorn does, and read the ceiling back through the public
accessors.
"""

import os
import subprocess
import sys

# Both accessors, because the operator ceiling outranks either default (text and vision).
PROBE = (
    "import mlxk2.core.server_base as sb; "
    "print(sb.get_effective_max_tokens(None), sb.get_effective_max_tokens_vision(None))"
)


def _probe(env_value):
    """Import server_base in a fresh process with MLXK2_MAX_TOKENS set to env_value."""
    env = dict(os.environ)
    env.pop("MLXK2_MAX_TOKENS", None)
    if env_value is not None:
        env["MLXK2_MAX_TOKENS"] = env_value
    return subprocess.run(
        [sys.executable, "-c", PROBE], env=env, capture_output=True, text=True
    )


def _ceilings(env_value):
    result = _probe(env_value)
    assert result.returncode == 0, result.stderr
    return tuple(int(n) for n in result.stdout.split())


def test_env_ceiling_reaches_a_fresh_import():
    assert _ceilings("5") == (5, 5)


def test_without_the_env_the_defaults_stand():
    assert _ceilings(None) == (32768, 2048)


def test_a_ceiling_below_one_fails_loudly():
    """Silently ignoring it would recreate the bug in a quieter form."""
    result = _probe("0")
    assert result.returncode != 0
    assert "MLXK2_MAX_TOKENS" in result.stderr
