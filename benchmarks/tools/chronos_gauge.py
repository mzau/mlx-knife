#!/usr/bin/env python3
"""Server overhead gauge - `mlxk serve` against `mlx_lm.server`, measured by mlx-chronos.

For one text model, starts each server in turn on the same port, runs
`mlx-chronos run --engine mlx-lm` against it, and prints both results side by side
with their ratio. The reference is measured in the same session, so the ratio stays
comparable across thermal state, machines and changes to the instrument itself.

Setup, once - mlx-chronos lives in its own venv and is only called over its command line,
so it never enters the test environment. Build that venv with a Python >= 3.10 (a bare
`python3` finds macOS' 3.9), and do not activate it:
    python -m venv venv-chronos
    venv-chronos/bin/pip install "mlx-chronos[thermal]==0.4.1"
    venv-chronos/bin/pip install --no-deps "mlx-lm==0.31.3"  # the version of the dev env

Run with the mlx-knife development environment activated - the venv that carries mlxk and
MLX - and HF_HOME pointing at the cache that holds the model:
    source <development environment>/bin/activate
    python benchmarks/tools/chronos_gauge.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

The gauge starts both servers from its own interpreter's venv, so running it out of
venv-chronos fails on the missing mlxk. mlx-chronos itself is found at
venv-chronos/bin/mlx-chronos; a venv elsewhere goes into $MLXK_CHRONOS_BIN or --chronos.

mlx-chronos only checks that the mlx-lm package is present in its own environment and
records its version; it never imports it, so --no-deps pulls in no MLX. The gauge checks
that the two versions match and prints the exact command when they do not.

The model must be text-only and one mlxk would run. --model takes whatever mlxk takes: a
cached org/name, or a workspace model by its bare name with MLXK_WORKSPACE_HOME set. mlxk
gets that spec and resolves it itself; mlx_lm knows no workspaces, so the reference server
is handed the directory mlxk resolved it to. Both servers run with HF_HUB_OFFLINE=1.
Results stay local (default: benchmarks/reports/chronos/). Both runs carry the mlx-lm
engine label, so they must never be submitted to the mlx-chronos leaderboard.

Platform: macOS + Apple Silicon (MLX requirement)
See: TESTING-DETAILS.md -> Server Overhead Gauge (mlx-chronos)
"""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parents[2]
BIN = Path(sys.executable).parent
CHRONOS_VERSION = "0.4.1"  # results from another version are not comparable
DEFAULT_CHRONOS = os.environ.get("MLXK_CHRONOS_BIN", str(REPO / "venv-chronos" / "bin" / "mlx-chronos"))
SERVERS = ("mlx-lm", "mlxk")  # the reference runs first
LABELS = {"mlx-lm": "mlx_lm.server", "mlxk": "mlxk serve"}
TEXT_CAPABILITIES = {"text-generation", "chat"}

# (row label, summary key, which direction is better)
ROWS = (
    ("TTFT cold (s)", "ttft_cold_s", "lower"),
    ("TTFT cached (s)", "ttft_cached_s", "lower"),
    ("request tok/s", "request_tps", "higher"),
    ("decode tok/s", "decode_tps", "higher"),
    ("system RAM peak (GB)", "system_ram_peak_gb", "lower"),
)


def fail(message: str) -> None:
    sys.exit(f"chronos_gauge: {message}")


def check_model(model: str) -> Dict:
    """mlxk's own verdict: a text model it would run, cached or in the workspace.

    mlx_lm.server executes a checkpoint's `model_file` unconditionally. mlxk reports such
    a model as not runnable, so this check keeps the reference server off that path.
    """
    result = subprocess.run(
        [str(BIN / "mlxk"), "show", model, "--json"], capture_output=True, text=True
    )
    try:
        info = json.loads(result.stdout)["data"]["model"]
    except (json.JSONDecodeError, KeyError, TypeError):
        fail(f"`mlxk show {model} --json` returned no model: {result.stderr.strip()[-300:]}")
    # A cached model resolves to its org/name, a workspace model to its directory.
    resolved = info.get("name") or ""
    if resolved != model and not Path(resolved).is_dir():
        fail(f"{model!r} resolves to {resolved!r}; pass a cached org/name or a workspace model")
    if not info.get("runtime_compatible"):
        fail(f"mlxk will not run {model}: {info.get('reason')}")
    capabilities = set(info.get("capabilities") or [])
    if "text-generation" not in capabilities or capabilities - TEXT_CAPABILITIES:
        fail(f"{model} is not a text-only model (capabilities: {sorted(capabilities)})")
    return info


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def start_server(kind: str, model: str, port: int, log: Path) -> subprocess.Popen:
    """`model` is what this server understands: mlxk takes the spec, mlx_lm the resolved one."""
    if kind == "mlx-lm":
        cmd = [sys.executable, "-m", "mlx_lm.server", "--model", model]
    else:
        cmd = [str(BIN / "mlxk"), "serve", "--model", model]
    cmd += ["--host", "127.0.0.1", "--port", str(port)]
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    with log.open("w") as out:
        return subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )


def wait_ready(proc: subprocess.Popen, port: int, timeout: float, log: Path) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            fail(f"server exited with {proc.returncode} before it was ready:\n{tail(log)}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1.0)
    fail(f"server not ready after {timeout:.0f} s:\n{tail(log)}")


def stop_server(proc: subprocess.Popen, port: int) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    deadline = time.monotonic() + 30
    while port_in_use(port) and time.monotonic() < deadline:
        time.sleep(0.5)


def run_chronos(
    request_model: str, port: int, out_dir: Path, args: argparse.Namespace, log: Path
) -> Path:
    cmd = [
        str(args.chronos), "run", "--engine", "mlx-lm", "--model", request_model,
        "--profile", args.profile, "--format", "json", "--output-dir", str(out_dir),
        "--notes", "chronos_gauge local run - not for submission",
    ]
    if args.trials:
        cmd += ["--trials", str(args.trials)]
    env = {
        **os.environ,
        "MLX_CHRONOS_MLX_LM_PORT": str(port),
        "MLX_CHRONOS_DISABLE_UPDATE_CHECK": "1",
    }
    with log.open("w") as out:
        code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
    results = sorted(out_dir.glob("*.json"))
    if code != 0 or not results:
        fail(f"mlx-chronos failed (exit {code}):\n{tail(log)}")
    return results[-1]


def summarize(result: Path) -> Dict:
    data = json.loads(result.read_text())
    metrics, meta = data["metrics"], data["meta"]

    def mean(key: str) -> Optional[float]:
        return (metrics.get(key) or {}).get("mean")

    return {
        "ttft_cold_s": mean("ttft_cold"),
        "ttft_cached_s": mean("ttft_cached"),
        "request_tps": mean("request_tokens_per_second"),
        "decode_tps": mean("decode_tokens_per_second"),
        "system_ram_peak_gb": metrics.get("system_ram_peak_gb"),
        "token_count_source": metrics.get("token_count_source"),
        "thermal_worst": (meta.get("thermal_monitor") or {}).get("worst_state"),
        "result_file": str(result),
    }


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def run_quiet(*cmd: str) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else ""


def chronos_version(chronos: Path) -> str:
    output = run_quiet(str(chronos), "--version")
    return output.split()[-1] if output else "unknown"


def gpu_busy(samples: int = 5) -> Optional[float]:
    """Mean GPU `Device Utilization %` over a few seconds (ioreg, no sudo).

    A game or a video stream holding the GPU slows every token that waits on it, so a
    run on a busy machine measures the other application, not the server.
    """
    values = []
    for i in range(samples):
        out = run_quiet("ioreg", "-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator")
        match = re.search(r'"Device Utilization %"=(\d+)', out)
        if match:
            values.append(int(match.group(1)))
        if i < samples - 1:
            time.sleep(1.0)
    return sum(values) / len(values) if values else None


def require_quiet_gpu(limit: float) -> Optional[float]:
    busy = gpu_busy()
    if busy is not None and busy > limit:
        fail(f"GPU is {busy:.0f}% busy before the run (limit {limit:.0f}%); close games and "
             "video (browser tabs too) or raise --max-gpu-busy")
    return busy


def check_chronos_env(chronos: Path) -> None:
    """The mlx-lm package chronos sees must be the one the reference server runs."""
    found = chronos_version(chronos)
    if found != CHRONOS_VERSION:
        print(f"warning: mlx-chronos {found}, gauge pinned to {CHRONOS_VERSION}; "
              "results are not comparable with other sessions", flush=True)
    python = chronos.parent / "python"
    seen = run_quiet(str(python), "-c", "import importlib.metadata as m; print(m.version('mlx-lm'))")
    wanted = version("mlx-lm")
    if seen != wanted:
        fail(f"mlx-chronos sees mlx-lm {seen or 'none'}, the servers run {wanted}:\n"
             f'  {chronos.parent / "pip"} install --no-deps "mlx-lm=={wanted}"')


def environment(chronos: Path) -> Dict:
    commit = run_quiet("git", "-C", str(REPO), "rev-parse", "--short", "HEAD")
    if commit and run_quiet("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"):
        commit += "+dirty"
    # An editable install keeps the version it was installed with; mlxk reports the tree's.
    mlxk = run_quiet(str(BIN / "mlxk"), "--version")
    return {
        "mlx-knife": mlxk.split()[-1] if mlxk else "unknown",
        "mlx-lm": version("mlx-lm"),
        "mlx": version("mlx"),
        "mlx-chronos": chronos_version(chronos),
        "chip": run_quiet("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0],
        "commit": commit or "unknown",
    }


def fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def report(model: str, resolved: str, env: Dict, summaries: Dict[str, Dict]) -> str:
    ref, own = summaries["mlx-lm"], summaries["mlxk"]
    lines = [
        f"## {model} - {datetime.now():%Y-%m-%d %H:%M}",
        " · ".join(f"{k} {v}" for k, v in env.items()),
        "",
    ]
    # Which directory a workspace name stood for is not recoverable from the name later.
    if resolved != model:
        lines += [f"resolved to `{resolved}`", ""]
    lines += [
        f"| metric | {LABELS['mlx-lm']} | {LABELS['mlxk']} | mlxk / mlx-lm | better |",
        "|---|---:|---:|---:|---|",
    ]
    for label, key, better in ROWS:
        a, b = ref.get(key), own.get(key)
        ratio = fmt(b / a) if a and b is not None else "n/a"
        lines.append(f"| {label} | {fmt(a)} | {fmt(b)} | {ratio} | {better} |")
    for label, key in (
        ("GPU busy before run (%)", "gpu_busy_before_pct"),
        ("token counts", "token_count_source"),
        ("thermal (worst)", "thermal_worst"),
    ):
        lines.append(f"| {label} | {ref.get(key)} | {own.get(key)} | | |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Cached text model, exact org/name")
    parser.add_argument("--port", type=int, default=8080, help="Port for both servers (default: 8080)")
    parser.add_argument("--profile", choices=("baseline", "sustained"), default="baseline")
    parser.add_argument("--trials", type=int, help="Trials per phase (default: chronos profile)")
    parser.add_argument("--pause", type=int, default=60, help="Seconds between the two runs (default: 60)")
    parser.add_argument("--timeout", type=int, default=600, help="Server start timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=REPO / "benchmarks" / "reports" / "chronos")
    parser.add_argument(
        "--max-gpu-busy", type=float, default=10.0,
        help="Refuse to start a run while the GPU is busier than this, in %% (default: 10)",
    )
    parser.add_argument(
        "--chronos", type=Path, default=Path(DEFAULT_CHRONOS),
        help="mlx-chronos executable in its own venv (default: $MLXK_CHRONOS_BIN or venv-chronos/)",
    )
    args = parser.parse_args()

    if not args.chronos.exists():
        fail(f"no mlx-chronos at {args.chronos}; see the usage in this file's docstring")
    check_chronos_env(args.chronos)
    if port_in_use(args.port):
        fail(f"port {args.port} is already in use")
    info = check_model(args.model)
    # mlx_lm resolves a model name against the Hugging Face cache only, so a workspace model
    # reaches the reference server as the directory mlxk resolved it to. mlxk keeps the spec:
    # it resolves workspace-first itself, and the spec is also the name chronos then requests.
    reference_model = info["name"]
    models = {"mlx-lm": reference_model, "mlxk": args.model}

    # A workspace model passed as a path would drag the whole path into the directory name.
    slug = Path(args.model).name if os.path.isabs(args.model) else args.model.replace("/", "--")
    session = args.output_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    session.mkdir(parents=True)
    env = environment(args.chronos)
    summaries: Dict[str, Dict] = {}

    for i, kind in enumerate(SERVERS):
        if i:
            print(f"pausing {args.pause} s before the next server ...", flush=True)
            time.sleep(args.pause)
        busy = require_quiet_gpu(args.max_gpu_busy)
        print(f"{LABELS[kind]}: starting with {models[kind]} on port {args.port} ...", flush=True)
        server_log = session / f"{kind}.server.log"
        proc = start_server(kind, models[kind], args.port, server_log)
        try:
            wait_ready(proc, args.port, args.timeout, server_log)
            # mlx_lm.server keys its preloaded model as "default_model"; any other name
            # makes it load the model a second time. mlxk keys it by the --model string.
            request_model = "default_model" if kind == "mlx-lm" else args.model
            print(f"{LABELS[kind]}: running mlx-chronos ({args.profile}) ...", flush=True)
            out_dir = session / kind
            out_dir.mkdir()
            result = run_chronos(request_model, args.port, out_dir, args, session / f"{kind}.chronos.log")
            summaries[kind] = summarize(result)
            summaries[kind]["gpu_busy_before_pct"] = busy
        finally:
            stop_server(proc, args.port)

    text = report(args.model, reference_model, env, summaries)
    (session / "summary.json").write_text(
        json.dumps(
            {"model": args.model, "resolved": reference_model,
             "environment": env, "servers": summaries},
            indent=2,
        ) + "\n"
    )
    (session / "summary.md").write_text(text + "\n")
    print()
    print(text)
    print(f"\nresults: {session}")


if __name__ == "__main__":
    main()
