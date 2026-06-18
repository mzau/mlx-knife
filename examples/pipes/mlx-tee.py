#!/usr/bin/env python3
"""
mlx-tee: Broadcast stdin to multiple commands in parallel.

Usage:
    echo "prompt" | mlx-tee.py "mlx-run model1 -" "mlx-run model2 -"

    # Remote execution (SSH placeholder for broke-cluster)
    echo "prompt" | mlx-tee.py "@mac_mini:mlx-run model1 -" "@mac_studio:mlx-run model2 -"

Requires: MLXK2_ENABLE_PIPES=1 for mlx-run stdin support.

This is a reference implementation showing the concept of parallel model
execution. For production distributed workflows, see broke-cluster.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_command(cmd: str, stdin_data: str) -> dict:
    """Execute a command with stdin, return result dict."""

    # Remote execution: @node:command → ssh node "command"
    if cmd.startswith("@"):
        node, remote_cmd = cmd[1:].split(":", 1)
        actual_cmd = f'ssh {node} "{remote_cmd}"'
        is_remote = True
    else:
        actual_cmd = cmd
        is_remote = False

    try:
        result = subprocess.run(
            actual_cmd,
            shell=True,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        return {
            "command": cmd,
            "remote": is_remote,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "remote": is_remote,
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "error": "timeout",
        }
    except Exception as e:
        return {
            "command": cmd,
            "remote": is_remote,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": str(type(e).__name__),
        }


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    commands = sys.argv[1:]
    stdin_data = sys.stdin.read()

    # Execute all commands in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = {
            executor.submit(run_command, cmd, stdin_data): cmd
            for cmd in commands
        }
        for future in as_completed(futures):
            results.append(future.result())

    # Output results (preserve command order)
    results_by_cmd = {r["command"]: r for r in results}
    for cmd in commands:
        r = results_by_cmd[cmd]
        if r["error"]:
            print(f"[ERROR: {cmd}] {r['error']}: {r['stderr']}", file=sys.stderr)
        else:
            print(r["stdout"], end="")
            if r["stderr"]:
                print(r["stderr"], file=sys.stderr, end="")


if __name__ == "__main__":
    main()
