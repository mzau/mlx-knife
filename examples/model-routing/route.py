#!/usr/bin/env python3
"""
model-routing POC — single-node task -> model selection over mlx-knife's Contract.

This prototypes the *model-routing* dimension of broke-cluster ("which model for
this task?") as a standalone consumer of `mlxk list --json`. It deliberately does
NOT do node-routing (there is only one node), commissioning, SSH, or any learned
classification. See README.md for the broke boundary and the scope discipline.

Selection is capability-driven, not learned:
  - modality is deterministic   (--image -> 'vision', --audio -> 'audio')
  - task category maps to a capability + an optional preferred model (routing.json)
  - a pinned --model collapses routing to a passthrough

Decision rule: choose only from models that are BOTH capability-matching AND
`runtime_compatible: true` (the declared n runnable contract). A model that
*declares* a capability but is not runnable is reported, not executed.

Usage:
    ./route.py "Explain quicksort"                  # chat (default)
    ./route.py --task coding "Write a bubble sort"  # task category from routing.json
    ./route.py --image photo.jpg "What is this?"    # modality: requires 'vision'
    ./route.py --audio clip.wav  "Transcribe this"  # modality: requires 'audio'
    ./route.py --model org/Some-Model "hi"          # pinned: passthrough, no routing
    ./route.py --task coding --exec "..."           # actually run via `mlxk run`
    ./route.py --json "hi"                           # emit the full decision as JSON
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "routing.json")


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def load_roster():
    """Run `mlxk list --json` and return the list of model dicts (the Contract)."""
    try:
        out = subprocess.run(
            ["mlxk", "list", "--json"],
            capture_output=True, text=True, check=True,
        ).stdout
    except FileNotFoundError:
        sys.exit("error: `mlxk` not found on PATH — activate the env that has mlx-knife.")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: `mlxk list --json` failed:\n{exc.stderr}")

    payload = json.loads(out)
    # Contract envelope: {status, command, data:{models, count}, error}
    return payload.get("data", {}).get("models", [])


def resolve_capability(args, config):
    """Map the request to a required capability + a task label. Deterministic."""
    if args.image:
        return "vision", "vision"
    if args.audio:
        return "audio", "audio"
    if args.task:
        cap = config.get("task_capability", {}).get(args.task)
        if cap is None:
            known = ", ".join(sorted(config.get("task_capability", {})))
            sys.exit(f"error: unknown --task '{args.task}' (known: {known})")
        return cap, args.task
    return "chat", "chat"


def has_cap(model, capability):
    return capability in model.get("capabilities", [])


def select(roster, capability, task_label, config):
    """Return a decision dict. Choose from capability-matching AND runnable models."""
    matching = [m for m in roster if has_cap(m, capability)]
    runnable = [m for m in matching if m.get("runtime_compatible")]
    blocked = [m for m in matching if not m.get("runtime_compatible")]

    decision = {
        "task": task_label,
        "capability": capability,
        "declared_count": len(matching),
        "runnable_count": len(runnable),
        "model": None,
        "reason": None,
        "source": None,
        "blocked": [{"name": m["name"], "reason": m.get("reason")} for m in blocked],
    }

    if not runnable:
        decision["reason"] = (
            f"no runnable model declares '{capability}'. "
            f"{len(blocked)} declare it but are not runtime_compatible "
            f"(re-pull / convert the weights, then re-check `mlxk list --json`)."
        )
        return decision

    # Preferred default for this task category (the "remembered default" — declarative).
    preferred = config.get("defaults", {}).get(task_label)
    runnable_names = {m["name"] for m in runnable}
    if preferred and preferred in runnable_names:
        decision.update(model=preferred, source="configured-default",
                        reason=f"routing.json default for '{task_label}' is runnable.")
    elif preferred:
        chosen = runnable[0]["name"]
        decision.update(model=chosen, source="fallback-first-runnable",
                        reason=f"configured default '{preferred}' is not runnable; "
                               f"fell back to first runnable '{capability}' model.")
    else:
        chosen = runnable[0]["name"]
        decision.update(model=chosen, source="first-runnable",
                        reason=f"no routing.json default for '{task_label}'; "
                               f"picked first runnable '{capability}' model.")
    return decision


def select_pinned(roster, name, capability):
    """Pinned model collapses routing to passthrough; validate + warn, never block."""
    match = next((m for m in roster if m["name"] == name), None)
    decision = {"task": "pinned", "capability": capability, "model": name,
                "source": "pinned", "reason": "model named in request; routing skipped.",
                "blocked": []}
    if match is None:
        decision["reason"] = "WARNING: pinned model not in `mlxk list` — passing through anyway."
    elif not match.get("runtime_compatible"):
        decision["reason"] = (f"WARNING: pinned model is not runtime_compatible "
                              f"({match.get('reason')}) — passing through anyway.")
    elif not has_cap(match, capability):
        decision["reason"] = (f"WARNING: pinned model does not declare '{capability}' "
                              f"(has {match.get('capabilities')}) — passing through anyway.")
    return decision


def run_exec(model, args):
    cmd = ["mlxk", "run", model]
    if args.image:
        cmd += ["--image", args.image]
    if args.audio:
        sys.exit("error: --exec is not wired for audio (the audio CLI surface differs); "
                 "run `mlxk` manually with the printed model — see README.")
    if args.prompt:
        cmd.append(args.prompt)
    os.execvp("mlxk", cmd)  # replace process; inherit stdio


def main():
    ap = argparse.ArgumentParser(description="single-node task -> model router (POC)")
    ap.add_argument("prompt", nargs="?", default="", help="the user prompt")
    ap.add_argument("--task", help="task category (see routing.json task_capability)")
    ap.add_argument("--image", help="image path -> requires 'vision'")
    ap.add_argument("--audio", help="audio path -> requires 'audio'")
    ap.add_argument("--model", help="pinned model name -> passthrough (no routing)")
    ap.add_argument("--exec", action="store_true", help="run the chosen model via `mlxk run`")
    ap.add_argument("--json", action="store_true", help="print the full decision as JSON")
    args = ap.parse_args()

    config = load_config()
    roster = load_roster()
    capability, task_label = resolve_capability(args, config)

    if args.model:
        decision = select_pinned(roster, args.model, capability)
    else:
        decision = select(roster, capability, task_label, config)

    if args.json:
        print(json.dumps(decision, indent=2))
    else:
        # reasoning -> stderr, chosen model -> stdout (so it is pipeable)
        print(f"[route] task={decision['task']} capability={decision['capability']} "
              f"-> {decision['model']}", file=sys.stderr)
        print(f"[route] {decision['reason']}", file=sys.stderr)
        if decision.get("blocked"):
            for b in decision["blocked"]:
                print(f"[route]   declared-but-not-runnable: {b['name']} ({b['reason']})",
                      file=sys.stderr)

    if decision["model"] is None:
        sys.exit(2)  # no runnable model for the requested capability

    if args.exec:
        run_exec(decision["model"], args)
    elif not args.json:
        print(decision["model"])  # stdout: the routing result


if __name__ == "__main__":
    main()
