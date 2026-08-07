#!/usr/bin/env python3
"""Release guard: does SERVER-HANDBOOK.md still describe the server that is in the tree?

The handbook is the sole contract a client implements from, so a claim it makes that the code
does not honour is a defect even when nothing changed — which is exactly how HTTP 422 stayed
undocumented for a whole release line. This checks the parts that are mechanically decidable:

  routes        every route the servers declare appears in the handbook
  status codes  every status the servers raise appears in the status-code section
  error types   the error table matches the ErrorType enum, both directions
  pins          the requirements block matches pyproject.toml
  pointers      no source paths or code constants leak into the contract
  language      no roadmap wording (the handbook states what is, not what may come)
  anchors       every internal link resolves

Run from anywhere; exits 1 on the first failing rule with the offending items named.
Deliberately not a pytest: this is a release step, run before the final commit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDBOOK = ROOT / "docs/SERVER-HANDBOOK.md"
PYPROJECT = ROOT / "pyproject.toml"
ERRORS = ROOT / "mlxk2/errors.py"

# The HTTP surface: the two server apps plus their extracted handlers.
SERVER_SOURCES = sorted(
    {*(ROOT / "mlxk2/core").glob("server*.py"), *(ROOT / "mlxk2/core/server").rglob("*.py")}
)

# Wording that promises rather than describes. `not_implemented` (the error type) is not a
# match — the underscore keeps it out of "not yet".
ROADMAP_PHRASES = [
    "under consideration",
    "future release",
    "deferred to",
    "will evolve",
    "not yet",
    "planned for",
    "coming in",
]

failures: list[str] = []


def fail(rule: str, detail: str) -> None:
    failures.append(f"{rule}: {detail}")


def rule_routes(hb: str) -> str:
    declared = set()
    for src in SERVER_SOURCES:
        for _verb, path in re.findall(
            r'@app\.(get|post|put|delete)\(\s*["\']([^"\']+)["\']', src.read_text()
        ):
            declared.add(path)
    missing = sorted(p for p in declared if p not in hb)
    if missing:
        fail("routes", f"declared but absent from the handbook: {missing}")
    return f"{len(declared)} declared, {len(declared) - len(missing)} documented"


def rule_status_codes(hb: str) -> str:
    raised = set()
    for src in SERVER_SOURCES:
        text = src.read_text()
        raised |= {int(c) for c in re.findall(r"status_code\s*=\s*(\d{3})", text)}
        raised |= {int(c) for c in re.findall(r"HTTPException\(\s*(\d{3})", text)}
    raised = {c for c in raised if c >= 400}

    section = re.search(r"^## HTTP Status Codes$(.*?)^---", hb, re.M | re.S)
    if not section:
        fail("status codes", "no '## HTTP Status Codes' section found")
        return "section missing"
    listed = {int(c) for c in re.findall(r"\*\*(\d{3})\b", section.group(1))}

    missing = sorted(raised - listed)
    if missing:
        fail("status codes", f"raised by the server but not listed: {missing}")
    return f"{len(raised)} raised, {len(listed)} listed"


def rule_error_types(hb: str) -> str:
    enum_block = re.search(r"class ErrorType\b.*?(?=\nclass |\n@|\Z)", ERRORS.read_text(), re.S)
    if not enum_block:
        fail("error types", "ErrorType enum not found")
        return "enum missing"
    values = set(re.findall(r'=\s*["\']([a-z_]+)["\']', enum_block.group(0)))
    documented = set(re.findall(r"^\|\s*`([a-z_]+)`\s*\|\s*\d{3}\s*\|", hb, re.M))

    if undocumented := sorted(values - documented):
        fail("error types", f"in the enum but not in the handbook table: {undocumented}")
    if phantom := sorted(documented - values):
        fail("error types", f"documented but not in the enum: {phantom}")
    return f"{len(values)} in enum, {len(documented)} in table"


def rule_pins(hb: str) -> str:
    # Scoped to the requirements block on purpose: the migration notes legitimately carry
    # historical pins, and comparing against the whole document reports them as drift.
    block = re.search(r"\*\*Requirements.*?\n\n", hb, re.S)
    if not block:
        fail("pins", "no requirements block found")
        return "block missing"
    packages = ("mlx-lm", "mlx-vlm", "mlx-audio", "transformers")
    declared = dict(re.findall(rf'"({"|".join(packages)})\s*==\s*([\d.]+)"', PYPROJECT.read_text()))
    stated = dict(re.findall(rf'`({"|".join(packages)})==([\d.]+)`', block.group(0)))

    drift = sorted(p for p in declared if declared[p] != stated.get(p))
    if drift:
        detail = ", ".join(f"{p}: pyproject {declared[p]} vs handbook {stated.get(p, '—')}" for p in drift)
        fail("pins", detail)
    return f"{len(declared)} pinned packages agree"


def rule_pointers(hb: str) -> str:
    # A path into the source tree is never followable for a client reading the contract.
    if paths := sorted(set(re.findall(r"\bmlxk2/[\w/]+", hb))):
        fail("pointers", f"source paths in the contract: {paths}")

    # Module-level constants, identified by actually being defined as such — this keeps
    # environment variables (which the handbook may name) out of the match.
    defined = set()
    for src in ROOT.glob("mlxk2/**/*.py"):
        defined |= set(re.findall(r"^([A-Z][A-Z0-9_]{4,})\s*(?::[^=]+)?=", src.read_text(), re.M))
    if leaked := sorted(c for c in defined if re.search(rf"\b{c}\b", hb)):
        fail("pointers", f"code constants named in the contract: {leaked}")
    return "no source paths, no code constants"


def rule_language(hb: str) -> str:
    hits = []
    for lineno, line in enumerate(hb.splitlines(), 1):
        lowered = line.lower()
        for phrase in ROADMAP_PHRASES:
            if phrase in lowered:
                hits.append(f"line {lineno}: {phrase!r}")
    if hits:
        fail("language", "roadmap wording — " + "; ".join(hits))
    return f"{len(ROADMAP_PHRASES)} phrases checked, none present"


def rule_anchors(hb: str) -> str:
    def slug(heading: str) -> str:
        cleaned = re.sub(r"[^\w\s-]", "", heading.replace("`", "").lower())
        return re.sub(r"\s", "-", cleaned.strip())

    anchors = {slug(h) for h in re.findall(r"^#{1,6}\s+(.*)$", hb, re.M)}
    links = sorted(set(re.findall(r"\]\(#([^)]+)\)", hb)))
    if broken := [link for link in links if link not in anchors]:
        fail("anchors", f"internal links that do not resolve: {broken}")
    return f"{len(links)} internal links resolve"


def main() -> int:
    if not HANDBOOK.exists():
        print(f"ERROR: {HANDBOOK} not found", file=sys.stderr)
        return 2
    handbook = HANDBOOK.read_text()

    rules = [
        ("routes", rule_routes),
        ("status codes", rule_status_codes),
        ("error types", rule_error_types),
        ("pins", rule_pins),
        ("pointers", rule_pointers),
        ("language", rule_language),
        ("anchors", rule_anchors),
    ]
    print(f"SERVER-HANDBOOK contract check ({HANDBOOK.relative_to(ROOT)})\n")
    for name, rule in rules:
        before = len(failures)
        summary = rule(handbook)
        mark = "ok  " if len(failures) == before else "FAIL"
        print(f"  [{mark}] {name:<14} {summary}")

    if failures:
        print("\nThe handbook no longer describes the server in the tree:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print("\nFix the handbook (or the code) before the release commit.", file=sys.stderr)
        return 1

    print("\nHandbook matches the tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
