#!/usr/bin/env python3
"""Release guard: does SERVER-HANDBOOK.md still describe the server that is in the tree?

The handbook is the sole contract a client implements from, so a claim it makes that the code
does not honour is a defect even when nothing changed — which is exactly how HTTP 422 stayed
undocumented for a whole release line. This checks the parts that are mechanically decidable:

  routes        every route the servers declare appears in the handbook
  status codes  every status the servers raise appears in the status-code section
  error types   the error table lists exactly the types a server response can carry, each one
                in the ErrorType enum
  status pairs  a status written next to an error type is the table's status for that type
  pins          the requirements block matches pyproject.toml
  pointers      no source paths or code constants leak into the contract
  language      no roadmap wording (the handbook states what is, not what may come)
  anchors       every internal link resolves
  env vars      every variable the server reads is in the environment block and vice versa;
                the binding variables show the flag defaults
  limits        the Limits table agrees with the constants behind it
  symptoms      quoted error messages in Troubleshooting are ones the code produces
  ports         example URLs use the default port, or one the handbook starts with --port
  json blocks   every ```json block parses (a // comment is not JSON)

Run from anywhere; exits 1 with every failing rule and its offending items named. An optional
path checks another copy of the handbook — the previous commit's, to show a rule red.
Deliberately not a pytest: this is a release step, run before the final commit.
"""
from __future__ import annotations

import json
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

CLI = ROOT / "mlxk2/cli.py"
VISION_ADAPTER = ROOT / "mlxk2/tools/vision_adapter.py"
TOKEN_LIMITS = ROOT / "mlxk2/core/runner/token_limits.py"

# What runs inside the server process besides the HTTP surface: the runners the handlers
# call and the logging setup. A variable read here is one the server's operator can set.
RUNTIME_SOURCES = SERVER_SOURCES + [
    ROOT / "mlxk2/core/vision_runner.py",
    ROOT / "mlxk2/core/audio_runner.py",
    ROOT / "mlxk2/logging.py",
]
ENV_READ = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["'](MLXK2?_[A-Z0-9_]+)["']"""
    r"""|os\.environ\[\s*["'](MLXK2?_[A-Z0-9_]+)["']\s*\](?!\s*=[^=])"""
)
# The supervisor sets these from the flag on every start, so the value the block shows has
# to be the flag's default — an exported value never reaches the server.
FLAG_MIRRORS = {"MLXK2_HOST": "--host", "MLXK2_PORT": "--port", "MLXK2_LOG_LEVEL": "--log-level"}

MIB = 1024 * 1024
# Limits-table rows and the constant each must agree with; the divisor renders bytes as MB.
LIMITS = {
    "Image size": (VISION_ADAPTER, "MAX_IMAGE_SIZE_BYTES", MIB),
    "Total image size": (VISION_ADAPTER, "MAX_TOTAL_IMAGE_SIZE_BYTES", MIB),
    "Images per chunk": (VISION_ADAPTER, "MAX_SAFE_CHUNK_SIZE", 1),
    "Audio size": (VISION_ADAPTER, "MAX_AUDIO_SIZE_BYTES", MIB),
    "Vision max_tokens": (TOKEN_LIMITS, "DEFAULT_MAX_TOKENS_VISION", 1),
    "Text max_tokens": (TOKEN_LIMITS, "DEFAULT_MAX_TOKENS", 1),
}
# Rows the code bounds with no number at all, so the row may not carry one either.
UNBOUNDED_ROWS = ("Images per request",)

# What a quoted error message may leave out: the parts the code fills in at runtime.
PLACEHOLDERS = re.compile(r"xxx|[XYN] ?GB|\.\.\.|<[^>]+>|\{[^}]+\}")

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
    "future:",
    "may add",
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


def listed_statuses(hb: str) -> set[int] | None:
    section = re.search(r"^## HTTP Status Codes$(.*?)^---", hb, re.M | re.S)
    return {int(c) for c in re.findall(r"\*\*(\d{3})\b", section.group(1))} if section else None


def rule_status_codes(hb: str) -> str:
    raised = set()
    for src in SERVER_SOURCES:
        text = src.read_text()
        raised |= {int(c) for c in re.findall(r"status_code\s*=\s*(\d{3})", text)}
        raised |= {int(c) for c in re.findall(r"HTTPException\(\s*(\d{3})", text)}
    raised = {c for c in raised if c >= 400}

    listed = listed_statuses(hb)
    if listed is None:
        fail("status codes", "no '## HTTP Status Codes' section found")
        return "section missing"

    missing = sorted(raised - listed)
    if missing:
        fail("status codes", f"raised by the server but not listed: {missing}")
    return f"{len(raised)} raised, {len(listed)} listed"


# A row of the error table: `type` | status | meaning.
ERROR_ROW = re.compile(r"^\|\s*`([a-z_]+)`\s*\|\s*(\d{3})\s*\|", re.M)


def enum_values() -> dict[str, str]:
    """ErrorType member name -> the type string a response carries."""
    block = re.search(r"class ErrorType\b.*?(?=\nclass |\n@|\Z)", ERRORS.read_text(), re.S)
    return dict(re.findall(r'^\s*([A-Z_]+)\s*=\s*["\']([a-z_]+)["\']', block.group(0), re.M)) if block else {}


def server_error_types(hb: str) -> set[str]:
    """The types a server response can carry. A status-to-type mapping entry counts only for a
    status the handbook lists — the status rule holds every raised status to that list — so a
    mapped status no server path raises does not bring its type along. Every other type a server
    module names, directly or through an `errors.py` constructor, counts as it stands."""
    values = enum_values()
    listed = listed_statuses(hb) or set()
    constructors = {
        func.group(1): values[name]
        for func in re.finditer(r"^def (\w+)\(.*?(?=^def |\Z)", ERRORS.read_text(), re.M | re.S)
        for name in re.findall(r"type=ErrorType\.([A-Z_]+)", func.group(0))
        if name in values
    }
    mapping = re.compile(r"(\d{3}):\s*ErrorType\.([A-Z_]+)")
    carried = set()
    for src in SERVER_SOURCES:
        text = src.read_text()
        carried |= {values[n] for s, n in mapping.findall(text) if int(s) in listed and n in values}
        named = mapping.sub("", text)
        carried |= {values[n] for n in re.findall(r"ErrorType\.([A-Z_]+)", named) if n in values}
        carried |= {constructors[c] for c in re.findall(r"\b(\w+)\(", named) if c in constructors}
    return carried


def rule_error_types(hb: str) -> str:
    values = set(enum_values().values())
    if not values:
        fail("error types", "ErrorType enum not found")
        return "enum missing"
    documented = {m.group(1) for m in ERROR_ROW.finditer(hb)}
    carried = server_error_types(hb)

    if phantom := sorted(documented - values):
        fail("error types", f"documented but not in the enum: {phantom}")
    if undocumented := sorted(carried - documented):
        fail("error types", f"a server response can carry these, the table does not list them: {undocumented}")
    if foreign := sorted((documented & values) - carried):
        fail("error types", f"in the table, but no server response carries them: {foreign}")
    return f"{len(carried)} a server response can carry, {len(documented)} in table"


def rule_status_pairs(hb: str) -> str:
    # A status written right before an error type, as in "**422** `capability_not_supported`" or
    # "HTTP 500 `internal_error`". A pair split across lines is not seen.
    table = {m.group(1): m.group(2) for m in ERROR_ROW.finditer(hb)}
    values = set(enum_values().values())
    pair = re.compile(r"(?:\*\*|HTTP\s+)(\d{3})(?:\*\*)?\s+`([a-z_]+)`")
    checked, wrong = 0, []
    for lineno, line in enumerate(hb.splitlines(), 1):
        if ERROR_ROW.match(line):
            continue
        for status, etype in pair.findall(line):
            if etype in table:
                checked += 1
                if status != table[etype]:
                    wrong.append(f"line {lineno}: {status} `{etype}`, the table says {table[etype]}")
            elif etype in values:
                checked += 1
                wrong.append(f"line {lineno}: {status} `{etype}`, a type no server response carries")
    if wrong:
        fail("status pairs", "; ".join(wrong))
    return f"{checked - len(wrong)} of {checked} pairs agree with the table"


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


def serve_flag_default(flag: str) -> str:
    """The argparse default `mlxk serve` gives a flag, as the string an operator would export."""
    match = re.search(
        rf"""serve_parser\.add_argument\(\s*["']{flag}["'][^)]*?default=("[^"]*"|'[^']*'|[\w.]+)""",
        CLI.read_text(),
        re.S,
    )
    return match.group(1).strip("\"'") if match else ""


def rule_env_vars(hb: str) -> str:
    block = re.search(r"### Environment Variables\s*```bash\n(.*?)```", hb, re.S)
    if not block:
        fail("env vars", "no ```bash block under '### Environment Variables'")
        return "block missing"
    documented = dict(re.findall(r"^(MLXK2?_[A-Z0-9_]+)=(\S*)", block.group(1), re.M))

    read = set()
    for src in RUNTIME_SOURCES:
        read |= {a or b for a, b in ENV_READ.findall(src.read_text())}
    if missing := sorted(read - documented.keys()):
        fail("env vars", f"read by the server but absent from the environment block: {missing}")

    code = "\n".join(p.read_text() for p in ROOT.glob("mlxk2/**/*.py"))
    if phantom := sorted(n for n in documented if f'"{n}"' not in code and f"'{n}'" not in code):
        fail("env vars", f"in the environment block but named nowhere in the code: {phantom}")

    for name, flag in FLAG_MIRRORS.items():
        default = serve_flag_default(flag)
        if name in documented and documented[name] != default:
            fail("env vars", f"{name} shows {documented[name]!r}; `serve {flag}` defaults to {default!r}")
    return f"{len(read)} read by the server, {len(documented)} in the block"


def _constant(path: Path, name: str) -> int:
    match = re.search(rf"^{name}\s*=\s*([0-9][0-9 *]*)", path.read_text(), re.M)
    if not match:
        raise LookupError(f"{name} is not defined in {path.relative_to(ROOT)}")
    return eval(match.group(1), {"__builtins__": {}})  # digits and '*' only, by the regex


def rule_limits(hb: str) -> str:
    table = re.search(r"^\| Resource \| Limit \| Reason \|\n\|[-| ]+\|\n((?:\|.*\|\n)+)", hb, re.M)
    if not table:
        fail("limits", "no table with a 'Resource | Limit | Reason' header")
        return "table missing"
    rows = {}
    for line in table.group(1).splitlines():
        cells = [c.strip().strip("*").strip() for c in line.strip("|").split("|")]
        rows[cells[0]] = cells[1]

    def limit_of(label):
        return next((v for k, v in rows.items() if k.startswith(label)), None)

    for label, (path, name, divisor) in LIMITS.items():
        row = limit_of(label)
        if row is None:
            fail("limits", f"no row starting with {label!r}")
            continue
        number = re.search(r"\d[\d,]*", row)
        stated = int(number.group(0).replace(",", "")) if number else None
        expected = _constant(path, name) // divisor
        if stated != expected:
            fail("limits", f"{label}: the handbook says {stated}, the code says {expected}")
    for label in UNBOUNDED_ROWS:
        row = limit_of(label)
        if row is None:
            fail("limits", f"no row starting with {label!r}")
        elif re.search(r"\d", row):
            fail("limits", f"{label}: the code sets no number, the handbook shows {row!r}")
    return f"{len(LIMITS) + len(UNBOUNDED_ROWS)} rows checked against the code"


def _runtime_text() -> str:
    """The source as a message reads once formatted: implicit string concatenations joined,
    f-string fields removed — the parts only the runtime knows."""
    text = "\n".join(p.read_text() for p in sorted(ROOT.glob("mlxk2/**/*.py")))
    text = re.sub(r'"\s*\n\s*f?"', "", text)
    return re.sub(r"\{[^{}]*\}", "", text)


def rule_symptoms(hb: str) -> str:
    source = _runtime_text()
    quoted, wrong = 0, []
    for lineno, line in enumerate(hb.splitlines(), 1):
        if not line.startswith("**Symptom:**"):
            continue
        for backticked, double_quoted in re.findall(r"`([^`]+)`|\"([^\"]+)\"", line):
            text = backticked or double_quoted
            if len(text.split()) < 3:
                continue  # an endpoint or a flag, not a message
            quoted += 1
            if PLACEHOLDERS.sub("", text) not in source:
                wrong.append(f"line {lineno}: {text!r}")
    if wrong:
        fail("symptoms", "quoted messages the code does not produce — " + "; ".join(wrong))
    return f"{quoted - len(wrong)} of {quoted} quoted messages found in the code"


def rule_ports(hb: str) -> str:
    default = serve_flag_default("--port")
    declared = {default} | set(re.findall(r"--port (\d+)", hb))
    stray = []
    for lineno, line in enumerate(hb.splitlines(), 1):
        for port in re.findall(r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0):(\d+)", line):
            if port not in declared:
                stray.append(f"line {lineno}: {port}")
    if stray:
        fail("ports", f"URLs on a port no --port in the handbook starts (default {default}): " + "; ".join(stray))
    return f"default {default}, {len(declared) - 1} more started with --port"


def _parses(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except ValueError:
        return False


def rule_json_blocks(hb: str) -> str:
    bad = []
    for match in re.finditer(r"```json\n(.*?)```", hb, re.S):
        body = re.sub(r"^data: ", "", match.group(1), flags=re.M)  # an SSE event is JSON after its prefix
        body = body.replace("{...}", "{}").replace("[...]", "[]")
        body = re.sub(r'(?<!")\.\.\.(?!")', "null", body)  # a bare ... stands for a value
        lines = [line for line in body.splitlines() if line.strip()]
        if not (_parses(body) or all(_parses(line) for line in lines)):
            bad.append(f"line {hb[: match.start()].count(chr(10)) + 1}")
    if bad:
        fail("json blocks", "```json blocks that do not parse (a // comment is not JSON): " + ", ".join(bad))
    total = len(re.findall(r"```json", hb))
    return f"{total - len(bad)} of {total} blocks parse"


def main() -> int:
    handbook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HANDBOOK
    if not handbook_path.exists():
        print(f"ERROR: {handbook_path} not found", file=sys.stderr)
        return 2
    handbook = handbook_path.read_text()

    rules = [
        ("routes", rule_routes),
        ("status codes", rule_status_codes),
        ("error types", rule_error_types),
        ("status pairs", rule_status_pairs),
        ("pins", rule_pins),
        ("pointers", rule_pointers),
        ("language", rule_language),
        ("anchors", rule_anchors),
        ("env vars", rule_env_vars),
        ("limits", rule_limits),
        ("symptoms", rule_symptoms),
        ("ports", rule_ports),
        ("json blocks", rule_json_blocks),
    ]
    print(f"SERVER-HANDBOOK contract check ({handbook_path})\n")
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
