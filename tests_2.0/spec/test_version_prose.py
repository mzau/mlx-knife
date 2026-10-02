"""Version numbers in prose: written in forms a pattern tells apart from everything else.

2.0.8 shipped a handbook that called itself "unreleased" and named 2.0.7 the latest stable
release, and ADRs carried plan versions past their release. Each slip was a version in a form
nothing checked. CONTRIBUTING.md "Version numbers in prose" sets the forms; this holds them.

A package version counts where the pattern can tell it apart: three-part (`2.0.9`), `2.1+` or
`2.1.x` in the package's own major, or named at any major (`MLX Knife 3.0`, `mlx-knife 2.1`,
`mlxk 2.1.0`, `Version 2.1`). A bare `2.1` is not checked: section numbers, measurements and
model names share its shape. Schema versions share theirs with the whole MLX stack (0.x.y), so
only the named form counts (`JSON API 0.3.0`, `schema v0.3.0`).
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from mlxk2 import __version__
from mlxk2.spec import JSON_API_SPEC_VERSION

_REPO = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO / "CHANGELOG.md"
_HANDBOOK = _REPO / "docs" / "SERVER-HANDBOOK.md"
_BENCH_SCHEMA = _REPO / "benchmarks" / "schemas" / "report-current.schema.json"
_BENCH_MIGRATIONS = "benchmarks/schemas/MIGRATIONS.md"

_SUFFIXES = {".md", ".py", ".toml", ".txt", ".sh", ".json", ".tape", ".yml", ".yaml", ".cfg", ".ini"}
# CHANGELOG has its own test; reports and assets are recorded output; the two test files
# quote the forms they check.
_SKIP = (
    "CHANGELOG.md",
    "benchmarks/reports/",
    "tests_2.0/assets/",
    "tests_2.0/spec/test_changelog_discipline.py",
    "tests_2.0/spec/test_version_prose.py",
)
# A name directly before the number makes it someone else's: `LibreSSL 2.8.3`,
# `Apache License, Version 2.0`, `Contributor Covenant, version 2.0`.
_FOREIGN = ("libressl", "license", "covenant")

_PRE = r"(?:-(?:alpha|beta|rc)\.\d+|(?:a|b|rc)\d+)?"
_END = r"(?!\.?\d)(?![A-Za-z])"
_NAMED = re.compile(
    r"(?i)(?<![\w-])(?:mlx[- ]knife|mlxk|version)(?![\w-])\s*(?:==|>=|<=|~=|≥|≤|>|<)?\s*v?"
    rf"(?P<num>\d+\.\d+(?:\.\d+{_PRE}|\.x)?\+?){_END}"
)
_SCHEMA = re.compile(
    r"(?i)(?<![\w-])(?P<anchor>json\s+api(?:\s+schema)?|json\s+schema|spec(?:ification)?|schema)"
    rf"(?:\s+version)?\s+v?(?P<num>\d+\.\d+(?:\.\d+)?){_END}"
)
_UNRELEASED = re.compile(r"(?i)(?<!\[)unreleased(?!\])")
_LATEST = re.compile(
    r"(?i)\b(?:latest|current|newest)(?:\s+stable)?\s+(?:release|version)(?:\s+is)?\W{0,6}?"
    r"(?P<v>\d+\.\d+\.\d+)(?!\.?\d)"
)
_ECHOES = (
    re.compile(r"--version\b.*?(?:→|->|Should show:)\s*(?:mlxk(?:2|-json)?\s+)?(?P<v>\d+\.\d+\.\d+[\w.+-]*)"),
    re.compile(r'"cli_version":\s*"(?P<v>[^"]+)"'),
)
_SPEC_ECHO = re.compile(r'"json_api_spec_version":\s*"(?P<v>[^"]+)"')


def _release(version: str) -> tuple:
    """`2.0.8b2` → (2, 0, 8), `2.1+` → (2, 1, 0), `2.1.x` → (2, 1, 0)."""
    parts = re.match(r"\d+(?:\.\d+)*", version).group().split(".")
    return tuple(int(p) for p in (parts + ["0", "0"])[:3])


_CURRENT = _release(__version__)
_BARE = re.compile(
    rf"(?<![\w./<>=~!-])(?P<num>{_CURRENT[0]}\.\d+(?:\.\d+{_PRE}|\.x|\+)){_END}"
)


def _changelog_labels() -> list:
    return re.findall(r"^## \[([^\]]+)\]", _CHANGELOG.read_text(encoding="utf-8"), re.MULTILINE)


def _newest_final() -> str:
    """The version a reader installs: the newest CHANGELOG section that is neither
    `[Unreleased]` nor a pre-release."""
    return next(label for label in _changelog_labels() if re.fullmatch(r"\d+\.\d+\.\d+", label))


def _at_release_cut() -> bool:
    """The tree is a final release: CHANGELOG cut to `__version__`, no pre-release suffix."""
    labels = _changelog_labels()
    return bool(labels) and labels[0] == __version__ and re.fullmatch(r"\d+\.\d+\.\d+", __version__)


def _tracked(suffixes=_SUFFIXES) -> list:
    try:
        done = subprocess.run(
            ("git", "-C", str(_REPO), "ls-files", "-z"), capture_output=True, text=True, check=False,
        )
    except OSError:
        pytest.skip("git not available")
    if done.returncode != 0:
        pytest.skip("not a git checkout")
    return [
        f for f in done.stdout.split("\0")
        if f and Path(f).suffix in suffixes and not f.startswith(_SKIP) and (_REPO / f).is_file()
    ]


def _lines(suffixes=_SUFFIXES):
    for f in _tracked(suffixes):
        text = (_REPO / f).read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            yield f, lineno, line


def _foreign(line: str, pos: int) -> bool:
    before = re.search(r"(\S+)\s*$", line[:pos])
    return bool(before) and any(name in before.group(1).lower() for name in _FOREIGN)


def _versions(line: str):
    """(start, number) for each package version on the line; start is where its form begins,
    the anchor for a named one."""
    found = {}
    for pattern in (_NAMED, _BARE):
        for m in pattern.finditer(line):
            if m.start("num") not in found and not _foreign(line, m.start()):
                found[m.start("num")] = (m.start(), m.group("num"))
    return found.values()


def _report(hits: list) -> str:
    return "\n  ".join(hits[:20]) + (f"\n  … {len(hits) - 20} more" if len(hits) > 20 else "")


@pytest.mark.spec
def test_no_version_after_the_current_release():
    """F1: what is not released carries no number — "From 2.0.8 → unreleased", "deferred"."""
    hits = [
        f"{f}:{n}: {num}"
        for f, n, line in _lines()
        for _, num in _versions(line)
        if _release(num) > _CURRENT
    ]
    assert not hits, (
        f"versions after {__version__} (CONTRIBUTING.md, Version numbers in prose):\n  {_report(hits)}"
    )


@pytest.mark.spec
def test_unreleased_only_before_the_cut():
    """F4: "unreleased" marks a document ahead of the release; at the cut none is left."""
    if not _at_release_cut():
        pytest.skip("CHANGELOG is not cut to a final release: 'unreleased' is allowed")
    hits = [f"{f}:{n}" for f, n, line in _lines({".md"}) if _UNRELEASED.search(line)]
    assert not hits, f"'unreleased' in release {__version__}:\n  {_report(hits)}"


@pytest.mark.spec
def test_handbook_marks_unreleased_in_both_places():
    """F4: the handbook's header and its migration section switch together."""
    text = _HANDBOOK.read_text(encoding="utf-8")
    header = re.search(r"^\*\*Version:\*\*.*$", text, re.MULTILINE)
    assert header, "SERVER-HANDBOOK has no '**Version:**' line"
    section = re.search(r"^#{2,4} From \S+ → unreleased\s*$", text, re.MULTILINE | re.IGNORECASE)
    in_header = "unreleased" in header.group(0).lower()
    assert in_header == bool(section), (
        "SERVER-HANDBOOK marks 'unreleased' in "
        + ("the header but has no 'From X → unreleased' section" if in_header
           else f"'{section.group(0).strip()}' but not in the header")
    )


@pytest.mark.spec
def test_released_is_not_written_before_a_version():
    """F5a: "released 2.0.7" — the number alone says it."""
    hits = [
        f"{f}:{n}: {num}"
        for f, n, line in _lines()
        for start, num in _versions(line)
        if re.search(r"(?i)\breleased[\s-]+$", line[:start])
    ]
    assert not hits, f"'released' before a version — drop the word:\n  {_report(hits)}"


@pytest.mark.spec
def test_latest_release_names_the_newest_final():
    """F5b: "latest/current (stable) release/version X" names what a reader installs."""
    newest = _newest_final()
    hits = [
        f"{f}:{n}: {m.group('v')}"
        for f, n, line in _lines({".md"})
        for m in _LATEST.finditer(line)
        if m.group("v") != newest
    ]
    assert not hits, f"latest/current release is {newest}, the docs say otherwise:\n  {_report(hits)}"


@pytest.mark.spec
def test_version_echoes_match():
    """F6: `--version → …` and `cli_version` show the newest final release,
    `json_api_spec_version` the spec in `mlxk2/spec.py`."""
    newest = _newest_final()
    hits = []
    for f, n, line in _lines({".md"}):
        hits += [
            f"{f}:{n}: {m.group('v')} (expected {newest})"
            for pattern in _ECHOES for m in pattern.finditer(line) if m.group("v") != newest
        ]
        hits += [
            f"{f}:{n}: {m.group('v')} (expected {JSON_API_SPEC_VERSION})"
            for m in _SPEC_ECHO.finditer(line) if m.group("v") != JSON_API_SPEC_VERSION
        ]
    assert not hits, f"version echoes out of date:\n  {_report(hits)}"


@pytest.mark.spec
def test_no_schema_version_after_the_current():
    """F7: JSON API against `mlxk2/spec.py`, the benchmark schema against its `enum`.

    A bare "schema" is the benchmark schema, except in the JSON API documents.
    """
    enum = json.loads(_BENCH_SCHEMA.read_text(encoding="utf-8"))["properties"]["schema_version"]["enum"]
    bench = max(enum, key=_release)
    hits = []
    for f, n, line in _lines():
        for m in _SCHEMA.finditer(line):
            anchor = m.group("anchor").lower()
            current = JSON_API_SPEC_VERSION if ("json" in anchor or "spec" in anchor or "json-api" in f) else bench
            if _release(m.group("num")) > _release(current):
                hits.append(f"{f}:{n}: {m.group(0)} (current {current})")
        heading = re.match(r"^#{2,4}\s+(\d+\.\d+\.\d+)\b", line) if f == _BENCH_MIGRATIONS else None
        if heading and _release(heading.group(1)) > _release(bench):
            hits.append(f"{f}:{n}: {heading.group(1)} (current {bench})")
    assert not hits, f"schema versions after the current one:\n  {_report(hits)}"
