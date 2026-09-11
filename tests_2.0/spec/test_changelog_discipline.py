"""CHANGELOG discipline: the top section tells the truth, released sections stay frozen.

Two rules that were kept by hand until one of them nearly slipped — the 2.0.8-beta.1 release
commit was first built with `[Unreleased]` still standing as the version heading, and only
caught before it was published. In a published release `[Unreleased]` is simply wrong: the
cut renames it, and the first commit after the cut creates a fresh one. That is recurring mechanics, so it belongs in
the tree as a test rather than in a planning document.

A third rule concerns what the top section says rather than what it is called: it names no
release after the one it belongs to. Public text states what exists — `deferred`, `not built`,
a tracking issue — never the release something might land in, the rule `docs/ADR/README.md`
sets for ADRs. The top section is where new prose enters, so that is where it is held.

This sits beside the JSON API version triple rather than inside it: the triple holds the
*spec* version (0.2.x), this holds the *package* version (2.0.x). Asserting them equal would
be wrong, so they stay separate assertions about separate numbers.
"""

import re
import subprocess
from pathlib import Path

import pytest

from mlxk2 import __version__

_REPO = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO / "CHANGELOG.md"

_HEADING = re.compile(r"^## \[(?P<label>[^\]]+)\].*$", re.MULTILINE)

# A release as prose names one: `2.1`, `2.1+`, `2.0.9`, `2.0.9-beta.1`, `2.0.9b1`. Numbers of the
# same shape are left alone by context — `Qwen2.5`, `numpy>=2.1`, `LGPL-2.1`, `2.5-VL`, `2.1 GB` —
# and three-part versions with a non-zero minor (`torch 2.13.0`) never match.
_RELEASE_TOKEN = re.compile(
    r"(?<![\w./<>=~!])(?<![A-Z]-)(?P<major>\d+)\.(?:0\.\d+|[1-9]\d*)"
    r"(?:-(?:alpha|beta|rc)\.\d+|(?:a|b|rc)\d+)?\+?"
    r"(?!\.?\d)(?![A-Za-z])(?!-[A-Za-z])(?!\s?[KMGT]i?B\b)"
)


def _canonical(version: str) -> str:
    """One spelling for `2.0.8b1` (PEP 440, what the package carries) and `2.0.8-beta.1`
    (what the CHANGELOG and the git tag carry)."""
    v = version.strip().lower()
    for long, short in (("-beta.", "b"), ("-alpha.", "a"), ("-rc.", "rc")):
        v = v.replace(long, short)
    return v.replace("-", "").replace("_", "")


def _release(version: str) -> tuple:
    """The release a version belongs to: `2.0.8b2` → (2, 0, 8), `2.1+` → (2, 1, 0)."""
    parts = re.match(r"\d+(?:\.\d+)*", version).group().split(".")
    return tuple(int(p) for p in (parts + ["0", "0"])[:3])


def _sections(text: str) -> dict:
    """Every `## [label]` section, label → the section text including its heading."""
    matches = list(_HEADING.finditer(text))
    ends = [m.start() for m in matches[1:]] + [len(text)]
    return {
        m.group("label"): text[m.start():end]
        for m, end in zip(matches, ends)
    }


def _git(*args) -> str:
    """Git output, or a skip — a release tarball is not a repository."""
    try:
        done = subprocess.run(
            ("git", "-C", str(_REPO), *args),
            capture_output=True, text=True, check=False,
        )
    except OSError:
        pytest.skip("git not available")
    if done.returncode != 0:
        pytest.skip(f"git {' '.join(args)} failed: {done.stderr.strip()[:120]}")
    return done.stdout


@pytest.mark.spec
def test_top_section_is_unreleased_or_this_version():
    """The fifth place a version has to agree: whatever stands at the top of the CHANGELOG.

    `[Unreleased]` while developing, the package version once the cut renamed it. Anything
    else means a fold was forgotten or a heading drifted from `__version__`.
    """
    headings = _HEADING.findall(_CHANGELOG.read_text(encoding="utf-8"))
    assert headings, "CHANGELOG has no '## [...]' section at all"

    top = headings[0]
    assert top == "Unreleased" or _canonical(top) == _canonical(__version__), (
        f"top CHANGELOG section is [{top}], which is neither [Unreleased] nor "
        f"__version__ ({__version__}). A release commit must carry its own version as the "
        f"top heading; the first commit after a cut must re-create [Unreleased]."
    )


@pytest.mark.spec
def test_top_section_names_no_later_release():
    """Nothing in the section being written points past the release it belongs to.

    Measured against `__version__` by release, so the cycle's own number (`2.0.8` while its
    betas are cut) names the cycle and stays allowed; `2.0.9`, `2.1` or `2.1+` do not.
    """
    sections = _sections(_CHANGELOG.read_text(encoding="utf-8"))
    assert sections, "CHANGELOG has no '## [...]' section at all"

    label, text = next(iter(sections.items()))
    current = _release(__version__)
    later = sorted({
        m.group(0) for m in _RELEASE_TOKEN.finditer(text)
        if int(m.group("major")) == current[0] and _release(m.group(0)) > current
    })
    assert not later, (
        f"[{label}] names a release after {__version__}: {', '.join(later)}. Write the state — "
        f"deferred, not built, a tracking issue — not the release it might land in."
    )


@pytest.mark.spec
def test_newest_released_section_is_byte_identical_to_its_tag():
    """What shipped last is frozen — that is the section an absent `[Unreleased]` invites.

    Scoped to the newest released section, not all of them, because history carries
    deliberate corrections: measured here, `2.0.3` had a stale `review_report.md` reference
    removed and `2.0.4-beta.1` had `- WIP` replaced by its real date, both long after the
    tag. Freezing all of history would assert those were mistakes. The hazard this guards is
    narrower and concrete: writing a new bullet under the version just cut instead of
    creating `[Unreleased]` above it.
    """
    sections = _sections(_CHANGELOG.read_text(encoding="utf-8"))
    released = [label for label in sections if label != "Unreleased"]
    assert released, "CHANGELOG has no released section"

    label = released[0]
    tags = {_canonical(t): t for t in _git("tag", "--list").split() if not t.startswith("json-")}
    tag = tags.get(_canonical(label))
    if tag is None:
        pytest.skip(f"newest released section [{label}] has no tag yet")

    at_tag = _sections(_git("show", f"{tag}:CHANGELOG.md")).get(label)
    if at_tag is None:
        # The tag points somewhere that has no such section — a mislabeled lightweight tag
        # can do that. It cannot be the authority for what it lacks.
        pytest.skip(f"tag {tag} carries no [{label}] section")

    assert at_tag == sections[label], (
        f"the [{label}] section differs from tag {tag}. Released text is frozen — put the "
        f"change under `## [Unreleased]` instead, creating it if the cut removed it."
    )
