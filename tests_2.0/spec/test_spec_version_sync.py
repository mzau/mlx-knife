"""Ensures the code’s spec version matches docs/json-api-specification.md.

This enforces discipline: Spec, code, and tests must evolve together.
"""

from pathlib import Path
import re
import pytest

from mlxk2.spec import JSON_API_SPEC_VERSION


@pytest.mark.spec
def test_spec_version_matches_docs():
    docs_path = Path("docs/json-api-specification.md")
    assert docs_path.exists(), "Spec document missing"
    content = docs_path.read_text(encoding="utf-8")

    # Extract the version from the first lines like: **Specification Version:** 0.1.2
    m = re.search(r"\*\*Specification Version:\*\*\s*([0-9]+\.[0-9]+\.[0-9]+)", content)
    assert m, "Could not parse spec version from docs"
    docs_version = m.group(1)

    assert (
        docs_version == JSON_API_SPEC_VERSION
    ), f"Spec version mismatch: docs={docs_version} code={JSON_API_SPEC_VERSION}"
