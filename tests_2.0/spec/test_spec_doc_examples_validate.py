"""Validate JSON examples in docs/json-api-specification.md against the schema.

This ensures the Spec document examples stay in sync with the current schema.
If jsonschema is not installed locally, these tests are skipped.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import pytest


def _load_schema():
    try:
        import jsonschema  # noqa: F401
    except Exception:
        pytest.skip("jsonschema not installed; skipping schema validation tests", allow_module_level=True)

    schema_path = Path("docs/json-api-schema.json")
    assert schema_path.exists(), "Schema file docs/json-api-schema.json missing"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _iter_json_blocks(md_text: str):
    # Capture fenced code blocks marked as json
    # ```json\n ... \n```
    pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
    for m in pattern.finditer(md_text):
        block = m.group(1).strip()
        if not block:
            continue
        yield block


@pytest.mark.spec
def test_spec_document_examples_validate_against_schema():
    schema = _load_schema()
    try:
        from jsonschema import Draft7Validator
    except Exception:
        pytest.skip("jsonschema not available", allow_module_level=True)

    validator = Draft7Validator(schema)
    md_path = Path("docs/json-api-specification.md")
    assert md_path.exists(), "Spec document missing"
    text = md_path.read_text(encoding="utf-8")

    had_errors = []
    validated = 0
    skipped = 0
    for idx, block in enumerate(_iter_json_blocks(text), start=1):
        # Skip illustrative/pseudo examples (contain non-JSON constructs)
        if "/*" in block or "|" in block or "... omitted" in block:
            skipped += 1
            continue

        try:
            data = json.loads(block)
        except Exception:
            # Treat unparsable fenced blocks as illustrative and skip
            skipped += 1
            continue

        errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        validated += 1
        if errors:
            first = errors[0]
            path = "/".join(map(str, first.path)) or "<root>"
            had_errors.append(f"Example #{idx} invalid at {path}: {first.message}")

    # Ensure we validated at least one real example
    assert validated > 0, "No valid JSON examples found to validate in the spec document"

    if had_errors:
        import os
        verbose = os.environ.get("MLXK2_SPEC_VALIDATION_VERBOSE") == "1"
        if verbose:
            joined = "\n".join(had_errors)
        else:
            MAX_SHOW = 5
            shown = had_errors[:MAX_SHOW]
            joined = "\n".join(shown)
            if len(had_errors) > MAX_SHOW:
                joined += f"\n... and {len(had_errors) - MAX_SHOW} more. Set MLXK2_SPEC_VALIDATION_VERBOSE=1 to see all."

        pytest.fail(
            "Spec examples do not match the current schema.\n"
            + joined
            + "\nUpdate docs examples or docs/json-api-schema.json accordingly."
        )
