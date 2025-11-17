#!/usr/bin/env python3
"""Validate JSONL benchmark reports against schema (ADR-013 Phase 0).

Usage:
    python benchmarks/validate_reports.py benchmarks/reports/2ndtest.jsonl
    python benchmarks/validate_reports.py benchmarks/reports/*.jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import jsonschema
except ImportError:
    print("Error: jsonschema not installed. Install with: pip install jsonschema")
    sys.exit(1)


def load_schema(schema_path: Path) -> dict:
    """Load JSON schema from file."""
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_report(report: dict, schema: dict, line_num: int) -> Tuple[bool, str]:
    """Validate single report against schema.

    Returns:
        (valid, error_message) tuple
    """
    try:
        jsonschema.validate(instance=report, schema=schema)
        return True, ""
    except jsonschema.ValidationError as e:
        return False, f"Line {line_num}: {e.message}"
    except jsonschema.SchemaError as e:
        return False, f"Line {line_num}: Schema error: {e.message}"


def validate_jsonl_file(jsonl_path: Path, schema: dict) -> Tuple[int, int, List[str]]:
    """Validate JSONL file against schema.

    Returns:
        (total_reports, valid_reports, errors) tuple
    """
    total = 0
    valid = 0
    errors = []

    with open(jsonl_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                report = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON: {e}")
                continue

            is_valid, error_msg = validate_report(report, schema, line_num)
            if is_valid:
                valid += 1
            else:
                errors.append(error_msg)

    return total, valid, errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmarks/validate_reports.py <jsonl_file> [<jsonl_file> ...]")
        sys.exit(1)

    # Load schema
    schema_path = Path("benchmarks/schemas/report-v0.1.schema.json")
    if not schema_path.exists():
        print(f"Error: Schema not found at {schema_path}")
        sys.exit(1)

    schema = load_schema(schema_path)
    print(f"📋 Loaded schema: {schema_path}")
    print()

    # Validate each file
    all_valid = True
    total_reports = 0
    total_valid = 0

    for jsonl_file in sys.argv[1:]:
        jsonl_path = Path(jsonl_file)
        if not jsonl_path.exists():
            print(f"❌ File not found: {jsonl_path}")
            all_valid = False
            continue

        print(f"📊 Validating: {jsonl_path}")

        total, valid, errors = validate_jsonl_file(jsonl_path, schema)
        total_reports += total
        total_valid += valid

        if errors:
            all_valid = False
            print(f"   ❌ {valid}/{total} reports valid")
            for error in errors:
                print(f"      {error}")
        else:
            print(f"   ✅ {valid}/{total} reports valid")

        print()

    # Summary
    print("=" * 60)
    print(f"Total: {total_valid}/{total_reports} reports valid across {len(sys.argv) - 1} file(s)")

    if all_valid:
        print("✅ All reports passed schema validation!")
        sys.exit(0)
    else:
        print("❌ Some reports failed validation")
        sys.exit(1)


if __name__ == "__main__":
    main()
