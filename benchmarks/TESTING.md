# Testing with Benchmark Reports (ADR-013 Phase 0)

This document explains how to generate benchmark reports during E2E tests.

## Generating Reports

### Basic Usage

```bash
# Run E2E tests with reporting
pytest -m live_e2e tests_2.0/live/ \
  --report-output benchmarks/reports/$(date +%Y-%m-%d)-v2.0.3.jsonl
```

### With Full Environment

```bash
# Use specific HF cache + generate reports
HF_HOME=/Volumes/mz-SSD/huggingface/cache \
  pytest -m live_e2e tests_2.0/live/ -v \
  --report-output benchmarks/reports/2025-11-16-v2.0.3.jsonl
```

## Adding Report Data to Tests

Tests can add structured data to reports using `request.node.user_properties`:

```python
def test_example(model_info, request):
    # ... test logic ...

    # Add model info
    request.node.user_properties.append(("model", {
        "id": model_info["id"],
        "size_gb": model_info["ram_needed_gb"],
        "family": extract_family(model_info["id"]),
        "variant": extract_variant(model_info["id"])
    }))

    # Add performance metrics
    request.node.user_properties.append(("performance", {
        "tokens_per_sec": measure_tokens_per_sec(response),
        "ram_peak_mb": get_peak_ram_usage(),
        "duration_s": response.elapsed
    }))

    # Add stop token data (ADR-009)
    request.node.user_properties.append(("stop_tokens", {
        "configured": model_stop_tokens,
        "detected": find_stop_tokens_in_response(response),
        "workaround": get_workaround_name(model_info["id"]),
        "leaked": check_for_leaked_tokens(response)
    }))

    # Add system info (optional)
    request.node.user_properties.append(("system", {
        "platform": platform.system().lower(),
        "platform_version": get_os_version(),
        "python_version": platform.python_version(),
        "mlx_version": get_mlx_version(),
        "hardware": get_hardware_model(),
        "ram_total_gb": get_total_ram_gb()
    }))

    # Anything else goes to metadata
    request.node.user_properties.append(("custom_metric", "value"))
```

## Structured Sections

Reports have predefined structured sections that map to schema fields:

| user_properties key | Maps to report field | Description |
|---------------------|----------------------|-------------|
| `model` | `model` object | Model metadata (id, size, family, variant) |
| `performance` | `performance` object | Performance metrics (tokens/sec, RAM, duration) |
| `stop_tokens` | `stop_tokens` object | Stop token behavior (ADR-009 validation) |
| `system` | `system` object | Platform information (OS, Python, MLX, hardware) |
| _anything else_ | `metadata` object | Extensible catch-all for experiments |

## Schema Validation

```bash
# Validate reports against schema (requires jsonschema)
pip install jsonschema

# Validate all reports
for report in benchmarks/reports/*.jsonl; do
  echo "Validating $report..."
  cat "$report" | while read line; do
    echo "$line" | python3 -c "
import sys, json
from jsonschema import validate

with open('benchmarks/schemas/report-v0.1.schema.json') as f:
    schema = json.load(f)

report = json.load(sys.stdin)
validate(instance=report, schema=schema)
print('✓ Valid')
"
  done
done
```

## Example Report

```json
{
  "schema_version": "0.1.0",
  "timestamp": "2025-11-16T10:30:00Z",
  "mlx_knife_version": "2.0.3",
  "test": "tests_2.0/live/test_stop_tokens_live.py::test_stop_tokens[phi-3-mini]",
  "outcome": "passed",
  "duration": 12.3,
  "model": {
    "id": "mlx-community/phi-3-mini-4k-instruct",
    "size_gb": 2.8,
    "family": "phi-3",
    "variant": "mini-4k-instruct"
  },
  "performance": {
    "tokens_per_sec": 45.2,
    "ram_peak_mb": 3200,
    "prompt_tokens": 15,
    "completion_tokens": 42
  },
  "stop_tokens": {
    "configured": ["<|end|>", "<|endoftext|>"],
    "detected": ["<|end|>"],
    "workaround": "phi-3-dual-eos",
    "leaked": false
  }
}
```

## Analyzing Reports

See `reports/README.md` for analysis examples (jq queries, statistics, trends).

## Best Practices

1. **File Naming:** Use `YYYY-MM-DD-vX.Y.Z.jsonl` format
2. **Append Only:** Never edit existing reports (historical data)
3. **Commit Reports:** Reports are git-tracked for trend analysis
4. **Schema Version:** Always include `schema_version` for evolution tracking
5. **Optional Data:** Only add what you can measure reliably
6. **No PII:** Never include personal information in reports

## Future Enhancements (Phase 1+)

- Automatic validation during `pytest --report-output`
- Performance regression detection
- Report comparison tools (`mlxk report diff`)
- Schema migration utilities
