"""Convenience: Show E2E test portfolio.

Usage: pytest -m show_model_portfolio -s
"""

from __future__ import annotations
import pytest

pytestmark = [pytest.mark.live, pytest.mark.live_e2e, pytest.mark.show_model_portfolio]


def test_show_portfolio(portfolio_models):
    """Display E2E test portfolio models (no actual testing).

    Shows which models would be tested by live_e2e tests.
    Uses same portfolio_models fixture as E2E tests.

    Usage:
        HF_HOME=/path/to/cache pytest -m show_model_portfolio -s
    """
    from .test_utils import should_skip_model

    print("\n" + "="*90)
    print("E2E TEST PORTFOLIO (live_e2e)")
    print("="*90)
    print(f"\nTotal models discovered: {len(portfolio_models)}\n")

    # Table header
    print(f"{'Key':<15} {'Status':<10} {'RAM (GB)':<10} {'Model ID':<50}")
    print("-"*90)

    # Count testable vs skipped
    testable = 0
    skipped = 0

    for key in sorted(portfolio_models.keys()):
        info = portfolio_models[key]
        model_id = info['id']
        ram = info.get('ram_needed_gb', 0)

        # Check if would be tested or skipped
        should_skip, skip_reason = should_skip_model(key, portfolio_models)

        if should_skip:
            status = "⏭️  SKIP"
            skipped += 1
            # Truncate model_id if too long
            display_id = model_id if len(model_id) <= 45 else model_id[:42] + "..."
            print(f"{key:<15} {status:<10} {ram:>6.1f}     {display_id}")
        else:
            status = "✅ TEST"
            testable += 1
            display_id = model_id if len(model_id) <= 45 else model_id[:42] + "..."
            print(f"{key:<15} {status:<10} {ram:>6.1f}     {display_id}")

    print("-"*90)
    print(f"\nSummary: {testable} testable, {skipped} skipped")
    print("="*90)

    # Always pass - this is just for display
    assert True
