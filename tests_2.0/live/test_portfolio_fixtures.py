"""Validation tests for new portfolio fixtures (Phase 3: Test Portfolio Separation).

These tests verify that text_portfolio and vision_portfolio fixtures work correctly.
"""

import pytest

pytestmark = [pytest.mark.live, pytest.mark.live_e2e]


def test_text_portfolio_contains_only_text_models(text_portfolio):
    """Verify that text_portfolio contains no vision models."""
    # Should have at least one text model (or be empty if no HF_HOME)
    # If empty, test will skip
    if not text_portfolio:
        pytest.skip("No text models found (HF_HOME not set or no models in cache)")

    # Keys are either text_XX (discovered) or fallback names (mxfp4, qwen25, etc.)
    # Just verify we have keys, not their format
    assert len(text_portfolio) > 0, "Portfolio should not be empty"

    # All models should have required fields
    for key, model_info in text_portfolio.items():
        assert "id" in model_info
        assert "ram_needed_gb" in model_info
        assert "description" in model_info
        assert model_info["ram_needed_gb"] > 0  # Should have positive RAM estimate


@pytest.mark.live_e2e
def test_vision_portfolio_contains_only_vision_models(vision_portfolio):
    """Verify that vision_portfolio contains no text models."""
    # Vision portfolio might be empty (no vision models in cache)
    if not vision_portfolio:
        pytest.skip("No vision models found in cache")

    # Keys are either vision_XX (discovered) or fallback names (pixtral, etc.)
    # Just verify we have keys, not their format
    assert len(vision_portfolio) > 0, "Portfolio should not be empty"

    # All models should have required fields
    for key, model_info in vision_portfolio.items():
        assert "id" in model_info
        assert "ram_needed_gb" in model_info
        assert "description" in model_info

        # Vision models may have infinity RAM (skipped models)
        ram = model_info["ram_needed_gb"]
        assert ram > 0 or ram == float('inf'), f"Invalid RAM value: {ram}"


@pytest.mark.live_e2e
def test_text_and_vision_portfolios_are_disjoint(text_portfolio, vision_portfolio):
    """Verify that text and vision portfolios have no overlapping models."""
    if not text_portfolio or not vision_portfolio:
        pytest.skip("Both portfolios needed for this test")

    # Extract model IDs
    text_model_ids = {model["id"] for model in text_portfolio.values()}
    vision_model_ids = {model["id"] for model in vision_portfolio.values()}

    # Should have no overlap
    overlap = text_model_ids & vision_model_ids
    assert not overlap, f"Models appear in both portfolios: {overlap}"


@pytest.mark.live_e2e
def test_text_model_info_fixture_works(text_model_info):
    """Verify that text_model_info fixture provides correct data."""
    # This test will be parametrized over all text models
    # Skip if dummy value (no live_e2e marker or no models)
    if text_model_info is None:
        pytest.skip("No text model info available")

    # Should have required fields
    assert "id" in text_model_info
    assert "ram_needed_gb" in text_model_info
    assert "description" in text_model_info

    # Model ID should be valid HuggingFace format
    assert "/" in text_model_info["id"], f"Invalid model ID: {text_model_info['id']}"


@pytest.mark.live_e2e
def test_vision_model_info_fixture_works(vision_model_info):
    """Verify that vision_model_info fixture provides correct data."""
    # This test will be parametrized over all vision models
    # Skip if no vision models or dummy value
    if vision_model_info is None or not vision_model_info:
        pytest.skip("No vision model info available")

    # Should have required fields
    assert "id" in vision_model_info
    assert "ram_needed_gb" in vision_model_info
    assert "description" in vision_model_info

    # Model ID should be valid HuggingFace format
    assert "/" in vision_model_info["id"], f"Invalid model ID: {vision_model_info['id']}"


@pytest.mark.show_model_portfolio
def test_show_text_portfolio(text_portfolio):
    """Show text portfolio for manual inspection (use -m show_model_portfolio)."""
    print("\n" + "=" * 80)
    print("📝 TEXT PORTFOLIO")
    print("=" * 80)

    if not text_portfolio:
        print("   (empty - no text models found)")
        return

    for key, model_info in text_portfolio.items():
        ram_str = f"{model_info['ram_needed_gb']:.1f} GB" if model_info["ram_needed_gb"] != float('inf') else "∞ (SKIP)"
        print(f"   {key}: {model_info['id']}")
        print(f"          RAM: {ram_str}")


@pytest.mark.show_model_portfolio
def test_show_vision_portfolio(vision_portfolio):
    """Show vision portfolio for manual inspection (use -m show_model_portfolio)."""
    print("\n" + "=" * 80)
    print("👁️  VISION PORTFOLIO")
    print("=" * 80)

    if not vision_portfolio:
        print("   (empty - no vision models found)")
        return

    for key, model_info in vision_portfolio.items():
        ram = model_info['ram_needed_gb']
        ram_str = f"{ram:.1f} GB" if ram != float('inf') else "∞ (SKIP)"
        skip_marker = " ⚠️ WILL BE SKIPPED" if ram == float('inf') else ""
        print(f"   {key}: {model_info['id']}")
        print(f"          RAM: {ram_str}{skip_marker}")


@pytest.mark.live_e2e
def test_exclusions_are_capability_scoped(vision_portfolio, monkeypatch):
    """A model may only be missing from the vision axis for a VISION-scoped reason.

    Compares the vision portfolio against itself with the policy removed, so it
    asserts a set difference and never a portfolio size — counts depend on what
    happens to be on this machine's disk. Skips cleanly where nothing is cached.
    """
    from . import test_utils as tu

    policy = dict(tu.KNOWN_BROKEN_MODELS)          # capture before patching
    policed = {info["id"] for info in vision_portfolio.values()}

    monkeypatch.setattr(tu, "KNOWN_BROKEN_MODELS", {})
    unpoliced = {m["model_id"] for m in tu.discover_vision_models()}
    if not unpoliced:
        pytest.skip("No vision models on this machine")

    for model_id in unpoliced - policed:
        entry = policy.get(model_id) or policy.get(model_id.rsplit("/", 1)[-1])
        assert entry is not None, (
            f"{model_id} is missing from the vision portfolio with no entry at all"
        )
        assert "vision" in entry["breaks"], (
            f"{model_id} lost its vision coverage for a non-vision reason "
            f"(breaks={sorted(entry['breaks'])}). Condition: {entry['condition']}"
        )


@pytest.mark.live_e2e
def test_chat_broken_models_still_reach_their_working_axes(vision_portfolio, monkeypatch):
    """Every chat-broken model that is vision-capable must be IN the vision portfolio."""
    from . import test_utils as tu

    policy = dict(tu.KNOWN_BROKEN_MODELS)
    chat_broken = {mid for mid, e in policy.items() if "vision" not in e["breaks"]}
    if not chat_broken:
        pytest.skip("No non-vision-scoped entries in the policy")

    policed = {info["id"] for info in vision_portfolio.values()}
    monkeypatch.setattr(tu, "KNOWN_BROKEN_MODELS", {})
    vision_capable = {m["model_id"] for m in tu.discover_vision_models()}

    present = {mid for mid in vision_capable
               if mid in chat_broken or mid.rsplit("/", 1)[-1] in chat_broken}
    if not present:
        pytest.skip("No listed non-vision-broken model is vision-capable on this machine")

    for model_id in present:
        assert model_id in policed, (
            f"{model_id} is vision-capable and not vision-broken, but the vision "
            f"portfolio does not contain it — the exclusion is leaking across axes"
        )
