"""The extended-thinking gate must not regress to an allow-list of model names.

It was one: `any(m in model for m in ("opus-4", "sonnet-5", "fable"))`. `claude-opus-5` matches none
of those, so pointing `deep_model` at the new frontier model would have run it with thinking OFF —
losing the only thing that makes the deep tier worth its cost — and on the cli path with
MAX_THINKING_TOKENS=0, which Opus rejects outright. The failure is silent both ways: the weekly
strategy review still returns a StrategyNote, just a shallower one.

So these tests pin the SHAPE of the predicate (deny-list), not a list of today's model names.
"""

from app import analyst


def test_every_frontier_model_reasons_including_ones_that_do_not_exist_yet():
    for model in (
        "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7",
        "claude-sonnet-5", "claude-fable-5",
        "claude-opus-9", "some-model-nobody-has-shipped-yet",   # the regression this replaces
    ):
        assert analyst._is_thinking_model(model), f"{model} would silently run without thinking"


def test_only_the_cheap_scan_tier_is_excluded():
    # Haiku is the deliberate exclusion: measured 22s -> 1.5s with no quality gain on the daily scan.
    assert not analyst._is_thinking_model("claude-haiku-4-5")
    assert not analyst._is_thinking_model("CLAUDE-HAIKU-4-5")     # settings are free text
    assert not analyst._is_thinking_model("claude-haiku-9-9")


def test_a_missing_model_does_not_crash_the_call():
    # settings.json is user-editable; an empty deep_model must not take the analyst down.
    assert analyst._is_thinking_model("") is True
    assert analyst._is_thinking_model(None) is True
