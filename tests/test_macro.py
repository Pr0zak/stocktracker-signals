"""Macro catalyst layer — the pure logic around the model call.

The theme of these tests is the distinction the whole module exists to protect: "nothing is
happening" and "we couldn't look" must never render the same way. A macro layer that silently goes
quiet reads to every downstream consumer as a calm world, which is the most dangerous thing it could
do — it would suppress risk at exactly the moment risk is real.
"""
from __future__ import annotations

import importlib
import tempfile
import time

import pytest


@pytest.fixture()
def macro(monkeypatch):
    """A macro module rooted at a throwaway data dir (the store path is bound at import)."""
    monkeypatch.setenv("SIGNALS_DATA_DIR", tempfile.mkdtemp())
    from app import macro as m
    return importlib.reload(m)


def _cat(key, severity=50, **over):
    c = {
        "key": key, "title": key.replace("-", " ").title(), "category": "geopolitics",
        "severity": severity, "direction": "risk_off", "horizon": "weeks",
        "affected": ["crude oil"], "tickers": [], "confidence": 70, "why": "because",
    }
    c.update(over)
    return c


# --------------------------------------------------------------------------- de-duplication

def test_unseen_drops_articles_already_graded(macro):
    arts = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert [a["id"] for a in macro.unseen_articles(arts, [1, 3])] == [2]


def test_unseen_with_no_history_returns_everything(macro):
    arts = [{"id": 7}, {"id": 8}]
    assert len(macro.unseen_articles(arts, [])) == 2


def test_seen_ids_are_capped(macro):
    assert len(macro.trim_seen(list(range(5000)))) == macro._MAX_SEEN_IDS


# --------------------------------------------------------------------------- catalyst merging

def test_same_situation_stays_one_catalyst_across_runs(macro):
    """The wires re-file an ongoing war for days. It must stay ONE catalyst, not stack up."""
    first = macro.merge_catalysts([], [_cat("iran-war", 80)], now=1000.0)
    second = macro.merge_catalysts(first, [_cat("iran-war", 90)], now=2000.0)
    assert len(second) == 1
    assert second[0]["severity"] == 90        # severity updates to the latest read
    assert second[0]["first_seen"] == 1000.0  # but it is still the same situation
    assert second[0]["last_seen"] == 2000.0
    assert second[0]["seen_count"] == 2


def test_distinct_situations_stay_separate(macro):
    merged = macro.merge_catalysts([], [_cat("iran-war", 80), _cat("fed-rate-path", 40)], now=1.0)
    assert {c["key"] for c in merged} == {"iran-war", "fed-rate-path"}


def test_merge_orders_by_severity(macro):
    merged = macro.merge_catalysts([], [_cat("small", 10), _cat("huge", 95), _cat("mid", 50)], now=1.0)
    assert [c["key"] for c in merged] == ["huge", "mid", "small"]


def test_prior_catalyst_survives_a_run_that_did_not_mention_it(macro):
    """A quiet news cycle must not delete an ongoing situation — it ages out on the TTL instead."""
    prior = macro.merge_catalysts([], [_cat("iran-war", 80)], now=1000.0)
    after = macro.merge_catalysts(prior, [_cat("fed-rate-path", 30)], now=1100.0)
    assert {c["key"] for c in after} == {"iran-war", "fed-rate-path"}


def test_catalyst_without_a_key_is_dropped(macro):
    """An unkeyed catalyst can't be tracked across runs, which defeats the point of merging."""
    merged = macro.merge_catalysts([], [_cat(""), _cat("real")], now=1.0)
    assert [c["key"] for c in merged] == ["real"]


# --------------------------------------------------------------------------- ageing

def test_stale_catalysts_age_out(macro):
    now = time.time()
    blob = {"generated_at": now, "catalysts": [
        _cat("fresh", 50, last_seen=now),
        _cat("ancient", 90, last_seen=now - macro.CATALYST_TTL_SECONDS - 10),
    ]}
    keys = [c["key"] for c in macro.active_catalysts(blob, now=now)]
    assert keys == ["fresh"]


def test_read_is_stale_past_the_window(macro):
    assert macro.is_stale({"generated_at": time.time() - macro.STALE_AFTER_SECONDS - 1})
    assert not macro.is_stale({"generated_at": time.time()})


def test_absent_read_counts_as_stale(macro):
    assert macro.is_stale(None)


# --------------------------------------------------------------------------- the honesty envelope

def test_no_read_reports_unavailable_not_calm(macro):
    """THE invariant. No stored read must not be presentable as an empty (i.e. clean) catalyst list."""
    state = macro.load_state()
    assert state["available"] is False
    assert state["risk_level"] is None
    assert macro.compact(state) is None      # so nothing is injected into any prompt


def test_quiet_world_is_distinguishable_from_a_broken_pipeline(macro):
    """A genuinely calm read IS available and IS injectable — the opposite of the case above."""
    macro.save({"generated_at": time.time(), "risk_level": "low", "headline": "Quiet.", "catalysts": []})
    state = macro.load_state()
    assert state["available"] is True
    assert state["degraded"] is False
    block = macro.compact(state)
    assert block is not None and block["risk_level"] == "low" and block["catalysts"] == []


def test_failed_run_over_an_old_read_is_flagged_degraded(macro):
    macro.save({
        "generated_at": time.time() - 100, "risk_level": "high", "headline": "War.",
        "catalysts": [_cat("iran-war", 90, last_seen=time.time())],
        "last_run_failed": True, "last_error": "feed down",
    })
    state = macro.load_state()
    assert state["degraded"] is True
    assert state["last_error"] == "feed down"
    assert state["risk_level"] == "high"     # the old read still stands; it is just marked


def test_corrupt_blob_reads_as_absent_never_as_calm(macro):
    macro.LATEST.parent.mkdir(parents=True, exist_ok=True)
    macro.LATEST.write_text("{not json")
    assert macro.load() is None
    assert macro.load_state()["available"] is False
    assert macro.compact(macro.load_state()) is None


def test_stale_read_is_still_served_but_marked(macro):
    """A day-old war is still a war — serve it, flag it, and let the consumer weight it down."""
    old = time.time() - macro.STALE_AFTER_SECONDS - 60
    macro.save({"generated_at": old, "risk_level": "elevated", "headline": "s",
                "catalysts": [_cat("iran-war", 70, last_seen=time.time())]})
    state = macro.load_state()
    assert state["available"] is True
    assert state["stale"] is True
    assert macro.compact(state)["stale"] is True


def test_compact_limits_and_orders_catalysts(macro):
    now = time.time()
    macro.save({"generated_at": now, "risk_level": "high", "headline": "s", "catalysts": [
        _cat(f"c{i}", severity=i * 10, last_seen=now) for i in range(1, 8)
    ]})
    block = macro.compact(macro.load_state(), limit=3)
    assert [c["severity"] for c in block["catalysts"]] == [70, 60, 50]


def test_age_hours_is_reported_for_the_prompt(macro):
    macro.save({"generated_at": time.time() - 7200, "risk_level": "low", "headline": "s", "catalysts": []})
    assert macro.compact(macro.load_state())["age_hours"] == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------- ticker links

def test_sanitize_drops_non_ticker_shaped_strings(macro):
    assert macro.sanitize_tickers(["xom", " CVX ", "Shell Plc", "", "TOOLONGSYM"]) == ["XOM", "CVX"]


def test_sanitize_keeps_share_class_suffixes_and_dedupes(macro):
    assert macro.sanitize_tickers(["BRK.B", "brk.b", "RDS.A"]) == ["BRK.B", "RDS.A"]


def test_hallucinated_tickers_match_nothing(macro):
    """The live run emitted RDS.A (retired 2022) and SHELL (never a US symbol) beside real ones.

    Intersecting against symbols the caller actually knows is what makes that survivable: the bogus
    ones simply never match, so they can't attach a risk to any real name.
    """
    now = time.time()
    macro.save({"generated_at": now, "risk_level": "high", "headline": "s", "catalysts": [
        _cat("iran-war", 85, tickers=["XOM", "CVX", "RDS.A", "SHELL"], last_seen=now),
    ]})
    hits = macro.affected_symbols(macro.load_state(), {"XOM", "AAPL", "MSFT"})
    assert set(hits) == {"XOM"}          # CVX isn't tracked; RDS.A/SHELL don't exist
    assert hits["XOM"][0]["key"] == "iran-war"


def test_affected_sector_words_do_not_match_symbols(macro):
    """"airlines" must not be fuzzy-mapped onto a ticker — that's how a confident wrong claim
    ends up on a stock's page."""
    now = time.time()
    macro.save({"generated_at": now, "risk_level": "high", "headline": "s", "catalysts": [
        _cat("iran-war", 85, affected=["airlines"], tickers=[], last_seen=now),
    ]})
    assert macro.affected_symbols(macro.load_state(), {"UAL", "DAL"}) == {}


def test_affected_symbols_on_an_absent_read_is_empty(macro):
    assert macro.affected_symbols(macro.load_state(), {"XOM"}) == {}


# --------------------------------------------------------------------------- readability

def test_one_line_strips_markdown_bullets(macro):
    """The prompt asks for no bullet glyph; models add one anyway. The UI must never see it."""
    assert macro.one_line("- Oil +7% after strikes", 90) == "Oil +7% after strikes"
    assert macro.one_line("• Hormuz shipping disrupted", 90) == "Hormuz shipping disrupted"
    assert macro.one_line("*  Fed on hold", 90) == "Fed on hold"


def test_one_line_collapses_newlines_and_runs_of_space(macro):
    assert macro.one_line("Oil  +7%\n\nafter   strikes", 90) == "Oil +7% after strikes"


def test_one_line_truncates_on_a_word_boundary(macro):
    out = macro.one_line("The quick brown fox jumps over the lazy dog", 20)
    assert out.endswith("…") and len(out) <= 21
    assert not out.startswith("The quick brown fox j")   # no mid-word cut


def test_one_line_leaves_short_text_untouched(macro):
    assert macro.one_line("Fed on hold", 90) == "Fed on hold"


def test_one_line_handles_empty_and_none(macro):
    assert macro.one_line("", 90) == ""
    assert macro.one_line(None, 90) == ""


def test_legacy_paragraph_blob_still_renders(macro):
    """A blob written before the bullet rewrite must not blank the card between deploy and next run."""
    macro.save({"generated_at": time.time(), "risk_level": "high",
                "summary": "An active Iran war is unfolding with direct US military strikes.",
                "catalysts": []})
    state = macro.load_state()
    assert state["available"] is True
    assert state["headline"].startswith("An active Iran war")
    assert state["bullets"] == []


def test_bullets_survive_into_the_prompt_block(macro):
    macro.save({"generated_at": time.time(), "risk_level": "high", "headline": "Iran war repricing energy",
                "bullets": ["Oil +7% after US strikes", "Hormuz shipping disrupted"], "catalysts": []})
    block = macro.compact(macro.load_state())
    assert block["headline"] == "Iran war repricing energy"
    assert len(block["bullets"]) == 2


def test_long_why_on_a_carried_over_catalyst_is_trimmed_on_read(macro):
    """Entries graded under an older, laxer length rule must not keep leaking paragraphs.

    The first live blobs held `why` strings of 230-310 characters. Trimming only on write would have
    left those on screen until they aged out days later.
    """
    now = time.time()
    macro.save({"generated_at": now, "risk_level": "high", "headline": "h", "catalysts": [
        _cat("iran-war", 85, why="x " * 200, title="y " * 80, last_seen=now),
    ]})
    c = macro.load_state()["catalysts"][0]
    assert len(c["why"]) <= 151 and c["why"].endswith("…")
    assert len(c["title"]) <= 71


def test_blank_bullets_are_dropped(macro):
    macro.save({"generated_at": time.time(), "risk_level": "low", "headline": "h",
                "bullets": ["real", "", "   "], "catalysts": []})
    assert macro.load_state()["bullets"] == ["real"]
