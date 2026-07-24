# Gap study (2026-07-24)

Re-runnable scripts behind the base rates encoded in `app/gaps.py`.

- `study.py` — down/up gap fill rates by size bucket and volume, plus 5-day forward returns.
- `baseline.py` — the control: unconditional 5-day forward return, and "closed down >1%" (buying
  weakness generally), which the gap edge must beat.
- `timing.py` — whether the edge survives to the sandbox's ~15:35 ET decision time (buy the open vs
  buy that day's close).

Universe: 17 mega-cap/ETF names from the app's own watchlist, 5 years of Yahoo daily bars.
Run with plain `python3 research/study.py` (stdlib only, no deps).

Headline: only the **2-5% down-gap on normal volume** beat baseline (+1.09pp over 5 days from the
open; ~+0.5pp still there buying the close). Small gaps and catalyst (high-volume) gaps showed no
edge, and buying general weakness measured NEGATIVE. Big gaps (>5%) filled within 10 days only ~39%
of the time — "gaps always fill" is false where it matters most.

# Dividend-ETF study (2026-07-24) — `div.py`

Does a dividend ETF make sense for pure ROI, or is it a near/post-retirement vehicle?
Adjusted closes (dividends reinvested) = true total return.

| 10y | CAGR | vol | max DD | 2022 bear |
|---|---|---|---|---|
| QQQ  | 20.5% | 22.5% | -35.1% | **-33.2%** |
| VOO  | 14.9% | 18.0% | -34.0% | **-18.7%** |
| VIG  | 12.9% | 16.0% | -31.7% | — |
| **SCHD** | **12.5%** | 16.7% | -33.4% | **-3.2%** |
| VYM  | 11.6% | 16.3% | -35.2% | — |
| JEPI | 11.2% (6y) | 10.7% | -13.7% | — |

**Conclusion:** a dividend tilt cost ~2.4pp/yr of total return vs VOO over the decade, so it is the
wrong vehicle for pure accumulation — a dividend isn't free money (price drops by the payout, and it
forces a taxable event in a taxable account). What it actually buys is drawdown protection from its
quality/value tilt (2022: -3.2% vs -18.7%), which is what matters near/in retirement because a deep
drawdown while withdrawing causes permanent damage (sequence-of-returns risk).

**Caveat:** the 10-year window is dominated by a mega-cap growth bull market, which flatters VOO/QQQ.
