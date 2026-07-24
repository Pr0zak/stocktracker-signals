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
