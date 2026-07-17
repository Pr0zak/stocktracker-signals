# StockTracker Signals (Tier 2)

A small self-hosted FastAPI service that turns a stock/crypto's daily technical picture into a
**structured, explained buy/sell verdict** from Claude. It's the Tier-2 "analyst" layer of the
StockTracker AI-signals roadmap (`stocktracker/docs/ai-signals-roadmap.md`).

**Decision support only — not investment advice.** The API key lives here, server-side, never in
the Android app.

## What it does

1. Fetches ~1 year of daily bars (Yahoo chart endpoint, query1→query2 failover).
2. Computes a compact technical snapshot — the same indicators the phone's Tier-1 engine uses
   (RSI, MACD, SMA20/50, Bollinger %B, Stochastic, 52-week position, 3-month relative strength
   vs the S&P).
3. Asks Claude (structured output) for a verdict: `signal`, `conviction`, `horizon`, `thesis`,
   `rationale[]`, `key_risks[]`, `invalidation`, `catalysts[]`.
4. Caches the verdict (default 4h; signals are daily).

Models: `claude-haiku-4-5` for the watchlist scan, `claude-opus-4-8` (with adaptive thinking) for
on-demand deep dives.

## Run locally

```bash
cp .env.example .env      # add your ANTHROPIC_API_KEY
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Or Docker:

```bash
docker build -t stocktracker-signals .
docker run --rm -p 8000:8000 --env-file .env stocktracker-signals
```

## Endpoints

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/health` | liveness + configured models |
| GET  | `/signal/{symbol}?deep=false&crypto=false` | one verdict. `deep=true` → Opus. Crypto uses Yahoo's `BTC-USD` form + `crypto=true` (skips the S&P benchmark). |
| POST | `/scan` | `{"symbols": [...], "crypto_symbols": [...]}` — scores a watchlist with the cheap model. |

```bash
curl 'localhost:8000/signal/NVDA'
curl 'localhost:8000/signal/NVDA?deep=true'
curl 'localhost:8000/signal/BTC-USD?crypto=true'
curl -X POST localhost:8000/scan -H 'content-type: application/json' \
     -d '{"symbols":["NVDA","MSFT"],"crypto_symbols":["BTC-USD"]}'
```

## Not done yet (roadmap)

- **Nightly Batch-API scan + push** (task #6): move `/scan` to the Anthropic Batch API (~50% cost)
  with prompt caching on the system prompt, on a cron, pushing notable flips to the app's existing
  alert channel.
- **News / earnings context** (task #3→Tier 3): feed Finnhub company-news + earnings-calendar into
  the snapshot to sharpen `catalysts`/`key_risks`.
- Deploy to a self-hosted container; persist the cache; add auth if exposed beyond the LAN.
