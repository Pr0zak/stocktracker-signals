#!/usr/bin/env bash
# Post-tick verification for the sandbox rule changes (app v0.77.0 / signals 0.11.0).
#
# Lives on the CT because every check needs the LAN — a cloud agent cannot reach this host.
# Appends to /var/log/signals-check.log. Argument 1 is a label for the run.
#
# The rules added on 2026-08-03 are PROMPT-LEVEL, not mechanical gates — the validator cannot verify
# an intention. So this script's real job is to surface every SELL and let a human judge whether it
# stood on its own merits, and to flag the language that would mean the rule was ignored.
API=http://127.0.0.1:8000
LOG=/var/log/signals-check.log
exec >>"$LOG" 2>&1

echo
echo "================ $(date -Is) — ${1:-check} ================"

echo "-- service --"
systemctl is-active signals
echo "version: $(cat /opt/signals/VERSION 2>/dev/null)"

echo "-- macro read --"
curl -fsS -m 20 "$API/macro/catalysts" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("available=%s risk=%s catalysts=%d age=%ss stale=%s degraded=%s" % (
    d["available"], d["risk_level"], len(d["catalysts"]), d["age_seconds"], d["stale"], d["degraded"]))
print("headline:", d.get("headline"))
if not d["available"]: print("!! NO MACRO READ — the news pipeline is not producing")
if d["degraded"]:      print("!! DEGRADED — last refresh failed, showing an older read")
' || echo "!! MACRO CHECK FAILED"

echo "-- sandbox state --"
curl -fsS -m 20 "$API/sandbox/state" | python3 -c '
import json, sys
d = json.load(sys.stdin); s = d["settings"]
runway = (s.get("retirement_age") or 0) - (s.get("current_age") or 0)
print("equity=%.2f funded=%.2f return=%s%% vs_spy=%s%% realized=%.2f" % (
    d["equity"], d["funded_total"], d["total_return_pct"], d["vs_benchmark_pct"],
    d["realized_pl_total"]))
print("last_tick=%s runway=%dy taxable=%s preferred_btc_etf=%s" % (
    d["last_tick_date"], runway, s.get("taxable_account"), s.get("preferred_btc_etf")))
for p in d["positions"]:
    print("   %-6s %s sh @ avg %.2f" % (p["symbol"], p["shares"], p["avg_cost"]))
' || echo "!! SANDBOX CHECK FAILED"

echo "-- todays orders --"
curl -fsS -m 20 "$API/sandbox/trades?limit=40" | python3 -c '
import json, sys, datetime
rows = json.load(sys.stdin)["trades"]
today = datetime.date.today().isoformat()
todays = [t for t in rows if t["date"] == today]
print("%d order row(s) dated %s" % (len(todays), today))
if not todays:
    print("   (the daily tick fires at 14:35 CT)")

# Language that would mean the "never sell to fund a purchase" rule was ignored. Prompt rules cannot
# be enforced by the validator, so this is the only place a violation becomes visible.
# Language that means the sell rules were bypassed. FUNDING was the original list; the FEE/SWAP
# terms were added after 2026-08-06, when the model sold SPY to buy VTI on expense-ratio grounds and
# this check reported "no funding language" — a true statement and a useless one. The rationalisation
# does not have to mention funding to be the forbidden trade.
FUNDING = ("to fund", "fund the", "funding", "rotate into", "raise cash", "free up cash",
           "to buy ", "reallocat", "redeploy",
           "expense ratio", " er ", "lower fee", "cheaper", "swap", "same exposure",
           "rebalance within", "consolidat")
NEW_INPUTS = ("200-week", "200w", "long-cycle", "below its 200", "drawdown", "mayer",
              "accumulation", "holding_days", "short-term", "long-term", "capital gain",
              "iran", "macro", "oil", "hormuz", "routed")

for t in todays:
    print("   %-5s %-8s %-8s conv=%s" % (t["side"], t["status"], t["symbol"], t.get("conviction")))
    print("       %s" % t["reason"][:200])
    if t.get("skip_reason"):
        print("       blocked: %s" % t["skip_reason"])
    r = t["reason"].lower()
    hits = [k for k in NEW_INPUTS if k in r]
    if hits:
        print("       ^ cites:", ", ".join(sorted(set(hits))))
    if t["side"] == "sell":
        bad = [k for k in FUNDING if k in r]
        if bad:
            print("       !! SELL cites FUNDING language %s — the rule says sells must stand on" % bad)
            print("          their own (extreme extension, or thesis breaking). REVIEW THIS.")
        else:
            print("       ^ sell reason carries no funding language")
' || echo "!! TRADES CHECK FAILED"

echo "-- warnings in the last 24h --"
journalctl -u signals -u signals-macro -u signals-sandbox --since "24 hours ago" \
    --no-pager -p warning 2>/dev/null | tail -20

echo "================ end ================"
