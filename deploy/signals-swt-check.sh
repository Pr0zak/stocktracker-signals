#!/usr/bin/env bash
# Post-wave verification for SWT-1..10 (shipped 2026-08-21/22). Appends to /var/log/signals-swt.log.
#
# Lives on the container because every check needs the local API. Argument 1 labels the run, so a
# Monday line and a Tuesday line are distinguishable in the log.
#
# WHY IT FIRES ON TWO DAYS, WHICH IS NOT ARBITRARY.
#
# The nightly scan at 05:45 measures the most recent session whose bars are FINAL, and 05:45 CT is
# 06:45 ET — before the opening bell. So Monday morning resolves to FRIDAY, which Saturday morning
# already stored, and the run correctly skips. Verified against session_date():
#
#     Sat 05:45 -> 2026-08-21     Mon 05:45 -> 2026-08-21   (all three resolve to Friday)
#     Sun 05:45 -> 2026-08-21     Tue 05:45 -> 2026-08-24   <- first genuinely new session
#
# A Monday-only check would find a skip every week and prove nothing about the scan. Monday is still
# worth having for everything NOT scan-dependent — the gate re-reads live index prices and the arms
# tick — but TUESDAY is when the pipeline advances, so both fire and the script says which case it
# is in rather than leaving a reader to work out whether a skip was expected.
#
# EVERY PYTHON BLOCK IS A HEREDOC READING A TEMP FILE, not an inline -c string. The first version
# passed JSON on stdin and embedded the Python as `python3 -c '...'`, which forced escaped double
# quotes inside f-strings — a SyntaxError on this container's Python 3.11 (backslashes in f-string
# expressions are only legal from 3.12). Every block failed and the log filled with tracebacks.
# A heredoc needs stdin, hence the temp file; do not "simplify" this back into a pipe.
set -uo pipefail

API=http://127.0.0.1:8000
LOG=/var/log/signals-swt.log
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
LABEL="${1:-check}"
exec >>"$LOG" 2>&1

DOW=$(date +%u)          # 1=Mon .. 7=Sun
get() { curl -s -m 90 "$API/$1" > "$TMP/r.json"; }

echo ""
echo "================ $(date -Is) — ${LABEL} ================"
echo "-- service --"
echo "active: $(systemctl is-active signals)   VERSION: $(cat /opt/signals/VERSION 2>/dev/null)"

echo ""
echo "-- market scan (SWT-1) --"
get "market_scan?limit=1"
python3 - "$TMP/r.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"   UNREADABLE: {e}"); raise SystemExit
if "detail" in d:
    print("   REFUSED: " + str(d["detail"])); raise SystemExit
print(f"   session {d.get('as_of')} · scanned {d.get('scanned')} / failed {d.get('fetch_failed')}"
      f" / short {d.get('too_short')} of {d.get('universe_size')}")
print(f"   universe_stale={d.get('universe_stale')}  total_matching={d.get('total_matching')}")
PY

# The job's own summary says whether the run worked, skipped or refused — and a skip is only healthy
# when the session it resolved to was already stored.
python3 - "$DOW" <<'PY'
import json, sys, pathlib
dow = sys.argv[1]
p = pathlib.Path("/opt/signals/data/market_scan_latest.json")
if not p.exists():
    print("   no run summary on disk — the job has never completed"); raise SystemExit
d = json.loads(p.read_text())
status = d.get("status")
print(f"   last run: status={status} session={d.get('session')} scanned={d.get('scanned')} "
      f"suspect={d.get('suspect_series')} retired={d.get('retired')} pruned={d.get('pruned')}")
if status == "skipped":
    expected = dow in ("1", "6", "7")   # Mon, Sat, Sun all resolve to the prior week's last session
    print("   -> skip EXPECTED (this morning resolves to a session already stored)" if expected
          else "   -> skip UNEXPECTED on a weekday — two runs resolved to the same session")
elif status == "ok":
    print("   -> scan advanced")
else:
    print(f"   -> NOT OK: {d.get('reason')}")
PY

echo ""
echo "-- breadth (SWT-1 feeds the gate's third leg) --"
get "market_scan/breadth"
python3 - "$TMP/r.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if not d.get("available"):
    print("   available=False — no scan to report on. This is the honest state, not a zero.")
else:
    print(f"   n={d.get('n')} · above 50-SMA {d.get('pct_above_sma50')}% · "
          f"above 200-SMA {d.get('pct_above_sma200')}%")
    print(f"   at a 52w high {d.get('new_52w_highs')} · within {d.get('near_52w_high_pct')}% "
          f"{d.get('near_52w_high')} · age {d.get('age_hours')}h")
PY

echo ""
echo "-- gate (SWT-2) --"
get "gate"
python3 - "$TMP/r.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"   passed={d.get('passed')} available={d.get('available')} score={d.get('market_score')}")
if d.get("failing"):
    print("   FAILING:    " + str(d["failing"]))
if d.get("unmeasured"):
    print("   UNMEASURED: " + str(d["unmeasured"]) + "   <- not the same claim as failing")
for l in d.get("legs") or []:
    print(f"     {str(l.get('ok')):5} {l.get('name')}")
PY

# The gate history is the DENOMINATOR for reading the gated arm: a flat main-vs-gated comparison
# means nothing until some days are known to have been shut.
get "gate/history?limit=30"
python3 - "$TMP/r.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
rows = (d.get("history") if isinstance(d, dict) else d) or []
shut = [r for r in rows if r.get("passed") is False]
unk  = [r for r in rows if r.get("passed") is None]
print(f"   history: {len(rows)} day(s) · {len(shut)} shut · {len(unk)} undecidable")
if not shut:
    print("   -> the gate has never closed, so the gated arm CANNOT have diverged yet.")
    print("      A flat main-vs-gated curve is expected — not evidence the gate does nothing.")
else:
    print("   -> shut on: " + str([r.get("d") for r in shut]))
PY

echo ""
echo "-- percentiles (SWT-4) --"
get "market_scan/NVDA"
python3 - "$TMP/r.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if "detail" in d:
    print("   " + str(d["detail"])); raise SystemExit
p = d.get("percentiles") or {}
ranked   = [k for k, v in p.items() if v is not None]
unranked = [k for k, v in p.items() if v is None]
print(f"   NVDA as_of {d.get('as_of')} · latest {d.get('latest_scan_date')} · "
      f"is_latest={d.get('is_latest_night')}")
print(f"   ranked {len(ranked)}/{len(p)} metrics over {d.get('percentiles_over')} names")
if unranked:
    print("   unranked: " + str(unranked) + "   (null, never 0 — the pass may not have run)")
PY

echo ""
echo "-- sandbox: main vs gated (the SWT-2 experiment) --"
get "sandbox/arms"
python3 - "$TMP/r.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for a in d.get("arms") or []:
    if a.get("arm") in ("main", "gated"):
        print(f"   {str(a.get('arm')):6} cash={a.get('cash')} cash%={a.get('cash_pct')}  "
              f"{a.get('label')}")
PY

echo ""
echo "-- endpoint sweep --"
for r in health regime scan/latest sandbox/state "screener/value?limit=2" "heatmap?limit=5"; do
  printf '   %-26s HTTP %s\n' "$r" "$(curl -s -o /dev/null -w '%{http_code}' -m 90 "$API/$r")"
done

echo ""
echo "-- done --"
