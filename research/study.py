"""Empirical gap study on the user's own watchlist — does a gap-down actually fill, and does buying it pay?
Definitions: gap = today's OPEN vs yesterday's CLOSE. A DOWN gap 'fills' when a later day's HIGH >= the
prior close (price returns to where it gapped from). Also measures forward returns from the open."""
import json, urllib.request, time, statistics as st

HOSTS = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
UA = "Mozilla/5.0 (Linux; Android 13)"

def fetch(sym, rng="5y"):
    for h in HOSTS:
        try:
            url = f"https://{h}/v8/finance/chart/{sym}?range={rng}&interval=1d"
            r = urllib.request.Request(url, headers={"User-Agent": UA})
            d = json.load(urllib.request.urlopen(r, timeout=20))
            res = d["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            return list(zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"], q["volume"]))
        except Exception:
            continue
    return []

def study(sym):
    bars = [b for b in fetch(sym) if None not in b]
    if len(bars) < 300: return None
    out = []
    for i in range(21, len(bars)-11):
        _, o, hi, lo, c, v = bars[i]
        prev_c = bars[i-1][4]
        if prev_c <= 0: continue
        gap = (o - prev_c) / prev_c * 100
        if abs(gap) < 0.5: continue          # ignore noise-level gaps
        avg_vol = st.mean([b[5] for b in bars[i-21:i-1]] ) or 1
        vol_ratio = v / avg_vol
        # fill: for a DOWN gap, a later HIGH reaches prior close; for UP, a later LOW reaches it
        filled_by = None
        for k in range(0, 11):
            fb = bars[i+k]
            if gap < 0 and fb[2] >= prev_c: filled_by = k; break
            if gap > 0 and fb[3] <= prev_c: filled_by = k; break
        fwd5 = (bars[i+5][4] - o) / o * 100
        out.append({"gap": gap, "vol_ratio": vol_ratio, "filled_by": filled_by, "fwd5": fwd5})
    return out

SYMS = ["NVDA","AAPL","MSFT","GOOGL","AMZN","META","V","MA","COST","UNH","JNJ","LLY","PFE","SPY","VOO","QQQM","SCHD"]
rows = []
for s in SYMS:
    r = study(s)
    if r: rows += r; print(f"  {s}: {len(r)} gaps", flush=True)
    time.sleep(0.3)

def summarize(sel, label):
    if not sel: return
    n = len(sel)
    for horizon in (1, 3, 5, 10):
        filled = sum(1 for r in sel if r["filled_by"] is not None and r["filled_by"] < horizon)
        print(f"    fill within {horizon:2}d: {filled/n*100:5.1f}%")
    f5 = [r["fwd5"] for r in sel]
    print(f"    5-day fwd return from the open: median {statistics_median(f5):+.2f}%  mean {st.mean(f5):+.2f}%  win {sum(1 for x in f5 if x>0)/n*100:.0f}%")

def statistics_median(x): return st.median(x)

print(f"\n=== TOTAL GAPS: {len(rows)} ===")
downs = [r for r in rows if r["gap"] < 0]
ups   = [r for r in rows if r["gap"] > 0]
print(f"\nDOWN gaps (n={len(downs)}):"); summarize(downs, "down")
print(f"\nUP gaps (n={len(ups)}):");     summarize(ups, "up")

print("\n--- DOWN gaps by SIZE ---")
for lo, hi in [(0.5,1),(1,2),(2,5),(5,100)]:
    sel = [r for r in downs if lo <= abs(r["gap"]) < hi]
    if len(sel) > 30:
        print(f"  gap -{lo}..-{hi}% (n={len(sel)}):"); summarize(sel, "")

print("\n--- DOWN gaps by VOLUME (catalyst proxy) ---")
for lab, f in [("normal vol <1.5x", lambda r: r["vol_ratio"] < 1.5), ("high vol >=1.5x", lambda r: r["vol_ratio"] >= 1.5)]:
    sel = [r for r in downs if f(r)]
    if len(sel) > 30:
        print(f"  {lab} (n={len(sel)}):"); summarize(sel, "")
