"""The control the gap study needs: what does a RANDOM day's next-5-day return look like for the same
names/period? Any gap 'edge' must beat this, otherwise it's just market drift."""
import json, urllib.request, time, statistics as st
HOSTS=["query1.finance.yahoo.com","query2.finance.yahoo.com"]; UA="Mozilla/5.0 (Linux; Android 13)"
def fetch(sym):
    for h in HOSTS:
        try:
            url=f"https://{h}/v8/finance/chart/{sym}?range=5y&interval=1d"
            r=urllib.request.Request(url,headers={"User-Agent":UA})
            d=json.load(urllib.request.urlopen(r,timeout=20)); res=d["chart"]["result"][0]
            q=res["indicators"]["quote"][0]
            return [b for b in zip(res["timestamp"],q["open"],q["high"],q["low"],q["close"]) if None not in b]
        except Exception: continue
    return []
SYMS=["NVDA","AAPL","MSFT","GOOGL","AMZN","META","V","MA","COST","UNH","JNJ","LLY","PFE","SPY","VOO","QQQM","SCHD"]
base=[]; downday=[]
for s in SYMS:
    bars=fetch(s)
    if len(bars)<300: continue
    for i in range(21,len(bars)-11):
        o=bars[i][1]; c5=bars[i+5][4]
        base.append((c5-o)/o*100)
        # control 2: days that merely CLOSED down >1% (no gap requirement)
        pc=bars[i-1][4]
        if pc>0 and (bars[i][4]-pc)/pc*100 <= -1.0: downday.append((c5-o)/o*100)
    time.sleep(0.3)
def show(lbl,x):
    print(f"{lbl:34} n={len(x):6}  median {st.median(x):+.2f}%  mean {st.mean(x):+.2f}%  win {sum(1 for v in x if v>0)/len(x)*100:.0f}%")
show("BASELINE any day (5d fwd)", base)
show("CONTROL closed down >1% (5d fwd)", downday)
