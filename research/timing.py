"""Is the gap edge capturable at 15:35 (the sandbox's decision time)? Compare buying the OPEN vs buying
that day's CLOSE. If the edge lives open->close, a late-day system can't capture it."""
import json, urllib.request, time, statistics as st
HOSTS=["query1.finance.yahoo.com","query2.finance.yahoo.com"]; UA="Mozilla/5.0 (Linux; Android 13)"
def fetch(s):
    for h in HOSTS:
        try:
            u=f"https://{h}/v8/finance/chart/{s}?range=5y&interval=1d"
            r=urllib.request.Request(u,headers={"User-Agent":UA})
            d=json.load(urllib.request.urlopen(r,timeout=20)); res=d["chart"]["result"][0]
            q=res["indicators"]["quote"][0]
            return [b for b in zip(res["timestamp"],q["open"],q["high"],q["low"],q["close"],q["volume"]) if None not in b]
        except Exception: continue
    return []
SYMS=["NVDA","AAPL","MSFT","GOOGL","AMZN","META","V","MA","COST","UNH","JNJ","LLY","PFE","SPY","VOO","QQQM","SCHD"]
from_open=[]; from_close=[]; intraday=[]; base_close=[]
for s in SYMS:
    bars=fetch(s)
    if len(bars)<300: continue
    for i in range(21,len(bars)-11):
        _,o,hi,lo,c,v=bars[i]; pc=bars[i-1][4]
        if pc<=0: continue
        base_close.append((bars[i+5][4]-c)/c*100)
        g=(o-pc)/pc*100
        if -5 <= g <= -2:                      # the bucket that showed an edge
            c5=bars[i+5][4]
            from_open.append((c5-o)/o*100)
            from_close.append((c5-c)/c*100)
            intraday.append((c-o)/o*100)
    time.sleep(0.3)
def show(l,x): print(f"{l:38} n={len(x):5}  median {st.median(x):+.2f}%  mean {st.mean(x):+.2f}%  win {sum(1 for v in x if v>0)/len(x)*100:.0f}%")
print("GAP DOWN 2-5% —")
show("  buy the OPEN, hold 5d", from_open)
show("  buy that day's CLOSE, hold 5d", from_close)
show("  intraday open->close (the reversion)", intraday)
print()
show("BASELINE buy any close, hold 5d", base_close)
