"""Is a dividend ETF (SCHD) a good vehicle for pure ROI, or better suited near/after retirement?
Uses ADJUSTED closes (dividends reinvested) = true total return, so the comparison is apples-to-apples."""
import json, urllib.request, time, statistics as st, math
HOSTS=["query1.finance.yahoo.com","query2.finance.yahoo.com"]; UA="Mozilla/5.0 (Linux; Android 13)"

def series(sym, rng="10y"):
    for h in HOSTS:
        try:
            u=f"https://{h}/v8/finance/chart/{sym}?range={rng}&interval=1d&includeAdjustedClose=true"
            r=urllib.request.Request(u,headers={"User-Agent":UA})
            d=json.load(urllib.request.urlopen(r,timeout=25)); res=d["chart"]["result"][0]
            adj=res["indicators"]["adjclose"][0]["adjclose"]
            ts=res["timestamp"]
            return [(t,a) for t,a in zip(ts,adj) if a]
        except Exception: continue
    return []

def stats(name, s):
    if len(s)<200: print(f"{name}: insufficient data"); return None
    px=[a for _,a in s]
    yrs=(s[-1][0]-s[0][0])/(365.25*86400)
    cagr=((px[-1]/px[0])**(1/yrs)-1)*100
    rets=[(px[i]/px[i-1]-1) for i in range(1,len(px))]
    vol=st.pstdev(rets)*math.sqrt(252)*100
    peak=-1e9; mdd=0
    for p in px:
        peak=max(peak,p); mdd=min(mdd,(p-peak)/peak*100)
    sharpe=(cagr-4.0)/vol if vol else 0     # ~4% cash proxy
    print(f"  {name:6} {yrs:4.1f}y  CAGR {cagr:6.2f}%  vol {vol:5.1f}%  maxDD {mdd:7.1f}%  return/risk {sharpe:5.2f}")
    return dict(cagr=cagr,vol=vol,mdd=mdd,px=px,ts=[t for t,_ in s])

print("=== TOTAL RETURN (dividends reinvested), 10 years ===")
out={}
for s in ["SCHD","VOO","SPY","VIG","VYM","QQQ","JEPI"]:
    d=series(s); 
    if d: out[s]=stats(s,d)
    time.sleep(0.3)

# Down-market behavior: 2022 bear + worst 60-day windows
print("\n=== DOWNSIDE BEHAVIOR (what dividend/quality tilts are supposed to buy you) ===")
def window_ret(d, start_iso, end_iso):
    import datetime as dt
    a=dt.datetime.fromisoformat(start_iso).timestamp(); b=dt.datetime.fromisoformat(end_iso).timestamp()
    pts=[(t,p) for t,p in zip(d["ts"],d["px"]) if a<=t<=b]
    return (pts[-1][1]/pts[0][1]-1)*100 if len(pts)>2 else None
for s in ["SCHD","VOO","QQQ"]:
    if s in out:
        r=window_ret(out[s],"2022-01-01","2022-12-31")
        print(f"  {s:6} 2022 bear year: {r:+.1f}%")
