from pathlib import Path
from datetime import date, timedelta
import time
import numpy as np
import pandas as pd
from openbb import obb

OUT = Path("data")
OUT.mkdir(exist_ok=True)
LOOKBACK = 370

def download(index):
    frames=[]
    end=date.today()-timedelta(days=1)
    start=end-timedelta(days=LOOKBACK)
    for d in pd.date_range(start,end,freq="D"):
        try:
            x=obb.cftc.cds_index_trades(index=index,tenor="5Y",date=d.date().isoformat(),provider="cftc").to_df()
            if x is not None and not x.empty:
                frames.append(x)
        except Exception:
            pass
        time.sleep(.03)
    if not frames:
        raise RuntimeError(f"No DTCC data returned for {index}")
    return pd.concat(frames,ignore_index=True)

def clean(raw, kind):
    raw=raw.copy()
    raw["execution_timestamp"]=pd.to_datetime(raw["execution_timestamp"],utc=True,errors="coerce")
    raw["maturity_date"]=pd.to_datetime(raw["maturity_date"],errors="coerce")
    raw["spread"]=pd.to_numeric(raw["spread"],errors="coerce")
    raw["spread_notation"]=pd.to_numeric(raw["spread_notation"],errors="coerce")
    raw["notional_amount"]=pd.to_numeric(raw["notional_amount"],errors="coerce")
    b=raw[(raw.action_type=="NEWT")&(raw.event_type=="TRAD")&raw.spread.notna()&
          raw.execution_timestamp.notna()&raw.maturity_date.notna()&
          raw.notional_currency.eq("EUR")&(raw.spread_notation==3.0)].copy()
    b["spread_bp"]=b.spread*10000
    b["trade_date"]=b.execution_timestamp.dt.tz_convert(None).dt.normalize()
    bounds=(25,2500) if kind=="Xover" else (5,500)
    b=b[b.spread_bp.between(*bounds)]
    u=b[(~b.is_capped.fillna(False))&b.notional_amount.notna()]
    liq=(u.groupby(["trade_date","maturity_date"],as_index=False)
          .agg(notional=("notional_amount","sum"),trades=("trade_key","nunique"))
          .sort_values(["trade_date","notional","trades"],ascending=[True,False,False]))
    w=(liq.drop_duplicates("trade_date")[["trade_date","maturity_date"]]
         .sort_values("trade_date").reset_index(drop=True)
         .rename(columns={"maturity_date":"candidate"}))
    selected=[]; current=None
    for proposed in w.candidate:
        if current is None or proposed>=current: current=proposed
        selected.append(current)
    w["otr_maturity"]=selected
    o=b.merge(w[["trade_date","otr_maturity"]],on="trade_date")
    o=o[o.maturity_date.eq(o.otr_maturity)].copy()
    keep=pd.Series(True,index=o.index,dtype=bool)
    for _,idx in o.groupby(["trade_date","maturity_date"]).groups.items():
        v=o.loc[idx,"spread_bp"]; med=v.median(); mad=np.median(np.abs(v-med))
        if np.isfinite(mad) and mad>0:
            keep.loc[idx]=(0.67448975*(v-med)/mad).abs()<=5
    o=o.loc[keep].sort_values(["trade_date","execution_timestamp"])
    prefix="xover" if kind=="Xover" else "main"
    d=(o.groupby("trade_date",as_index=False)
        .agg(**{f"{prefix}_bp":("spread_bp","median"),f"{prefix}_maturity":("maturity_date","first")})
        .sort_values("trade_date").reset_index(drop=True))
    d[f"{prefix}_roll"]=d[f"{prefix}_maturity"].ne(d[f"{prefix}_maturity"].shift())
    d.loc[0,f"{prefix}_roll"]=False
    return d

def add_analytics(df,value,regime,prefix):
    df[f"{prefix}_change_1d"]=df.groupby(regime)[value].diff()
    for n in (5,20,60):
        df[f"{prefix}_change_{n}obs"]=df.groupby(regime)[value].transform(lambda s:s-s.shift(n))
    g=df.groupby(regime,group_keys=False)
    for w,m in ((20,10),(60,30)):
        mu=g[value].transform(lambda s:s.rolling(w,min_periods=m).mean())
        sd=g[value].transform(lambda s:s.rolling(w,min_periods=m).std())
        df[f"{prefix}_z{w}"]=(df[value]-mu)/sd
        df[f"{prefix}_vol{w}"]=g[f"{prefix}_change_1d"].transform(lambda s:s.rolling(w,min_periods=m).std())
    df[f"{prefix}_regime_percentile"]=g[value].transform(
        lambda s:s.expanding().apply(lambda x:pd.Series(x).rank(pct=True).iloc[-1]*100,raw=False))
    return df

def ou_table(df,value,regime,label):
    rows=[]
    for r,g in df.groupby(regime):
        x=g[value].dropna()
        if len(x)<20: continue
        lag=x.iloc[:-1].to_numpy(); nxt=x.iloc[1:].to_numpy()
        X=np.column_stack([np.ones(len(lag)),lag])
        alpha,phi=np.linalg.lstsq(X,nxt,rcond=None)[0]
        resid=nxt-X@np.array([alpha,phi])
        if 0<phi<1:
            k=-np.log(phi); hl=np.log(2)/k; lr=alpha/(1-phi)
        else: k=hl=lr=np.nan
        rows.append(dict(series=label,regime=r,start=g.trade_date.min(),end=g.trade_date.max(),obs=len(x),
                         alpha=alpha,phi=phi,long_run_mean_bp=lr,kappa=k,half_life_obs=hl,
                         residual_std_bp=np.std(resid,ddof=1)))
    return pd.DataFrame(rows)

def persistence(phi):
    if pd.isna(phi): return "N/A"
    if phi>=.98:return "Very High"
    if phi>=.95:return "High"
    if phi>=.90:return "Moderate"
    return "Low"

xo=clean(download("ITRAXX EUROPE CROSSOVER"),"Xover")
mn=clean(download("ITRAXX EUROPE"),"Main")
df=xo.merge(mn,on="trade_date",how="inner").sort_values("trade_date").reset_index(drop=True)
df["rv_bp"]=df.xover_bp-df.main_bp
df["either_roll"]=df.xover_roll|df.main_roll
df["xover_regime"]=df.xover_roll.astype(int).cumsum()
df["main_regime"]=df.main_roll.astype(int).cumsum()
df["rv_regime"]=df.either_roll.astype(int).cumsum()
df=add_analytics(df,"xover_bp","xover_regime","xover")
df=add_analytics(df,"main_bp","main_regime","main")
df=add_analytics(df,"rv_bp","rv_regime","rv")

ou=pd.concat([ou_table(df,"xover_bp","xover_regime","Xover"),
              ou_table(df,"main_bp","main_regime","Main"),
              ou_table(df,"rv_bp","rv_regime","Xover-Main")],ignore_index=True)
if len(ou): ou["persistence"]=ou.phi.apply(persistence)

cfg={
"Xover":("xover_bp","xover_change_1d","xover_change_5obs","xover_change_20obs","xover_z20","xover_z60","xover_vol20","xover_vol60","xover_regime_percentile","xover_regime"),
"Main":("main_bp","main_change_1d","main_change_5obs","main_change_20obs","main_z20","main_z60","main_vol20","main_vol60","main_regime_percentile","main_regime"),
"Xover-Main":("rv_bp","rv_change_1d","rv_change_5obs","rv_change_20obs","rv_z20","rv_z60","rv_vol20","rv_vol60","rv_regime_percentile","rv_regime")}
latest=df.iloc[-1]; snaps=[]; charts=[]
for name,c in cfg.items():
    level,ch,c5,c20,z20,z60,v20,v60,pct,reg=c; rg=latest[reg]; obs=int((df[reg]==rg).sum())
    mm=ou[(ou.series==name)&(ou.regime==rg)] if len(ou) else pd.DataFrame()
    m=mm.iloc[-1] if len(mm) else None
    snaps.append(dict(series=name,date=latest.trade_date,level_bp=latest[level],change_1d_bp=latest[ch],
      change_5obs_bp=latest[c5],change_20obs_bp=latest[c20],zscore_20=latest[z20],zscore_60=latest[z60],
      current_series_percentile=latest[pct],realized_vol_20_bp=latest[v20],realized_vol_60_bp=latest[v60],
      ar1=np.nan if m is None else m.phi,ou_half_life_obs=np.nan if m is None else m.half_life_obs,
      persistence="N/A" if m is None else persistence(m.phi),current_regime=rg,current_regime_obs=obs))
    q=df[["trade_date",level,ch,z20,z60,v20,v60,pct,reg]].copy()
    q.columns=["date","level_bp","change_1d_bp","zscore_20","zscore_60","realized_vol_20_bp","realized_vol_60_bp","current_series_percentile","regime"]
    q["series"]=name; charts.append(q)

pd.DataFrame(snaps).to_csv(OUT/"cds_dashboard_snapshot.csv",index=False)
pd.concat(charts,ignore_index=True).to_csv(OUT/"cds_dashboard_history.csv",index=False)
ou.to_csv(OUT/"cds_dashboard_model_history.csv",index=False)
rolls=[]
for _,r in df.iterrows():
    if r.xover_roll: rolls.append(dict(date=r.trade_date,series="Xover",event="Index Roll",maturity=r.xover_maturity))
    if r.main_roll: rolls.append(dict(date=r.trade_date,series="Main",event="Index Roll",maturity=r.main_maturity))
pd.DataFrame(rolls).to_csv(OUT/"cds_dashboard_rolls.csv",index=False)
print("Updated through",latest.trade_date.date())
