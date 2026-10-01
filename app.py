from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="European CDS Credit Conditions", page_icon="📉", layout="wide")
DATA=Path("data"); HIST=DATA/"cds_dashboard_history.csv"; SNAP=DATA/"cds_dashboard_snapshot.csv"
MODELS=DATA/"cds_dashboard_model_history.csv"; ROLLS=DATA/"cds_dashboard_rolls.csv"

st.markdown("""<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1500px}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);padding:11px 13px;border-radius:8px}
.read{border:1px solid rgba(128,128,128,.24);border-radius:10px;padding:16px 18px;margin:.4rem 0 1rem 0;background:rgba(128,128,128,.035)}
.read h3{margin:0 0 .35rem 0}.muted{opacity:.68;font-size:.83rem}.section{margin-top:.8rem}
</style>""",unsafe_allow_html=True)

st.title("European CDS Credit Conditions")
st.caption("Transaction-derived iTraxx Europe Xover 5Y · Main 5Y · Xover–Main relative value")

if not HIST.exists() or not SNAP.exists():
    st.warning("Dashboard data are not available yet. Run the ‘Update CDS data’ workflow.")
    st.stop()

hist=pd.read_csv(HIST,parse_dates=["date"]); snap=pd.read_csv(SNAP,parse_dates=["date"])
models=pd.read_csv(MODELS,parse_dates=["start","end"]) if MODELS.exists() else pd.DataFrame()
rolls=pd.read_csv(ROLLS,parse_dates=["date"]) if ROLLS.exists() else pd.DataFrame()
CONTRACTS = DATA / "cds_contract_series.csv"; contracts=pd.read_csv(CONTRACTS,parse_dates=["date","maturity"]) if CONTRACTS.exists() else pd.DataFrame()
latest_date=hist.date.max()

def row(name): return snap.loc[snap.series.eq(name)].iloc[-1]
xo,ma,rv=row("Xover"),row("Main"),row("Xover-Main")
def num(v,d=2): return None if pd.isna(v) else round(float(v),d)
def signed(v): return "N/A" if pd.isna(v) else f"{v:+.2f} bp"
def pending(s,need):
    n=int(s.current_regime_obs)
    return f"{s.ou_half_life_obs:.1f} trading observations" if pd.notna(s.ou_half_life_obs) else f"Not enough data ({n}/{need} observations)"
def prior_model(name):
    m=models.loc[models.series.eq(name)].sort_values("end") if not models.empty else pd.DataFrame()
    return None if m.empty else m.iloc[-1]

# Downloads and audit
st.markdown("#### Downloads & data quality")
dl1,dl2,dl3=st.columns([1,1,1])

csv=hist.to_csv(index=False).encode()
with dl1:
    st.download_button("Download CSV",csv,"cds_dashboard_history.csv","text/csv",use_container_width=True)

def latex_escape(x):
    s=str(x)
    for a,b in [("\\","\\textbackslash{}"),("&","\\&"),("%","\\%"),("$","\\$"),("#","\\#"),("_","\\_"),("{","\\{"),("}","\\}")]:
        s=s.replace(a,b)
    return s

def make_latex():
    pmx=prior_model("Xover"); pmm=prior_model("Main"); pmr=prior_model("Xover-Main")
    def hl(m): return "N/A" if m is None or pd.isna(m.half_life_obs) else f"{m.half_life_obs:.1f}"
    rows=[
        ("Xover 5Y",xo.level_bp,xo.change_1d_bp,xo.change_5obs_bp,xo.current_series_percentile,hl(pmx)),
        ("Main 5Y",ma.level_bp,ma.change_1d_bp,ma.change_5obs_bp,ma.current_series_percentile,hl(pmm)),
        ("Xover--Main",rv.level_bp,rv.change_1d_bp,rv.change_5obs_bp,rv.current_series_percentile,hl(pmr)),
    ]
    body="\\n".join([f"{latex_escape(n)} & {lv:.2f} & {d1:+.2f} & {d5:+.2f} & {pct:.1f}\\% & {half} \\\\" for n,lv,d1,d5,pct,half in rows])
    return f"""% European CDS Credit Conditions
% Generated from DTCC/CFTC SDR-derived dashboard data
\\begin{{table}}[htbp]
\\centering
\\caption{{European CDS Credit Conditions -- {latest_date.date()}}}
\\begin{{tabular}}{{lrrrrr}}
\\hline
Series & Level (bp) & 1D (bp) & 5-obs (bp) & Current-series pct. & Prior Mean-reversion half-life \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\end{{table}}

% Current Xover-Main regime observations: {int(rv.current_regime_obs)}.
"""
with dl2:
    if st.button("LaTeX report",use_container_width=True):
        st.session_state["page"]="latex"

# Audit the actual dashboard dataset, not just whether files exist
audit=[]
def check(label,ok,detail):
    audit.append({"Check":label,"Status":"PASS" if bool(ok) else "FAIL","Detail":detail})

check("Latest date aligned",snap.date.nunique()==1 and snap.date.max()==hist.date.max(),f"Snapshot/history: {snap.date.max().date()} / {hist.date.max().date()}")
check("All three series present",set(["Xover","Main","Xover-Main"]).issubset(set(hist.series.unique())),", ".join(sorted(hist.series.unique())))
check("No missing latest levels",snap.level_bp.notna().all(),f"{snap.level_bp.notna().sum()}/{len(snap)} available")
check("No duplicate date/series rows",not hist.duplicated(["date","series"]).any(),f"{int(hist.duplicated(['date','series']).sum())} duplicates")
check("Chronology valid",hist.groupby("series").date.apply(lambda x:x.is_monotonic_increasing).all(),"Dates increase within each series")
check("Xover latest plausible",25 < float(xo.level_bp) < 2500,f"{xo.level_bp:.2f} bp")
check("Main latest plausible",5 < float(ma.level_bp) < 500,f"{ma.level_bp:.2f} bp")
check("RV arithmetic",abs(float(rv.level_bp)-(float(xo.level_bp)-float(ma.level_bp)))<1e-6,f"{rv.level_bp:.2f} bp vs Xover-Main")
if not rolls.empty:
    check("Roll table available",len(rolls)>=2,f"{len(rolls)} roll markers")
if not models.empty:
    check("Mean-reversion estimates finite",models.phi.notna().all() and models.half_life_obs.notna().all(),f"{len(models)} completed index-series models")

audit_df=pd.DataFrame(audit)
audit_pass=(audit_df.Status=="PASS").all()
with dl3:
    if st.button("Data audit",use_container_width=True):
        st.session_state["page"]="audit"

if "page" not in st.session_state:
    st.session_state["page"]="dashboard"

if st.session_state["page"]=="latex":
    top_left,top_right=st.columns([8,1])
    with top_left:
        st.header("LaTeX report")
        st.caption("Ready-to-copy LaTeX generated from the latest dashboard state.")
    with top_right:
        if st.button("Close",key="close_latex",use_container_width=True):
            st.session_state["page"]="dashboard"
            st.rerun()
    st.code(make_latex(),language="latex",line_numbers=True)
    st.stop()

if st.session_state["page"]=="audit":
    top_left,top_right=st.columns([8,1])
    with top_left:
        st.header("Data audit")
        st.caption("Integrity checks on the data currently feeding this dashboard.")
    with top_right:
        if st.button("Close",key="close_audit",use_container_width=True):
            st.session_state["page"]="dashboard"
            st.rerun()
    status="PASS" if audit_pass else "FAIL"
    st.metric("Overall audit",status,f"{int((audit_df.Status=='PASS').sum())}/{len(audit_df)} checks passed")
    st.dataframe(audit_df,hide_index=True,use_container_width=True)
    st.stop()



# Current credit read
x5=xo.change_5obs_bp; m5=ma.change_5obs_bp; r5=rv.change_5obs_bp
if pd.notna(x5) and pd.notna(m5):
    if x5>0 and m5>0:
        direction="Credit spreads have widened over the last five observations."
    elif x5<0 and m5<0:
        direction="Credit spreads have tightened over the last five observations."
    else:
        direction="Xover and Main have moved in different directions over the last five observations."
    if abs(x5)>abs(m5)*1.5:
        driver="The move is concentrated in lower-quality credit: Xover has moved materially more than Main."
    elif abs(m5)>abs(x5)*1.5:
        driver="Investment-grade Main has contributed unusually strongly to the move."
    else:
        driver="Xover and Main have contributed more evenly to the move."
else: direction="Recent directional comparison is not yet available."; driver=""

pm=prior_model("Xover-Main")
model_text=(f"The current Xover–Main regime has only {int(rv.current_regime_obs)} observations, so its mean-reversion estimate is not yet shown. "
            + (f"The previous completed index series had AR(1) {pm.phi:.3f} and a {pm.half_life_obs:.1f}-observation half-life." if pm is not None else ""))
st.markdown(f"""<div class="read"><h3>Current credit read</h3>
<b>{direction}</b> {driver}<br>
Over five observations Xover moved <b>{signed(x5)}</b>, Main <b>{signed(m5)}</b>, and Xover–Main moved <b>{signed(r5)}</b>.
{model_text}<br><span class="muted">Latest DTCC-derived observation: {latest_date.date()}</span></div>""",unsafe_allow_html=True)

# Market overview ribbon
st.subheader("Market overview")
a,b,c,d,e,f=st.columns(6)
a.metric("Xover 5Y",f"{xo.level_bp:.2f} bp",signed(xo.change_1d_bp))
b.metric("Main 5Y",f"{ma.level_bp:.2f} bp",signed(ma.change_1d_bp))
c.metric("Xover − Main",f"{rv.level_bp:.2f} bp",signed(rv.change_1d_bp))
d.metric("Xover 5-observation",signed(xo.change_5obs_bp))
e.metric("Main 5-observation",signed(ma.change_5obs_bp))
f.metric("RV 5-observation",signed(rv.change_5obs_bp))

series=st.segmented_control("Focus",["Xover","Main","Xover-Main"],default="Xover-Main")
s=row(series); h=hist.loc[hist.series.eq(series)].sort_values("date").copy()
ranges={"1M":31,"3M":92,"6M":183,"1Y":366,"All":None,"Custom":"custom"}
choice=st.segmented_control("Window",list(ranges),default="1Y")
if choice=="Custom":
    lo,hi=st.date_input("Custom range",(h.date.min().date(),h.date.max().date()),min_value=h.date.min().date(),max_value=h.date.max().date())
    hp=h.loc[h.date.between(pd.Timestamp(lo),pd.Timestamp(hi))].copy()
elif ranges[choice] is None: hp=h.copy()
else: hp=h.loc[h.date>=h.date.max()-pd.Timedelta(days=ranges[choice])].copy()

# Main level chart
fig=go.Figure()
fig.add_trace(go.Scatter(x=hp.date,y=hp.level_bp,mode="lines",name=series,line=dict(width=2)))
if not rolls.empty:
    rr=rolls.loc[rolls.date.between(hp.date.min(),hp.date.max())]
    if series!="Xover-Main": rr=rr.loc[rr.series.eq(series)]
    for dt in rr.date.unique(): fig.add_vline(x=dt,line_dash="dot",opacity=.35)
fig.update_layout(height=410,margin=dict(l=10,r=10,t=20,b=10),hovermode="x unified",yaxis_title="Spread (bp)",xaxis_title=None)
st.plotly_chart(fig,use_container_width=True)

# Context / model cards
pm=prior_model(series)
c1,c2,c3,c4,c5,c6=st.columns(6)
c1.metric("Current-series percentile",f"{s.current_series_percentile:.1f}%")
c2.metric("20D z-score","Building new roll history" if pd.isna(s.zscore_20) else f"{s.zscore_20:.2f}",help=f"Needs 10 observations in the current index series. Currently {int(s.current_regime_obs)}/10.")
c3.metric("60D z-score","Building new roll history" if pd.isna(s.zscore_60) else f"{s.zscore_60:.2f}",help=f"Needs 30 observations in the current index series. Currently {int(s.current_regime_obs)}/30.")
c4.metric("20D spread vol","Building new roll history" if pd.isna(s.realized_vol_20_bp) else f"{s.realized_vol_20_bp:.2f} bp/day",help=f"Needs 10 observations in the current index series. Currently {int(s.current_regime_obs)}/10.")
c5.metric("Current mean-reversion","Building new roll history" if pd.isna(s.ou_half_life_obs) else f"{s.ou_half_life_obs:.1f} obs",help=f"Needs 20 observations in the current index series. Currently {int(s.current_regime_obs)}/20.")
c6.metric("Previous index-series half-life","N/A" if pm is None else f"{pm.half_life_obs:.1f} obs")

# Xover vs Main comparison
st.subheader("Xover vs Main")
xh=hist.loc[hist.series.eq("Xover"),["date","level_bp"]].rename(columns={"level_bp":"Xover"})
mh=hist.loc[hist.series.eq("Main"),["date","level_bp"]].rename(columns={"level_bp":"Main"})
cmp=xh.merge(mh,on="date").sort_values("date")
if choice not in ("All","Custom"): cmp=cmp.loc[cmp.date>=cmp.date.max()-pd.Timedelta(days=ranges[choice])]
elif choice=="Custom": cmp=cmp.loc[cmp.date.between(pd.Timestamp(lo),pd.Timestamp(hi))]
fig2=go.Figure()
fig2.add_trace(go.Scatter(x=cmp.date,y=cmp.Xover,mode="lines",name="Xover"))
fig2.add_trace(go.Scatter(x=cmp.date,y=cmp.Main,mode="lines",name="Main",yaxis="y2"))
fig2.update_layout(height=360,margin=dict(l=10,r=10,t=15,b=10),hovermode="x unified",
 yaxis=dict(title="Xover (bp)"),yaxis2=dict(title="Main (bp)",overlaying="y",side="right"),legend=dict(orientation="h"))
st.plotly_chart(fig2,use_container_width=True)

# Change and z-score evidence
left,right=st.columns(2)
with left:
    st.subheader("Daily spread change")
    f=go.Figure(go.Bar(x=hp.date,y=hp.change_1d_bp,name="Daily change"))
    f.update_layout(height=320,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="bp",xaxis_title=None)
    st.plotly_chart(f,use_container_width=True)
with right:
    st.subheader("Relative rolling z-score")
    st.caption("Compares Xover, Main and Xover–Main on the same standardized scale. Gaps are intentional at index rolls; each series restarts within its new index series.")
    z=go.Figure()
    for nm in ["Xover","Main","Xover-Main"]:
        zh=hist.loc[hist.series.eq(nm)].sort_values("date").copy()
        if choice not in ("All","Custom"):
            zh=zh.loc[zh.date>=zh.date.max()-pd.Timedelta(days=ranges[choice])]
        elif choice=="Custom":
            zh=zh.loc[zh.date.between(pd.Timestamp(lo),pd.Timestamp(hi))]
        z.add_trace(go.Scatter(x=zh.date,y=zh.zscore_20,mode="lines",name=f"{nm} 20D"))
    z.add_hline(y=0,opacity=.25); z.add_hline(y=2,line_dash="dot",opacity=.25); z.add_hline(y=-2,line_dash="dot",opacity=.25)
    z.update_layout(height=320,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="z",xaxis_title=None,legend=dict(orientation="h"))
    st.plotly_chart(z,use_container_width=True)

# Transaction activity / liquidity
if "trade_count" in h.columns and h["trade_count"].notna().any():
    st.subheader("Transaction activity")
    st.caption("Daily number of filtered on-the-run CDS transactions and aggregate reported notional from the public transaction tape. Reported notional is a market-activity measure, not open interest.")
    vleft,vright=st.columns(2)
    with vleft:
        vf=go.Figure(go.Bar(x=hp.date,y=hp.trade_count,name="Trades"))
        vf.update_layout(height=300,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="Transactions",xaxis_title=None)
        st.plotly_chart(vf,use_container_width=True)
    with vright:
        nf=go.Figure(go.Bar(x=hp.date,y=hp.reported_notional_eur/1e6,name="Reported notional"))
        nf.update_layout(height=300,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="EUR millions",xaxis_title=None)
        st.plotly_chart(nf,use_container_width=True)

# Actual traded index series / roll transparency
st.subheader("Actual traded index series")
st.caption("These are the unadjusted on-the-run contracts actually observed in the DTCC tape. A new line starts when the market rolls to a new iTraxx series/maturity. This view is kept separate from any future roll-adjusted analytical series.")
if not contracts.empty:
    cc=contracts.loc[contracts.series.eq(series if series!="Xover-Main" else "Xover")].sort_values("date").copy()
    if choice not in ("All","Custom"):
        cc=cc.loc[cc.date>=cc.date.max()-pd.Timedelta(days=ranges[choice])]
    elif choice=="Custom":
        cc=cc.loc[cc.date.between(pd.Timestamp(lo),pd.Timestamp(hi))]
    cf=go.Figure()
    for mat,gc in cc.groupby("maturity"):
        cf.add_trace(go.Scatter(x=gc.date,y=gc.spread_bp,mode="lines",name=f"Matures {pd.Timestamp(mat).date()}"))
    cf.update_layout(height=330,margin=dict(l=10,r=10,t=10,b=10),hovermode="x unified",yaxis_title="Spread (bp)",xaxis_title=None,legend=dict(orientation="h"))
    st.plotly_chart(cf,use_container_width=True)
    latest_contract=cc.iloc[-1]
    st.caption(f"Current displayed contract maturity: {pd.Timestamp(latest_contract.maturity).date()} · Reported notional volume and trade count are retained for every daily observation.")

# Persistence
st.subheader("Persistence and mean reversion")
st.caption("This estimates how long a spread dislocation tends to persist. Half-life is the estimated number of trading observations required for roughly half of a deviation from the index series' statistical equilibrium to decay. A higher AR(1) value means greater persistence. These are statistical diagnostics, not forecasts or fundamental fair values.")
if not models.empty:
    mm=models.loc[models.series.eq(series)].copy().sort_values("start")
    if len(mm):
        show=mm[["start","end","obs","phi","half_life_obs","long_run_mean_bp","residual_std_bp","persistence"]].copy()
        show.columns=["Start","End","Observations","Persistence coefficient (AR1)","Mean-reversion half-life (obs)","Statistical equilibrium (bp)","Residual volatility (bp)","Persistence"]
        st.dataframe(show,hide_index=True,use_container_width=True)
st.caption(f"Current index series: {int(s.current_regime_obs)} observations. Mean-reversion estimate requires 20 observations; 20D z-score requires 10; 60D z-score requires 30.")

with st.expander("Methodology and data status"):
    st.write("Source: DTCC/CFTC SDR public dissemination retrieved via OpenBB. Daily levels are medians of filtered on-the-run executed transactions. Direct decimal spread quotes are converted to basis points. On-the-run maturity is selected by reported uncapped notional with maturity prevented from moving backward. A 5-MAD within-day filter removes extreme transaction outliers. Cross-roll changes are excluded and AR(1)/OU statistics are estimated separately by index series.")
    st.write(f"Latest data date: **{latest_date.date()}**. Xover current regime: **{int(xo.current_regime_obs)} obs**; Main: **{int(ma.current_regime_obs)} obs**; Xover–Main: **{int(rv.current_regime_obs)} obs**.")
