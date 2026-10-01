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
latest_date=hist.date.max()

def row(name): return snap.loc[snap.series.eq(name)].iloc[-1]
xo,ma,rv=row("Xover"),row("Main"),row("Xover-Main")
def num(v,d=2): return None if pd.isna(v) else round(float(v),d)
def signed(v): return "N/A" if pd.isna(v) else f"{v:+.2f} bp"
def pending(s,need):
    n=int(s.current_regime_obs)
    return f"{s.ou_half_life_obs:.1f} obs" if pd.notna(s.ou_half_life_obs) else f"Pending ({n}/{need})"
def prior_model(name):
    m=models.loc[models.series.eq(name)].sort_values("end") if not models.empty else pd.DataFrame()
    return None if m.empty else m.iloc[-1]

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
model_text=(f"The current Xover–Main regime has only {int(rv.current_regime_obs)} observations, so its OU estimate is withheld. "
            + (f"The previous completed regime had AR(1) {pm.phi:.3f} and a {pm.half_life_obs:.1f}-observation half-life." if pm is not None else ""))
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
c2.metric("20D z-score","Pending" if pd.isna(s.zscore_20) else f"{s.zscore_20:.2f}")
c3.metric("60D z-score","Pending" if pd.isna(s.zscore_60) else f"{s.zscore_60:.2f}")
c4.metric("20D spread vol","Pending" if pd.isna(s.realized_vol_20_bp) else f"{s.realized_vol_20_bp:.2f} bp/day")
c5.metric("Current OU",pending(s,20))
c6.metric("Previous OU","N/A" if pm is None else f"{pm.half_life_obs:.1f} obs")

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
    st.subheader("Rolling z-score")
    z=go.Figure()
    z.add_trace(go.Scatter(x=hp.date,y=hp.zscore_20,mode="lines",name="20D"))
    z.add_trace(go.Scatter(x=hp.date,y=hp.zscore_60,mode="lines",name="60D"))
    z.add_hline(y=0,opacity=.25); z.add_hline(y=2,line_dash="dot",opacity=.25); z.add_hline(y=-2,line_dash="dot",opacity=.25)
    z.update_layout(height=320,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="z",xaxis_title=None)
    st.plotly_chart(z,use_container_width=True)

# Persistence
st.subheader("Persistence and regime history")
if not models.empty:
    mm=models.loc[models.series.eq(series)].copy().sort_values("start")
    if len(mm):
        show=mm[["start","end","obs","phi","half_life_obs","long_run_mean_bp","residual_std_bp","persistence"]].copy()
        show.columns=["Start","End","Obs","AR(1)","Half-life (obs)","AR long-run level (bp)","Residual σ (bp)","Persistence"]
        st.dataframe(show,hide_index=True,use_container_width=True)
st.caption(f"Current regime: {int(s.current_regime_obs)} observations. OU requires 20 observations; 20D z-score requires 10; 60D z-score requires 30.")

with st.expander("Methodology and data status"):
    st.write("Source: DTCC/CFTC SDR public dissemination retrieved via OpenBB. Daily levels are medians of filtered on-the-run executed transactions. Direct decimal spread quotes are converted to basis points. On-the-run maturity is selected by reported uncapped notional with maturity prevented from moving backward. A 5-MAD within-day filter removes extreme transaction outliers. Cross-roll changes are excluded and AR(1)/OU statistics are estimated separately by contract regime.")
    st.write(f"Latest data date: **{latest_date.date()}**. Xover current regime: **{int(xo.current_regime_obs)} obs**; Main: **{int(ma.current_regime_obs)} obs**; Xover–Main: **{int(rv.current_regime_obs)} obs**.")

csv=hist.to_csv(index=False).encode()
st.download_button("Download dashboard history CSV",csv,"cds_dashboard_history.csv","text/csv")
