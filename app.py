from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="CDS Credit Dashboard", page_icon="📉", layout="wide")

DATA = Path("data")
HIST = DATA / "cds_dashboard_history.csv"
SNAP = DATA / "cds_dashboard_snapshot.csv"
MODELS = DATA / "cds_dashboard_model_history.csv"
ROLLS = DATA / "cds_dashboard_rolls.csv"

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25); padding:12px 14px; border-radius:8px;}
.small-note {opacity:.68;font-size:.82rem;}
</style>
""", unsafe_allow_html=True)

st.title("European CDS Credit Conditions")
st.caption("iTraxx Europe Xover 5Y · iTraxx Europe Main 5Y · Xover–Main relative value")

if not HIST.exists():
    st.warning("No dashboard data yet. Run the GitHub Actions workflow ‘Update CDS data’ once, then reload.")
    st.stop()

hist = pd.read_csv(HIST, parse_dates=["date"])
snap = pd.read_csv(SNAP, parse_dates=["date"])
models = pd.read_csv(MODELS, parse_dates=["start","end"]) if MODELS.exists() else pd.DataFrame()
rolls = pd.read_csv(ROLLS, parse_dates=["date"]) if ROLLS.exists() else pd.DataFrame()

series = st.segmented_control(
    "Series", ["Xover", "Main", "Xover-Main"], default="Xover", label_visibility="collapsed"
)
s = snap.loc[snap["series"].eq(series)].iloc[-1]
h = hist.loc[hist["series"].eq(series)].sort_values("date").copy()

def fmt(v, suffix="", digits=2):
    return "N/A" if pd.isna(v) else f"{v:.{digits}f}{suffix}"

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Level", fmt(s.level_bp," bp"))
c2.metric("1D move", fmt(s.change_1d_bp," bp"))
c3.metric("5-observation move", fmt(s.change_5obs_bp," bp"))
c4.metric("20D z-score", fmt(s.zscore_20))
c5.metric("Current-series percentile", fmt(s.current_series_percentile,"%",1))
c6.metric("OU half-life", fmt(s.ou_half_life_obs," obs",1))

st.caption(f"Latest observation: {s.date.date()} · Current regime observations: {int(s.current_regime_obs)} · Persistence: {s.persistence}")

ranges = {"1M":31,"3M":92,"6M":183,"1Y":366,"All":None}
choice = st.segmented_control("Window", list(ranges), default="1Y")
if ranges[choice] is not None:
    cutoff = h["date"].max() - pd.Timedelta(days=ranges[choice])
    hp = h.loc[h["date"] >= cutoff].copy()
else:
    hp = h.copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=hp.date, y=hp.level_bp, mode="lines", name=series, line=dict(width=2)))
if not rolls.empty and series != "Xover-Main":
    rr = rolls.loc[rolls.series.eq(series) & rolls.date.between(hp.date.min(), hp.date.max())]
    for d in rr.date:
        fig.add_vline(x=d, line_dash="dot", opacity=.45)
fig.update_layout(
    height=470, margin=dict(l=10,r=10,t=25,b=10), hovermode="x unified",
    yaxis_title="Spread (bp)", xaxis_title=None, legend_title=None
)
st.plotly_chart(fig, use_container_width=True)

left,right = st.columns(2)
with left:
    st.subheader("Signal state")
    sig = pd.DataFrame({
        "Metric":["20D z-score","60D z-score","20D realized spread vol","60D realized spread vol","Current-series percentile"],
        "Value":[fmt(s.zscore_20),fmt(s.zscore_60),fmt(s.realized_vol_20_bp," bp/day"),fmt(s.realized_vol_60_bp," bp/day"),fmt(s.current_series_percentile,"%",1)]
    })
    st.dataframe(sig, hide_index=True, use_container_width=True)

with right:
    st.subheader("Historical persistence regimes")
    if not models.empty:
        mm = models.loc[models.series.eq(series)].copy()
        if len(mm):
            show = mm[["start","end","obs","phi","half_life_obs","residual_std_bp","persistence"]].copy()
            show.columns = ["Start","End","Obs","AR(1)","Half-life","Residual σ (bp)","Persistence"]
            st.dataframe(show, hide_index=True, use_container_width=True)
        else:
            st.info("No completed regime model available.")
    else:
        st.info("No model history available.")

st.markdown('<div class="small-note">Transaction source: DTCC/CFTC SDR public dissemination via OpenBB. Daily levels are robust medians of filtered on-the-run transactions. Roll boundaries are excluded from change and persistence estimation; percentiles are within the current contract regime.</div>', unsafe_allow_html=True)
