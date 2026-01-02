import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Market Rotation Model – RRG", layout="wide")

# ------------------------------------------------------------
# Universes
# ------------------------------------------------------------
UNIVERSES = {
    "SPDR Sectors": {
        "XLB Materials": "XLB",
        "XLC Communication Services": "XLC",
        "XLE Energy": "XLE",
        "XLF Financials": "XLF",
        "XLI Industrials": "XLI",
        "XLK Technology": "XLK",
        "XLP Consumer Staples": "XLP",
        "XLRE Real Estate": "XLRE",
        "XLU Utilities": "XLU",
        "XLV Health Care": "XLV",
        "XLY Consumer Discretionary": "XLY",
    },
    "Themes ETFs": {
        "Semiconductors (SOXX)": "SOXX",
        "Cybersecurity (CIBR)": "CIBR",
        "Cloud (CLOU)": "CLOU",
        "Infrastructure (PAVE)": "PAVE",
        "Clean Energy (ICLN)": "ICLN",
        "Robotics (BOTZ)": "BOTZ",
        "Quantum (QTUM)": "QTUM",
        "Biotech (XBI)": "XBI",
        "Retail (XRT)": "XRT",
        "Bitcoin Mining (WGMI)": "WGMI",
    },
    "Commodity ETFs": {
        "Gold (GLD)": "GLD",
        "Silver (SLV)": "SLV",
        "Copper (CPER)": "CPER",
        "Oil (USO)": "USO",
        "Bitcoin (BITO)": "BITO",
        "Ethereum (ETHA)": "ETHA",
    },
    "Country ETFs": {
        "Germany (EWG)": "EWG",
        "China (MCHI)": "MCHI",
        "Brazil (EWZ)": "EWZ",
        "Canada (EWC)": "EWC",
        "Singapore (EWS)": "EWS",
        "Emerging ex-China (EMXC)": "EMXC",
        "All World ex-US (VEU)": "VEU",
    },
}

BENCHMARKS = ["SPY", "QQQ", "IWM", "ACWI"]

# ------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers, years):
    end = datetime.today()
    start = end - timedelta(days=365 * years)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.levels[1]:
            close = data.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Close column not found")
    else:
        close = data[["Close"]]
        close.columns = [tickers[0]]

    close = close.dropna(axis=1, how="all")
    return close

def to_weekly(df):
    return df.resample("W-FRI").last().dropna(how="all")

def zscore(df):
    return (df - df.mean()) / df.std(ddof=0)

# ------------------------------------------------------------
# RRG computation
# ------------------------------------------------------------
def compute_rrg(weekly, benchmark, lookback, momentum, tail):
    rs = weekly.div(weekly[benchmark], axis=0)
    rs_ratio = np.log(rs).rolling(lookback).mean()
    rs_mom = rs_ratio.diff(momentum)

    rs_ratio_z = zscore(rs_ratio)
    rs_mom_z = zscore(rs_mom)

    tails = {}
    for sym in rs_ratio_z.columns:
        if sym == benchmark:
            continue
        x = rs_ratio_z[sym].iloc[-tail:]
        y = rs_mom_z[sym].iloc[-tail:]
        valid = (~x.isna()) & (~y.isna())
        if valid.sum() < 3:
            continue
        tails[sym] = {
            "x": x[valid],
            "y": y[valid],
            "x_last": x[valid].iloc[-1],
            "y_last": y[valid].iloc[-1],
        }

    if not tails:
        raise ValueError("Not enough data to build RRG.")
    return tails

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
def make_rrg_figure(tails, names):
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    # Quadrant shading
    fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=3,
                  fillcolor="rgba(0,200,0,0.07)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-3, y0=0, x1=0, y1=3,
                  fillcolor="rgba(0,0,200,0.05)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=-3, y0=-3, x1=0, y1=0,
                  fillcolor="rgba(200,0,0,0.06)", layer="below", line_width=0)
    fig.add_shape(type="rect", x0=0, y0=-3, x1=3, y1=0,
                  fillcolor="rgba(200,200,0,0.06)", layer="below", line_width=0)

    # Axes
    fig.add_vline(x=0, line_width=1, line_color="gray")
    fig.add_hline(y=0, line_width=1, line_color="gray")

    for i, (sym, d) in enumerate(tails.items()):
        color = colors[i % len(colors)]
        name = names.get(sym, sym)

        # Tail
        fig.add_trace(go.Scatter(
            x=d["x"][:-1],
            y=d["y"][:-1],
            mode="lines+markers",
            line=dict(color=color, width=1.5),
            marker=dict(size=4),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Head (VERY visible)
        fig.add_trace(go.Scatter(
            x=[d["x_last"]],
            y=[d["y_last"]],
            mode="markers",
            name=name,
            marker=dict(
                size=14,
                symbol="diamond",
                color=color,
                line=dict(color="black", width=2),
            ),
            hovertemplate=(
                f"<b>{name}</b><br>"
                "RS-Ratio: %{x:.2f}<br>"
                "RS-Momentum: %{y:.2f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis_title="RS-Ratio (standardized)",
        yaxis_title="RS-Momentum (standardized)",
        template="plotly_white",
        legend_title="Most recent point",
        margin=dict(l=40, r=220, t=60, b=40),
    )
    return fig

# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.title("Relative Rotation Graph (RRG)")
st.caption("Sector, theme, commodity, and country rotation vs benchmark.")

with st.sidebar:
    universe = st.selectbox("Universe", list(UNIVERSES.keys()))
    benchmark = st.selectbox("Benchmark", BENCHMARKS)
    history = st.slider("History (years)", 1, 5, 3)
    lookback = st.slider("Lookback (weeks)", 20, 78, 52)
    momentum = st.slider("Momentum (weeks)", 5, 26, 13)
    tail = st.slider("Tail length (weeks)", 5, 26, 13)

    labels = list(UNIVERSES[universe].keys())
    selected = st.multiselect("Choose ETFs", labels, default=labels)

mapping = {UNIVERSES[universe][k]: k for k in selected}
tickers = list(mapping.keys()) + [benchmark]

prices = download_prices(tickers, history)
weekly = to_weekly(prices)

tails = compute_rrg(weekly, benchmark, lookback, momentum, tail)
fig = make_rrg_figure(tails, mapping)

st.plotly_chart(fig, use_container_width=True)
