# app.py
# Streamlit Relative Rotation Graph (RRG) dashboard
# Fixes added:
# 1) Auto-zoom / expand axes so points outside the default [-3,3] range remain visible
#    - Dynamic axis bounds computed from all tail points + head points
#    - Padding applied, with an optional "Auto-zoom" toggle and manual padding slider
# 2) Adds back the “dropdown box” (expander) that shows a full snapshot table:
#    Symbol | Description | RS-Ratio | Momentum | Direction | Rotation Speed | Quadrant

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ----------------------------
# Constants / Universes
# ----------------------------

SPDR_SECTORS: Dict[str, str] = {
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
}

THEMES: Dict[str, str] = {
    "Semiconductors (SOXX)": "SOXX",
    "Semiconductors 2 (SMH)": "SMH",
    "Cybersecurity (HACK)": "HACK",
    "Cybersecurity 2 (CIBR)": "CIBR",
    "Cloud (CLOU)": "CLOU",
    "Cloud 2 (SKYY)": "SKYY",
    "Software (IGV)": "IGV",
    "Defense & Aerospace (ITA)": "ITA",
    "Defense & Aerospace 2 (XAR)": "XAR",
    "European Defense (EUAD)": "EUAD",
    "Clean Energy (ICLN)": "ICLN",
    "Solar (TAN)": "TAN",
    "Fintech / Innovation (ARKF)": "ARKF",
    "Infrastructure (PAVE)": "PAVE",
    "Digital Infrastructure (DTCR)": "DTCR",
    "Digital Infrastructure 2 (TCAI)": "TCAI",
    "Bitcoin Mining / HPC (STCE)": "STCE",
    "Bitcoin Mining / HPC 2 (WGMI)": "WGMI",
    "Home Construction (ITB)": "ITB",
    "Natural Gas (BOIL)": "BOIL",
    "Natural Gas 2 (XOP)": "XOP",
    "Robotics (ROBO)": "ROBO",
    "Robotics 2 (BOTZ)": "BOTZ",
    "Nuclear (NLR)": "NLR",
    "Nuclear 2 (NUKZ)": "NUKZ",
    "Biotech (ARKG)": "ARKG",
    "Biotech 2 (BIB)": "BIB",
    "Pharmaceutical (PPH)": "PPH",
    "Drone (JEDI)": "JEDI",
    "Drone 2 (ARKQ)": "ARKQ",
    "Brokerage (RTH)": "RTH",
    "Brokerage 2 (IAI)": "IAI",
    "Retail Shopping (XRT)": "XRT",
    "Utilities (PUI)": "PUI",
    "Space (UFO)": "UFO",
    "Regional Banking (KRE)": "KRE",
    "Banking (KBE)": "KBE",
    "Airlines (JETS)": "JETS",
    "Rare Earth (REMX)": "REMX",
    "Quantum (QTUM)": "QTUM",
    "Cannabis (MSOS)": "MSOS",
}

COMMODITIES: Dict[str, str] = {
    "Gold (RING)": "RING",
    "Gold 2 (IAU)": "IAU",
    "Silver (SIL)": "SIL",
    "Silver 2 (SLV)": "SLV",
    "Copper (COPX)": "COPX",
    "Crude Oil (USO)": "USO",
    "Bitcoin proxy (BTC-USD)": "BTC-USD",
    "Ethereum proxy (ETH-USD)": "ETH-USD",
    "Solana proxy (SOL-USD)": "SOL-USD",
}

COUNTRIES: Dict[str, str] = {
    "All World ex-US (VEU)": "VEU",
    "Emerging Mkts ex-China (EMXC)": "EMXC",
    "Brazil (EWZ)": "EWZ",
    "China (MCHI)": "MCHI",
    "China Internet (KWEB)": "KWEB",
    "Germany (EWG)": "EWG",
    "Canada (EWC)": "EWC",
    "Singapore (EWS)": "EWS",
}

UNIVERSES: Dict[str, Dict[str, str]] = {
    "SPDR Sectors": SPDR_SECTORS,
    "Themes ETFs": THEMES,
    "Commodity ETFs": COMMODITIES,
    "Country ETFs": COUNTRIES,
}

DEFAULT_BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA"]


# ----------------------------
# Yahoo Finance download (robust)
# ----------------------------

@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices(
    tickers: Tuple[str, ...],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Returns DataFrame indexed by date with one column per ticker.
    Prefers Close; falls back to Adj Close.
    Handles:
      - single ticker => flat columns
      - multi-ticker => MultiIndex columns
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        progress=False,
        group_by="column",
        auto_adjust=False,
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in set(lvl0):
            close = df["Close"].copy()
        elif "Adj Close" in set(lvl0):
            close = df["Adj Close"].copy()
        else:
            return pd.DataFrame()

        close = close.reindex(columns=list(tickers))
        close.index = pd.to_datetime(close.index)
        return close.sort_index()

    # single ticker
    cols = list(df.columns)
    t0 = tickers[0]
    if "Close" in cols:
        close = df[["Close"]].rename(columns={"Close": t0}).copy()
    elif "Adj Close" in cols:
        close = df[["Adj Close"]].rename(columns={"Adj Close": t0}).copy()
    else:
        return pd.DataFrame()

    close.index = pd.to_datetime(close.index)
    return close.sort_index()


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    return prices.resample("W-FRI").last().dropna(how="all")


# ----------------------------
# RRG math (pragmatic approximation)
# ----------------------------

def _zscore(s: pd.Series) -> pd.Series:
    mu = s.rolling(52, min_periods=20).mean()
    sd = s.rolling(52, min_periods=20).std()
    return (s - mu) / sd


def compute_rrg_series(
    asset_prices: pd.DataFrame,
    bench_prices: pd.Series,
    lookback: int,
    momentum: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Proxy approximation of JdK RS-Ratio / RS-Momentum:
      RS = asset / benchmark
      RS-Ratio = zscore(rolling mean of RS over lookback)
      RS-Momentum = zscore(diff(RS-Ratio, momentum))
    """
    df = asset_prices.join(bench_prices.rename("BENCH"), how="inner")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bench = df["BENCH"]
    assets = df.drop(columns=["BENCH"])

    rs = assets.divide(bench, axis=0)
    rs_smooth = rs.rolling(lookback, min_periods=max(10, lookback // 3)).mean()

    rs_ratio_z = rs_smooth.apply(_zscore)
    rs_mom_z = rs_ratio_z.diff(momentum).apply(_zscore)

    return rs_ratio_z, rs_mom_z


# ----------------------------
# Interpretation
# ----------------------------

def state_from_value(v: float, flat_band: float = 0.25) -> str:
    if pd.isna(v):
        return "N/A"
    if abs(v) <= flat_band:
        return "Flat"
    return "Improving" if v > 0 else "Weakening"


def quadrant(rs: float, mom: float) -> str:
    if pd.isna(rs) or pd.isna(mom):
        return "Unknown"
    if rs >= 0 and mom >= 0:
        return "Leading"
    if rs < 0 and mom >= 0:
        return "Improving"
    if rs < 0 and mom < 0:
        return "Lagging"
    return "Weakening"


def direction_arrow(dx: float, dy: float) -> str:
    if pd.isna(dx) or pd.isna(dy) or (dx == 0 and dy == 0):
        return "•"
    ang = math.degrees(math.atan2(dy, dx))
    bins = [
        (-22.5, 22.5, "→"),
        (22.5, 67.5, "↗"),
        (67.5, 112.5, "↑"),
        (112.5, 157.5, "↖"),
        (157.5, 180.0, "←"),
        (-180.0, -157.5, "←"),
        (-157.5, -112.5, "↙"),
        (-112.5, -67.5, "↓"),
        (-67.5, -22.5, "↘"),
    ]
    for lo, hi, a in bins:
        if lo <= ang < hi:
            return a
    return "•"


def speed_bucket_label(speeds: pd.Series, symbol_speed: float) -> str:
    speeds = speeds.dropna()
    if speeds.empty or pd.isna(symbol_speed):
        return "N/A"
    pct = float((speeds < symbol_speed).mean())  # 0..1
    if pct < 0.25:
        return "Slow"
    if pct < 0.60:
        return "Medium"
    if pct < 0.85:
        return "Fast"
    return "Hot/Climactic"


# ----------------------------
# Plot: RRG with dynamic axis bounds
# ----------------------------

def _compute_axis_bounds(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    tail_len: int,
    pad_frac: float = 0.10,
    min_span: float = 2.5,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes x/y axis ranges from all tail points (including heads), plus padding.
    Ensures a minimum span so chart does not over-zoom.
    """
    xs: List[float] = []
    ys: List[float] = []

    symbols = [c for c in rs_ratio_z.columns if c in rs_mom_z.columns]
    for sym in symbols:
        x = rs_ratio_z[sym].dropna()
        y = rs_mom_z[sym].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < max(3, tail_len):
            continue

        tidx = idx[-tail_len:]
        xt = rs_ratio_z.loc[tidx, sym].astype(float).values
        yt = rs_mom_z.loc[tidx, sym].astype(float).values

        xs.extend([v for v in xt if np.isfinite(v)])
        ys.extend([v for v in yt if np.isfinite(v)])

    if not xs or not ys:
        return (-3.0, 3.0), (-3.0, 3.0)

    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))

    # enforce min spans
    xspan = max(xmax - xmin, min_span)
    yspan = max(ymax - ymin, min_span)

    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    # padded spans
    xspan *= (1.0 + pad_frac)
    yspan *= (1.0 + pad_frac)

    return (xmid - xspan / 2.0, xmid + xspan / 2.0), (ymid - yspan / 2.0, ymid + yspan / 2.0)


def make_rrg_figure(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    tail_len: int,
    title: str,
    auto_zoom: bool = True,
    zoom_padding: float = 0.10,
) -> go.Figure:
    """
    Tail connects to head:
      - Tail trace includes ALL points including the most recent point
      - Head trace overlays a larger diamond marker on the most recent point
    Axis bounds:
      - If auto_zoom=True, ranges are computed from visible points with padding.
      - Otherwise, uses a fixed symmetric range around 0.
    """
    fig = go.Figure()

    if rs_ratio_z.empty or rs_mom_z.empty:
        fig.update_layout(title=title, height=520, margin=dict(l=30, r=30, t=70, b=40))
        return fig

    if auto_zoom:
        (x_min, x_max), (y_min, y_max) = _compute_axis_bounds(
            rs_ratio_z, rs_mom_z, tail_len=tail_len, pad_frac=zoom_padding
        )
    else:
        x_min, x_max = -3.0, 3.0
        y_min, y_max = -3.0, 3.0

    # Quadrant shading (light)
    fig.add_shape(type="rect", x0=x_min, y0=0, x1=0, y1=y_max,
                  fillcolor="rgba(120,150,255,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=x_max, y1=y_max,
                  fillcolor="rgba(120,255,150,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=0, y1=0,
                  fillcolor="rgba(255,120,120,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=y_min, x1=x_max, y1=0,
                  fillcolor="rgba(255,210,120,0.08)", line_width=0, layer="below")

    # Axes crosshairs at 0
    fig.add_shape(type="line", x0=x_min, y0=0, x1=x_max, y1=0,
                  line=dict(color="rgba(0,0,0,0.35)", width=1))
    fig.add_shape(type="line", x0=0, y0=y_min, x1=0, y1=y_max,
                  line=dict(color="rgba(0,0,0,0.35)", width=1))

    # Quadrant labels (corners)
    fig.add_annotation(x=x_min + 0.03 * (x_max - x_min), y=y_max - 0.05 * (y_max - y_min),
                       text="Improving", showarrow=False, font=dict(size=12, color="blue"))
    fig.add_annotation(x=x_max - 0.03 * (x_max - x_min), y=y_max - 0.05 * (y_max - y_min),
                       text="Leading", showarrow=False, xanchor="right",
                       font=dict(size=12, color="green"))
    fig.add_annotation(x=x_min + 0.03 * (x_max - x_min), y=y_min + 0.05 * (y_max - y_min),
                       text="Lagging", showarrow=False, font=dict(size=12, color="red"))
    fig.add_annotation(x=x_max - 0.03 * (x_max - x_min), y=y_min + 0.05 * (y_max - y_min),
                       text="Weakening", showarrow=False, xanchor="right",
                       font=dict(size=12, color="orange"))

    # Plot each symbol
    symbols = [c for c in rs_ratio_z.columns if c in rs_mom_z.columns]
    for sym in symbols:
        x = rs_ratio_z[sym].dropna()
        y = rs_mom_z[sym].dropna()
        idx = x.index.intersection(y.index)
        if len(idx) < max(3, tail_len):
            continue

        tail_idx = idx[-tail_len:]  # includes most recent
        xt = rs_ratio_z.loc[tail_idx, sym]
        yt = rs_mom_z.loc[tail_idx, sym]

        # Tail trace (connected)
        fig.add_trace(
            go.Scatter(
                x=xt,
                y=yt,
                mode="lines+markers",
                name=sym,
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate=(
                    f"<b>{sym}</b><br>"
                    "RS-Ratio(z): %{x:.2f}<br>"
                    "RS-Mom(z): %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

        # HEAD marker (latest) - large diamond w/ black outline
        xh = float(xt.iloc[-1])
        yh = float(yt.iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[xh],
                y=[yh],
                mode="markers",
                name=f"{sym} (latest)",
                marker=dict(size=14, symbol="diamond", line=dict(width=2, color="black")),
                hovertemplate=(
                    f"<b>{sym} (latest)</b><br>"
                    "RS-Ratio(z): %{x:.2f}<br>"
                    "RS-Mom(z): %{y:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=30, r=30, t=70, b=40),
        xaxis=dict(title="RS-Ratio (standardized)", range=[x_min, x_max], zeroline=False),
        yaxis=dict(title="RS-Momentum (standardized)", range=[y_min, y_max], zeroline=False),
    )
    return fig


# ----------------------------
# Table build
# ----------------------------

@dataclass
class RRGRow:
    symbol: str
    description: str
    rs_state: str
    mom_state: str
    arrow: str
    speed_label: str
    quad: str
    rs_val: float
    mom_val: float


def build_snapshot_table(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    descriptions: Dict[str, str],
) -> pd.DataFrame:
    """
    Full snapshot for all valid symbols. Includes interpreted columns and quadrant.
    """
    symbols = [c for c in rs_ratio_z.columns if c in rs_mom_z.columns]
    rows: List[RRGRow] = []

    # speed distribution across selected symbols (last-step speed)
    spds = {}
    for sym in symbols:
        s1 = rs_ratio_z[sym].dropna()
        s2 = rs_mom_z[sym].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 3:
            continue
        x_last, y_last = float(s1.loc[idx[-1]]), float(s2.loc[idx[-1]])
        x_prev, y_prev = float(s1.loc[idx[-2]]), float(s2.loc[idx[-2]])
        spds[sym] = math.sqrt((x_last - x_prev) ** 2 + (y_last - y_prev) ** 2)

    spds_s = pd.Series(spds, dtype="float64")

    for sym in symbols:
        s1 = rs_ratio_z[sym].dropna()
        s2 = rs_mom_z[sym].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 3:
            continue

        x_last, y_last = float(s1.loc[idx[-1]]), float(s2.loc[idx[-1]])
        x_prev, y_prev = float(s1.loc[idx[-2]]), float(s2.loc[idx[-2]])
        dx, dy = x_last - x_prev, y_last - y_prev

        rs_state = state_from_value(x_last, flat_band=0.25)
        mom_state = state_from_value(y_last, flat_band=0.25)
        arr = direction_arrow(dx, dy)

        spd_label = speed_bucket_label(spds_s, spds.get(sym, np.nan))
        q = quadrant(x_last, y_last)

        rows.append(
            RRGRow(
                symbol=sym,
                description=descriptions.get(sym, sym),
                rs_state=rs_state,
                mom_state=mom_state,
                arrow=arr,
                speed_label=spd_label,
                quad=q,
                rs_val=x_last,
                mom_val=y_last,
            )
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [{
            "Symbol": r.symbol,
            "Description": r.description,
            "RS-Ratio": r.rs_state,
            "Momentum": r.mom_state,
            "Direction": r.arrow,
            "Rotation Speed": r.speed_label,
            "Quadrant": r.quad,
            "_RS": r.rs_val,
            "_MOM": r.mom_val,
        } for r in rows]
    )

    # Sort: quadrant priority then RS then MOM
    quad_order = {"Leading": 0, "Improving": 1, "Weakening": 2, "Lagging": 3, "Unknown": 4}
    df["_QO"] = df["Quadrant"].map(quad_order).fillna(9)
    df = df.sort_values(["_QO", "_RS", "_MOM"], ascending=[True, False, False])

    return df.drop(columns=["_QO"])


def top3_tables(snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["Symbol", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]
    if snapshot.empty:
        return pd.DataFrame(), pd.DataFrame()

    leading = snapshot[snapshot["Quadrant"] == "Leading"].copy()
    improving = snapshot[snapshot["Quadrant"] == "Improving"].copy()

    # Use numeric under-the-hood values to rank
    leading = leading.sort_values(["_RS", "_MOM"], ascending=[False, False]).head(3)
    improving = improving.sort_values(["_MOM", "_RS"], ascending=[False, False]).head(3)

    return leading[cols].reset_index(drop=True), improving[cols].reset_index(drop=True)


# ----------------------------
# Main app
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")

    st.title("Relative Rotation Graph (RRG)")
    st.caption(
        "Track sector, theme, commodity, and country rotation vs a benchmark. "
        "Approximation of JdK RS-Ratio and RS-Momentum."
    )

    st.sidebar.header("RRG Settings")

    universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys()), index=0)
    universe_map = UNIVERSES[universe_name]

    benchmark = st.sidebar.selectbox("Benchmark", DEFAULT_BENCHMARKS, index=0)

    timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"], index=1)
    default_hist_years = 1 if timeframe == "Daily" else 3
    history_years = st.sidebar.slider("History (years, daily data)", 1, 5, default_hist_years)

    lookback_weeks = st.sidebar.slider("Lookback window (weeks)", 12, 104, 52, 1)
    momentum_weeks = st.sidebar.slider("Momentum period (weeks)", 4, 52, 13, 1)
    tail_len_weeks = st.sidebar.slider("Tail length (weeks)", 4, 52, 13, 1)

    # Auto-zoom settings
    st.sidebar.subheader("Chart zoom")
    auto_zoom = st.sidebar.checkbox("Auto-zoom to fit points", value=True)
    zoom_padding = st.sidebar.slider("Zoom padding", 0.05, 0.35, 0.10, 0.01)

    # Choose from predefined list
    default_choices = list(universe_map.keys())[: min(12, len(universe_map))]
    chosen_labels = st.sidebar.multiselect("Choose ETFs", options=list(universe_map.keys()), default=default_choices)
    chosen_tickers = [universe_map[lbl] for lbl in chosen_labels]

    extra = st.sidebar.text_input("Extra tickers (comma-separated, e.g. 'QQQ, IWM, IHI')", value="")
    extra_tickers = [t.strip().upper() for t in extra.split(",") if t.strip()]

    # Merge unique tickers
    tickers: List[str] = []
    seen = set()
    for t in chosen_tickers + extra_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        st.warning("Select at least one ETF/ticker.")
        return

    # Descriptions mapping ticker -> label
    descriptions = {v: k for k, v in universe_map.items()}
    for t in extra_tickers:
        descriptions.setdefault(t, t)

    # Dates
    end = date.today() + timedelta(days=1)
    start = date.today() - timedelta(days=int(history_years * 365.25) + 7)

    # Download
    all_tickers = tuple(sorted(set(tickers + [benchmark])))
    prices = download_prices(all_tickers, start=start, end=end)

    if prices.empty:
        st.error("No data returned from Yahoo Finance for the selected tickers.")
        return

    if benchmark not in prices.columns:
        st.error(f"Benchmark '{benchmark}' data missing. Try a different benchmark.")
        return

    # Remove dead columns
    prices = prices[[c for c in prices.columns if prices[c].notna().sum() > 0]].copy()

    # Timeframe transforms + required points
    if timeframe == "Weekly":
        panel = to_weekly(prices)
        lookback = lookback_weeks
        momentum = momentum_weeks
        tail_points = tail_len_weeks
        required_points = lookback_weeks + momentum_weeks + tail_len_weeks + 10
        chart_title = f"{universe_name} vs {benchmark} (Weekly)"
    else:
        panel = prices.dropna(how="all").copy()
        lookback = int(lookback_weeks * 5)
        momentum = int(momentum_weeks * 5)
        tail_points = int(tail_len_weeks * 5)
        required_points = int((lookback_weeks + momentum_weeks + tail_len_weeks) * 5 + 30)
        chart_title = f"{universe_name} vs {benchmark} (Daily)"

    bench_s = panel[benchmark].dropna()

    asset_cols = [t for t in tickers if t in panel.columns and t != benchmark]

    dropped: List[str] = []
    kept: List[str] = []
    for t in asset_cols:
        n = int(panel[t].dropna().shape[0])
        if n < required_points:
            dropped.append(f"{t} (have {n} pts, need ≥ {required_points})")
        else:
            kept.append(t)

    if dropped:
        st.sidebar.warning("Some symbols were dropped due to insufficient history:\n\n" + "\n".join(dropped))

    if not kept:
        st.warning("Not enough data to build RRG (try shorter lookback/momentum/tail or increase history).")
        return

    assets = panel[kept].dropna(how="all")
    bench_s = bench_s.reindex(assets.index).dropna()
    assets = assets.reindex(bench_s.index).dropna(how="all")

    if assets.empty or bench_s.empty:
        st.warning("No overlapping data between benchmark and selected ETFs.")
        return

    rs_ratio_z, rs_mom_z = compute_rrg_series(assets, bench_s, lookback=lookback, momentum=momentum)

    fig = make_rrg_figure(
        rs_ratio_z=rs_ratio_z,
        rs_mom_z=rs_mom_z,
        tail_len=tail_points,
        title=chart_title,
        auto_zoom=auto_zoom,
        zoom_padding=zoom_padding,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Snapshot table (full universe selection)
    snapshot = build_snapshot_table(rs_ratio_z, rs_mom_z, descriptions)
    if snapshot.empty:
        st.warning("Could not compute snapshot table (insufficient valid data after alignment).")
        return

    # Top 3 tables
    top_leading, top_improving = top3_tables(snapshot)

    st.subheader("Latest RRG Snapshot (interpreted)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 3 Leading**")
        if top_leading.empty:
            st.info("No symbols currently in the Leading quadrant.")
        else:
            st.dataframe(top_leading, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Top 3 Improving**")
        if top_improving.empty:
            st.info("No symbols currently in the Improving quadrant.")
        else:
            st.dataframe(top_improving, use_container_width=True, hide_index=True)

    # “Dropdown box” (expander) with full table
    st.markdown("---")
    with st.expander("Universe Snapshot Table (all selected symbols)", expanded=False):
        # Present a clean, readable table (no hidden numeric columns)
        show_cols = ["Symbol", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant"]
        st.dataframe(snapshot[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
