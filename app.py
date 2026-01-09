# app.py
# Streamlit Relative Rotation Graph (RRG) dashboard
# - Universes: SPDR Sectors, Themes, Commodities, Countries
# - Extra tickers via UI
# - Daily vs Weekly toggle (Daily=1y history, Weekly=3y history)
# - Tail + clearly visible HEAD marker (diamond w/ black outline)
# - Summary tables: Top 3 Leading + Top 3 Improving
# - Interpreted columns: RS-Ratio state, Momentum state, Direction arrow, Rotation Speed bucket (percentile-based)
# - Robust Yahoo Finance download handling (Close vs Adj Close, single vs multi-ticker)

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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
# Helper: Yahoo Finance download
# ----------------------------

@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_prices(
    tickers: Tuple[str, ...],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by date with one column per ticker (Close preferred, fallback Adj Close).
    Handles:
      - single ticker (columns like ['Open','High',...])
      - multi-ticker (MultiIndex columns)
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        list(tickers),
        start=start,
        end=end,
        progress=False,
        group_by="column",
        auto_adjust=False,  # we explicitly choose Close vs Adj Close
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # Multi-ticker => MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer Close, fallback Adj Close
        if ("Close" in df.columns.get_level_values(0)):
            close = df["Close"].copy()
        elif ("Adj Close" in df.columns.get_level_values(0)):
            close = df["Adj Close"].copy()
        else:
            # Return empty w/ explicit error upstream
            return pd.DataFrame()

        # Ensure all requested tickers exist as columns
        close = close.reindex(columns=list(tickers))
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()
        return close

    # Single ticker => flat columns
    cols = [c for c in df.columns]
    if "Close" in cols:
        close = df[["Close"]].rename(columns={"Close": tickers[0]}).copy()
    elif "Adj Close" in cols:
        close = df[["Adj Close"]].rename(columns={"Adj Close": tickers[0]}).copy()
    else:
        return pd.DataFrame()

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """Weekly (Friday) close. Works for multi-column price panels."""
    if prices.empty:
        return prices
    w = prices.resample("W-FRI").last()
    return w.dropna(how="all")


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
    Build standardized RS-Ratio and RS-Momentum series for each asset.
    A pragmatic proxy:
      RS = asset / benchmark
      RS-Ratio = zscore(rolling mean of RS over lookback)
      RS-Momentum = zscore(change in RS-Ratio over momentum)
    Returns:
      rs_ratio_z (index aligned)
      rs_mom_z (index aligned)
    """
    # Align
    df = asset_prices.join(bench_prices.rename("BENCH"), how="inner")
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    bench = df["BENCH"]
    assets = df.drop(columns=["BENCH"])

    rs = assets.divide(bench, axis=0)

    # Smooth RS with lookback
    rs_smooth = rs.rolling(lookback, min_periods=max(10, lookback // 3)).mean()

    # Standardize per asset using rolling stats (52 periods baseline)
    rs_ratio_z = rs_smooth.apply(_zscore)

    # Momentum: change in RS-Ratio over momentum window, then zscore
    rs_change = rs_ratio_z.diff(momentum)
    rs_mom_z = rs_change.apply(_zscore)

    return rs_ratio_z, rs_mom_z


# ----------------------------
# Interpretation layer
# ----------------------------

def state_from_value(v: float, flat_band: float = 0.25) -> str:
    """Map a z-scored value to a readable state."""
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
    """
    8-direction arrow by angle of last step in RS-Ratio/Momentum plane.
    Uses Unicode arrows for portability.
    """
    if pd.isna(dx) or pd.isna(dy) or (dx == 0 and dy == 0):
        return "•"

    ang = math.degrees(math.atan2(dy, dx))  # -180..180, 0 = east
    # Map to 8 bins, centered on: E, NE, N, NW, W, SW, S, SE
    # boundaries every 45 degrees
    bins = [
        ( -22.5,  22.5, "→"),
        (  22.5,  67.5, "↗"),
        (  67.5, 112.5, "↑"),
        ( 112.5, 157.5, "↖"),
        ( 157.5, 180.0, "←"),
        (-180.0,-157.5, "←"),
        (-157.5,-112.5, "↙"),
        (-112.5, -67.5, "↓"),
        ( -67.5, -22.5, "↘"),
    ]
    for lo, hi, a in bins:
        if lo <= ang < hi:
            return a
    return "•"


def speed_bucket(
    speeds: pd.Series,
    symbol_speed: float,
) -> Tuple[str, float]:
    """
    4 buckets using percentiles across currently-selected symbols.
    Returns (label, percentile).
      0-25: Slow
      25-60: Medium
      60-85: Fast
      85-100: Hot/Climactic
    """
    speeds = speeds.dropna()
    if speeds.empty or pd.isna(symbol_speed):
        return "N/A", np.nan

    pct = (speeds.rank(pct=True).loc[speeds.index].median())  # not used directly
    # Use empirical percentile of symbol_speed vs distribution
    pct = float((speeds < symbol_speed).mean())  # 0..1

    if pct < 0.25:
        return "Slow", pct
    if pct < 0.60:
        return "Medium", pct
    if pct < 0.85:
        return "Fast", pct
    return "Hot/Climactic", pct


# ----------------------------
# Plot: RRG (with clearly-visible head)
# ----------------------------

def make_rrg_figure(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    tail_len: int,
    title: str,
) -> go.Figure:
    """
    Tail connects to head:
      - Tail trace includes ALL points including the most recent point
      - Head trace overlays a larger diamond marker on the most recent point
    """
    fig = go.Figure()

    if rs_ratio_z.empty or rs_mom_z.empty:
        fig.update_layout(
            title=title,
            height=520,
            margin=dict(l=30, r=30, t=70, b=40),
        )
        return fig

    # Quadrant shading (light)
    # Use fixed symmetric bounds that fit most z-score situations.
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -3.0, 3.0

    fig.add_shape(type="rect", x0=x_min, y0=0, x1=0, y1=y_max,
                  fillcolor="rgba(120,150,255,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=0, x1=x_max, y1=y_max,
                  fillcolor="rgba(120,255,150,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=0, y1=0,
                  fillcolor="rgba(255,120,120,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=y_min, x1=x_max, y1=0,
                  fillcolor="rgba(255,210,120,0.08)", line_width=0, layer="below")

    # Axes crosshairs
    fig.add_shape(type="line", x0=x_min, y0=0, x1=x_max, y1=0,
                  line=dict(color="rgba(0,0,0,0.35)", width=1))
    fig.add_shape(type="line", x0=0, y0=y_min, x1=0, y1=y_max,
                  line=dict(color="rgba(0,0,0,0.35)", width=1))

    # Quadrant labels (corners)
    fig.add_annotation(x=x_min + 0.2, y=y_max - 0.2, text="Improving",
                       showarrow=False, font=dict(size=12, color="blue"))
    fig.add_annotation(x=x_max - 0.2, y=y_max - 0.2, text="Leading",
                       showarrow=False, xanchor="right",
                       font=dict(size=12, color="green"))
    fig.add_annotation(x=x_min + 0.2, y=y_min + 0.2, text="Lagging",
                       showarrow=False, font=dict(size=12, color="red"))
    fig.add_annotation(x=x_max - 0.2, y=y_min + 0.2, text="Weakening",
                       showarrow=False, xanchor="right",
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

        # Tail trace (connected line + small markers)
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

        # Head marker (most recent point) - large diamond with black outline
        xh = float(xt.iloc[-1])
        yh = float(yt.iloc[-1])
        fig.add_trace(
            go.Scatter(
                x=[xh],
                y=[yh],
                mode="markers",
                name=f"{sym} (latest)",
                marker=dict(
                    size=14,
                    symbol="diamond",
                    line=dict(width=2, color="black"),
                ),
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
        legend=dict(title="Most recent point"),
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


def build_summary_tables(
    rs_ratio_z: pd.DataFrame,
    rs_mom_z: pd.DataFrame,
    descriptions: Dict[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (top_leading_df, top_improving_df) with interpreted columns only.
    """
    symbols = [c for c in rs_ratio_z.columns if c in rs_mom_z.columns]
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    latest_rows: List[RRGRow] = []

    # compute last-step speeds for percentile bucketing
    speeds = {}
    for sym in symbols:
        s1 = rs_ratio_z[sym].dropna()
        s2 = rs_mom_z[sym].dropna()
        idx = s1.index.intersection(s2.index)
        if len(idx) < 3:
            continue
        # last step vector
        x_last, y_last = float(s1.loc[idx[-1]]), float(s2.loc[idx[-1]])
        x_prev, y_prev = float(s1.loc[idx[-2]]), float(s2.loc[idx[-2]])
        spd = math.sqrt((x_last - x_prev) ** 2 + (y_last - y_prev) ** 2)
        speeds[sym] = spd

    speeds_s = pd.Series(speeds, dtype="float64")

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

        spd = speeds.get(sym, np.nan)
        spd_label, _pct = speed_bucket(speeds_s, spd)

        q = quadrant(x_last, y_last)

        latest_rows.append(
            RRGRow(
                symbol=sym,
                description=descriptions.get(sym, sym),
                rs_state=rs_state,
                mom_state=mom_state,
                arrow=arr,
                speed_label=spd_label,
                quad=q,
            )
        )

    if not latest_rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(
        [{
            "Symbol": r.symbol,
            "Description": r.description,
            "RS-Ratio": r.rs_state,
            "Momentum": r.mom_state,
            "Direction": r.arrow,
            "Rotation Speed": r.speed_label,
            "Quadrant": r.quad,
        } for r in latest_rows]
    )

    # Top 3 Leading + Improving
    leading = df[df["Quadrant"] == "Leading"].copy()
    improving = df[df["Quadrant"] == "Improving"].copy()

    # Ranking: keep it intuitive
    # Leading: prioritize RS-Ratio then Momentum as tie-break by mapping states to scores
    score_map = {"Weakening": -1, "Flat": 0, "Improving": 1, "N/A": -99}
    leading["Score"] = leading["RS-Ratio"].map(score_map).fillna(0) + 0.5 * leading["Momentum"].map(score_map).fillna(0)
    improving["Score"] = improving["Momentum"].map(score_map).fillna(0) + 0.5 * improving["RS-Ratio"].map(score_map).fillna(0)

    top_leading = leading.sort_values(["Score", "Symbol"], ascending=[False, True]).head(3).drop(columns=["Score"], errors="ignore")
    top_improving = improving.sort_values(["Score", "Symbol"], ascending=[False, True]).head(3).drop(columns=["Score"], errors="ignore")

    # Keep only what you wanted to read quickly (like your mock)
    cols = ["Symbol", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]
    return top_leading[cols].reset_index(drop=True), top_improving[cols].reset_index(drop=True)


# ----------------------------
# Main app
# ----------------------------

def main() -> None:
    st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")

    st.title("Relative Rotation Graph (RRG)")
    st.caption("Track sector, theme, commodity, and country rotation vs a benchmark. Approximation of JdK RS-Ratio and RS-Momentum.")

    # Sidebar controls
    st.sidebar.header("RRG Settings")

    universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys()), index=0)
    universe_map = UNIVERSES[universe_name]

    benchmark = st.sidebar.selectbox("Benchmark", DEFAULT_BENCHMARKS, index=0)

    # Daily vs Weekly toggle affects both:
    # - sampling (daily series vs weekly resample)
    # - default history: Daily=1y, Weekly=3y
    timeframe = st.sidebar.radio("Timeframe", ["Daily", "Weekly"], index=1)
    default_hist_years = 1 if timeframe == "Daily" else 3

    history_years = st.sidebar.slider("History (years, daily data)", min_value=1, max_value=5, value=default_hist_years)

    lookback_weeks = st.sidebar.slider("Lookback window (weeks)", min_value=12, max_value=104, value=52, step=1)
    momentum_weeks = st.sidebar.slider("Momentum period (weeks)", min_value=4, max_value=52, value=13, step=1)
    tail_len = st.sidebar.slider("Tail length (weeks)", min_value=4, max_value=52, value=13, step=1)

    # Choose from predefined list
    default_choices = list(universe_map.keys())[: min(12, len(universe_map))]
    chosen_labels = st.sidebar.multiselect("Choose ETFs", options=list(universe_map.keys()), default=default_choices)
    chosen_tickers = [universe_map[lbl] for lbl in chosen_labels]

    extra = st.sidebar.text_input("Extra tickers (comma-separated, e.g. 'QQQ, IWM, IHI')", value="")
    extra_tickers = [t.strip().upper() for t in extra.split(",") if t.strip()]
    # Merge unique
    tickers = []
    seen = set()
    for t in chosen_tickers + extra_tickers:
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    if not tickers:
        st.warning("Select at least one ETF/ticker.")
        return

    # Build descriptions dictionary (ticker -> friendly label)
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

    # Ensure we have benchmark
    if benchmark not in prices.columns:
        st.error(f"Benchmark '{benchmark}' data missing. Try a different benchmark.")
        return

    # Drop columns that are entirely missing
    valid_cols = [c for c in prices.columns if prices[c].notna().sum() > 0]
    prices = prices[valid_cols].copy()

    # Data health: filter symbols w/ minimum points required
    if timeframe == "Weekly":
        panel = to_weekly(prices)
        # windows expressed in weeks directly
        lookback = lookback_weeks
        momentum = momentum_weeks
        required_points = lookback_weeks + momentum_weeks + tail_len + 10
    else:
        panel = prices.dropna(how="all").copy()
        # convert weeks to trading days
        lookback = int(lookback_weeks * 5)
        momentum = int(momentum_weeks * 5)
        required_points = int((lookback_weeks + momentum_weeks + tail_len) * 5 + 30)

    # Split benchmark + assets
    bench_s = panel[benchmark].dropna()
    asset_cols = [t for t in tickers if t in panel.columns and t != benchmark]

    dropped: List[str] = []
    kept: List[str] = []
    for t in asset_cols:
        n = panel[t].dropna().shape[0]
        if n < required_points:
            dropped.append(f"{t} (have {n} pts, need ≥ {required_points})")
        else:
            kept.append(t)

    if dropped:
        st.sidebar.warning("Some symbols were dropped due to insufficient history:\n\n" + "\n".join(dropped))

    if not kept:
        st.warning("Not enough data to build RRG (try shorter lookback / momentum / tail, or increase history).")
        return

    assets = panel[kept].dropna(how="all")
    bench_s = bench_s.reindex(assets.index).dropna()
    assets = assets.reindex(bench_s.index).dropna(how="all")

    if assets.empty or bench_s.empty:
        st.warning("No overlapping data between benchmark and selected ETFs.")
        return

    # Compute RRG series
    rs_ratio_z, rs_mom_z = compute_rrg_series(
        asset_prices=assets,
        bench_prices=bench_s,
        lookback=lookback,
        momentum=momentum,
    )

    # Tail length in points depends on timeframe
    if timeframe == "Weekly":
        tail_points = tail_len
        title = f"{universe_name} vs {benchmark} (Weekly)"
    else:
        tail_points = int(tail_len * 5)
        title = f"{universe_name} vs {benchmark} (Daily)"

    # Build chart
    fig = make_rrg_figure(
        rs_ratio_z=rs_ratio_z,
        rs_mom_z=rs_mom_z,
        tail_len=tail_points,
        title=title,
    )

    # Layout: chart + legend
    st.plotly_chart(fig, use_container_width=True)

    # Summary tables
    top_leading, top_improving = build_summary_tables(rs_ratio_z, rs_mom_z, descriptions)

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


if __name__ == "__main__":
    main()

