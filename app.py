import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# ============================================================
# Universes (predefined lists)
# ============================================================

UNIVERSES = {
    "SPDR Sectors": {
        "etfs": {
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
    },
    "Themes ETFs": {
        "etfs": {
            "Semiconductors (SOXX)": "SOXX",
            "Semiconductors 2 (SMH)": "SMH",
            "Cybersecurity (CIBR)": "CIBR",
            "Cybersecurity 2 (HACK)": "HACK",
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
            "Space (UFO)": "UFO",
            "Regional Banking (KRE)": "KRE",
            "Banking (KBE)": "KBE",
            "Airlines (JETS)": "JETS",
            "Rare Earth (REMX)": "REMX",
            "Quantum (QTUM)": "QTUM",
            "Cannabis (MSOS)": "MSOS",
        }
    },
    "Commodity ETFs": {
        "etfs": {
            "Gold (RING)": "RING",
            "Gold 2 (IAU)": "IAU",
            "Silver (SIL)": "SIL",
            "Silver 2 (SLV)": "SLV",
            "Copper (COPX)": "COPX",
            "Oil (USO)": "USO",
            "Bitcoin (BTC-USD)": "BTC-USD",
            "Ethereum (ETH-USD)": "ETH-USD",
            "Solana (SOL-USD)": "SOL-USD",
        }
    },
    "Country ETFs": {
        "etfs": {
            "All World ex-US (VEU)": "VEU",
            "Emerging Mkts ex-China (EMXC)": "EMXC",
            "China (MCHI)": "MCHI",
            "China Internet (KWEB)": "KWEB",
            "Brazil (EWZ)": "EWZ",
            "India (INDA)": "INDA",
            "Germany (EWG)": "EWG",
            "Canada (EWC)": "EWC",
            "Singapore (EWS)": "EWS",
        }
    },
}

DEFAULT_BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA"]


# ============================================================
# Data download (robust Close/Adj Close parsing)
# ============================================================

def _normalize_downloaded_close(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    yfinance can return:
      - single-index columns (e.g., ['Open','High','Low','Close',...]) for 1 ticker
      - multi-index columns (Field, Ticker) for multiple tickers
      - sometimes only Adj Close
    This function returns a clean DataFrame: index=Datetime, columns=tickers, values=close.
    """
    if df is None or df.empty:
        raise ValueError("No price data returned from Yahoo Finance (empty dataframe).")

    # If one ticker, yfinance often returns columns like ['Open','High','Low','Close',...]
    if not isinstance(df.columns, pd.MultiIndex):
        # Single ticker case
        # Try Close then Adj Close
        if "Close" in df.columns:
            close = df[["Close"]].copy()
        elif "Adj Close" in df.columns:
            close = df[["Adj Close"]].copy()
        else:
            raise ValueError(f"Expected 'Close' or 'Adj Close' columns, got {list(df.columns)}")

        close.columns = [tickers[0]]
        return close.dropna()

    # Multi ticker case: columns are MultiIndex (Field, Ticker) or (Ticker, Field) depending on params
    # Detect which level has fields like 'Close'
    lvl0 = set(df.columns.get_level_values(0))
    lvl1 = set(df.columns.get_level_values(1))

    if "Close" in lvl0 or "Adj Close" in lvl0:
        # (Field, Ticker)
        if "Close" in lvl0:
            close = df["Close"].copy()
        elif "Adj Close" in lvl0:
            close = df["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close/Adj Close in multi-index level 0.")
    elif "Close" in lvl1 or "Adj Close" in lvl1:
        # (Ticker, Field)
        if "Close" in lvl1:
            close = df.xs("Close", level=1, axis=1).copy()
        elif "Adj Close" in lvl1:
            close = df.xs("Adj Close", level=1, axis=1).copy()
        else:
            raise ValueError("Could not find Close/Adj Close in multi-index level 1.")
    else:
        raise ValueError(f"Expected 'Close' or 'Adj Close' columns, got multi-index: {df.columns.names}")

    # Ensure we have only requested tickers (some may be missing)
    close = close.loc[:, [c for c in close.columns if c in tickers]].copy()
    close = close.sort_index()
    return close.dropna(how="all")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def download_daily_close(tickers: list[str], history_years: int) -> pd.DataFrame:
    end = datetime.utcnow()
    start = end - timedelta(days=int(history_years * 365.25) + 10)

    df = yf.download(
        tickers=tickers,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    close = _normalize_downloaded_close(df, tickers)
    close.index = pd.to_datetime(close.index)
    close = close[~close.index.duplicated(keep="last")]
    return close


def to_weekly_last(close_daily: pd.DataFrame) -> pd.DataFrame:
    """Weekly series using last close of each week (Friday-based)."""
    wk = close_daily.resample("W-FRI").last()
    return wk.dropna(how="all")


# ============================================================
# RRG math
# ============================================================

def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score across columns per date."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _compute_rrg_series(
    prices: pd.DataFrame,
    benchmark: str,
    lookback: int,
    momentum: int,
    tail_len: int,
) -> tuple[dict, dict]:
    """
    prices: columns include assets + benchmark
    lookback/momentum/tail_len are in "bars" of the provided timeframe.
    """
    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark {benchmark} not in price data columns.")

    # Require sufficient data
    needed = max(lookback + momentum + tail_len, lookback + 10)
    if len(prices) < needed:
        raise ValueError(f"Not enough data: have {len(prices)} bars, need ≥ {needed} bars.")

    # Relative price vs benchmark
    rel = prices.div(prices[benchmark], axis=0)

    # RS-Ratio approximation: relative return over lookback
    rs_ratio_raw = rel / rel.shift(lookback) - 1.0

    # RS-Momentum approximation: change in RS-Ratio over momentum
    rs_mom_raw = rs_ratio_raw - rs_ratio_raw.shift(momentum)

    # Standardize cross-sectionally (JdK-style spirit)
    rs_ratio_z = _cross_sectional_zscore(rs_ratio_raw)
    rs_mom_z = _cross_sectional_zscore(rs_mom_raw)

    # Tail window
    rs_ratio_tail = rs_ratio_z.iloc[-tail_len:]
    rs_mom_tail = rs_mom_z.iloc[-tail_len:]

    assets = [c for c in prices.columns if c != benchmark]
    rrg_tails = {}
    dropped = []

    for sym in assets:
        x = rs_ratio_tail[sym].copy()
        y = rs_mom_tail[sym].copy()
        valid = (~x.isna()) & (~y.isna())
        x = x[valid]
        y = y[valid]
        if len(x) < 2:
            dropped.append(sym)
            continue

        rrg_tails[sym] = {
            "x_series": x,
            "y_series": y,
            "x_last": float(x.iloc[-1]),
            "y_last": float(y.iloc[-1]),
        }

    meta = {"dropped_symbols": dropped, "needed_bars": needed, "have_bars": len(prices)}
    return rrg_tails, meta


# ============================================================
# Velocity, arrows, interpretation
# ============================================================

def _arrow_8(dx: float, dy: float) -> str:
    """
    8-direction arrow based on vector angle.
    """
    angle = math.degrees(math.atan2(dy, dx))  # -180..180
    # Map to nearest 45 degrees
    dirs = [
        ("→", -22.5, 22.5),
        ("↗", 22.5, 67.5),
        ("↑", 67.5, 112.5),
        ("↖", 112.5, 157.5),
        ("←", 157.5, 180.0),
        ("←", -180.0, -157.5),
        ("↙", -157.5, -112.5),
        ("↓", -112.5, -67.5),
        ("↘", -67.5, -22.5),
    ]
    for sym, lo, hi in dirs:
        if lo <= angle < hi:
            return sym
    return "•"


def _trend_label(d: float, deadband: float = 0.05) -> str:
    if d > deadband:
        return "Improving"
    if d < -deadband:
        return "Weakening"
    return "Flat"


def _speed_bucket_from_percentile(p: float) -> str:
    # 4 buckets: Slow / Medium / Fast / Hot/Climactic
    if p <= 50:
        return "Slow"
    if p <= 80:
        return "Medium"
    if p <= 95:
        return "Fast"
    return "Hot/Climactic"


def build_summary_tables(
    rrg_tails: dict,
    display_names: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - full table (all symbols)
      - top3 leading
      - top3 improving
    """
    rows = []
    for sym, tail in rrg_tails.items():
        x = tail["x_series"]
        y = tail["y_series"]
        # Velocity from last step
        dx = float(x.iloc[-1] - x.iloc[-2])
        dy = float(y.iloc[-1] - y.iloc[-2])
        speed = float(math.sqrt(dx * dx + dy * dy))
        arrow = _arrow_8(dx, dy)

        rows.append(
            {
                "Sector": sym,
                "Description": display_names.get(sym, sym),
                "RS-Ratio": _trend_label(dx),
                "Momentum": _trend_label(dy),
                "Direction": arrow,
                "SpeedRaw": speed,
                "x_last": float(x.iloc[-1]),
                "y_last": float(y.iloc[-1]),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df, df

    # Percentile rank of speed among selected
    df["SpeedPct"] = df["SpeedRaw"].rank(pct=True) * 100.0
    df["Rotation Speed"] = df["SpeedPct"].apply(_speed_bucket_from_percentile)

    # Final display columns
    out = df[["Sector", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "SpeedPct", "x_last", "y_last"]].copy()

    # Quadrants based on last point
    leading = out[(out["x_last"] > 0) & (out["y_last"] > 0)].copy()
    improving = out[(out["x_last"] < 0) & (out["y_last"] > 0)].copy()

    # Sort for top 3
    leading = leading.sort_values(["x_last", "y_last", "SpeedPct"], ascending=[False, False, False]).head(3)
    improving = improving.sort_values(["x_last", "y_last", "SpeedPct"], ascending=[False, False, False]).head(3)

    # Pretty tables (drop internals)
    out_pretty = out[["Sector", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]].copy()
    leading_pretty = leading[["Sector", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]].copy()
    improving_pretty = improving[["Sector", "Description", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]].copy()

    return out_pretty, leading_pretty, improving_pretty


# ============================================================
# Plotting
# ============================================================

def make_rrg_figure(rrg_tails: dict, display_names: dict) -> go.Figure:
    if not rrg_tails:
        raise ValueError("No symbols with valid RRG data.")

    palette = px.colors.qualitative.Plotly

    all_x = np.concatenate([v["x_series"].values for v in rrg_tails.values()])
    all_y = np.concatenate([v["y_series"].values for v in rrg_tails.values()])

    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())

    pad_x = max(0.5, (x_max - x_min) * 0.15)
    pad_y = max(0.5, (y_max - y_min) * 0.15)

    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    # Ensure axes include 0
    x_min = min(x_min, -0.5)
    x_max = max(x_max, 0.5)
    y_min = min(y_min, -0.5)
    y_max = max(y_max, 0.5)

    fig = go.Figure()

    # Quadrant shading
    fig.add_shape(type="rect", x0=0, y0=0, x1=x_max, y1=y_max, fillcolor="rgba(0,200,0,0.07)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=0, x1=0, y1=y_max, fillcolor="rgba(0,0,200,0.05)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=0, y1=0, fillcolor="rgba(200,0,0,0.06)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0, y0=y_min, x1=x_max, y1=0, fillcolor="rgba(200,200,0,0.06)", line=dict(width=0), layer="below")

    # Axis lines
    fig.add_shape(type="line", x0=0, x1=0, y0=y_min, y1=y_max, line=dict(color="gray", width=1))
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, line=dict(color="gray", width=1))

    # Corner labels (clearer)
    fig.add_annotation(x=x_max * 0.92, y=y_max * 0.92, text="Leading", showarrow=False, font=dict(color="green", size=12))
    fig.add_annotation(x=x_min * 0.92, y=y_max * 0.92, text="Improving", showarrow=False, font=dict(color="blue", size=12))
    fig.add_annotation(x=x_min * 0.92, y=y_min * 0.92, text="Lagging", showarrow=False, font=dict(color="red", size=12))
    fig.add_annotation(x=x_max * 0.92, y=y_min * 0.92, text="Weakening", showarrow=False, font=dict(color="orange", size=12))

    for idx, (sym, tail) in enumerate(rrg_tails.items()):
        x = tail["x_series"]
        y = tail["y_series"]
        name = display_names.get(sym, sym)
        color = palette[idx % len(palette)]

        # Tail (all but last)
        if len(x) > 1:
            fig.add_trace(
                go.Scatter(
                    x=x.iloc[:-1],
                    y=y.iloc[:-1],
                    mode="lines+markers",
                    name=name,
                    legendgroup=sym,
                    showlegend=False,
                    line=dict(width=1.5, color=color),
                    marker=dict(size=4, color=color),
                    hovertemplate=f"{name}<br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
                )
            )

        # Head (last point) - make it very obvious
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[-1]],
                y=[y.iloc[-1]],
                mode="markers",
                name=name,
                legendgroup=sym,
                showlegend=True,
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(width=2, color="black"),
                    symbol="diamond",
                ),
                hovertemplate=f"{name} (latest)<br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="RS-Ratio (standardized)",
        yaxis_title="RS-Momentum (standardized)",
        xaxis=dict(range=[x_min, x_max], zeroline=False),
        yaxis=dict(range=[y_min, y_max], zeroline=False),
        legend=dict(
            title="Most recent point (diamond marker)",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=40, r=260, t=60, b=40),
        template="plotly_white",
    )

    return fig


# ============================================================
# Streamlit App
# ============================================================

def main():
    st.set_page_config(page_title="RRG", layout="wide")

    st.title("Relative Rotation Graph (RRG)")
    st.caption("Sector, theme, commodity, and country rotation vs a benchmark. Weekly view approximates institutional rotation; Daily view is more tactical.")

    # Sidebar settings
    with st.sidebar:
        st.header("RRG Settings")

        universe_name = st.selectbox("Universe", list(UNIVERSES.keys()), index=0)
        uni = UNIVERSES[universe_name]["etfs"]

        benchmark = st.selectbox("Benchmark", DEFAULT_BENCHMARKS, index=0)

        timeframe = st.radio(
            "Timeframe",
            ["Daily", "Weekly"],
            index=1,
            help="Daily = tactical (more granular). Weekly = institutional (smoother).",
        )

        # Defaults requested
        default_history = 1 if timeframe == "Daily" else 3

        history_years = st.slider(
            "History (years, daily data)",
            min_value=1,
            max_value=10,
            value=default_history,
            help="Daily default is 1 year; Weekly default is 3 years.",
        )

        # Choose ETFs from universe
        default_labels = list(uni.keys())
        chosen_labels = st.multiselect(
            "Choose ETFs",
            options=default_labels,
            default=default_labels,
            help="Add/remove items from this universe.",
        )

        extra_raw = st.text_input("Extra tickers (comma-separated, e.g. 'QQQ, IWM, HII')").strip()

        lookback_weeks = st.slider("Lookback window (weeks)", min_value=20, max_value=78, value=52)
        momentum_weeks = st.slider("Momentum period (weeks)", min_value=5, max_value=26, value=13)
        tail_length_weeks = st.slider("Tail length (weeks)", min_value=5, max_value=26, value=13)

    # Build mapping display -> ticker
    display_to_sym = {label: uni[label] for label in chosen_labels}

    if extra_raw:
        for tok in extra_raw.split(","):
            sym = tok.strip().upper()
            if sym:
                display_to_sym[sym] = sym

    if not display_to_sym:
        st.warning("Please select at least one ETF.")
        return

    # tickers list includes benchmark
    tickers = list(display_to_sym.values())
    if benchmark not in tickers:
        tickers.append(benchmark)

    # Download daily
    try:
        daily_close = download_daily_close(tickers, history_years)
    except Exception as e:
        st.error(f"Error downloading data from Yahoo Finance: {e}")
        return

    # Build series for RRG based on timeframe
    if timeframe == "Weekly":
        prices = to_weekly_last(daily_close)
        lookback = lookback_weeks
        momentum = momentum_weeks
        tail_len = tail_length_weeks
        needed_label = "weeks"
    else:
        # Daily bars: convert weeks -> trading days (approx 5/day per week)
        prices = daily_close.copy()
        lookback = int(lookback_weeks * 5)
        momentum = int(momentum_weeks * 5)
        tail_len = int(tail_length_weeks * 5)
        needed_label = "trading days"

    # Ensure benchmark present
    if benchmark not in prices.columns:
        st.error(f"Benchmark {benchmark} has no data for the selected history window.")
        return

    # Restrict to selected assets + benchmark
    assets_symbols = [sym for sym in display_to_sym.values() if sym != benchmark]
    keep_cols = [c for c in prices.columns if c in assets_symbols + [benchmark]]
    prices = prices[keep_cols].dropna(how="all")

    # ticker -> display
    symbol_to_display = {v: k for k, v in display_to_sym.items()}

    # Compute RRG
    try:
        rrg_tails, meta = _compute_rrg_series(
            prices=prices,
            benchmark=benchmark,
            lookback=lookback,
            momentum=momentum,
            tail_len=tail_len,
        )
    except Exception as e:
        st.warning(str(e))
        return

    dropped = meta.get("dropped_symbols", [])
    if dropped:
        st.sidebar.info(
            "Some symbols were dropped due to insufficient history or missing data:\n\n"
            + ", ".join(dropped)
        )

    if not rrg_tails:
        st.warning(
            f"Not enough data to build RRG (try shorter lookback/momentum/tail length, or remove very new ETFs). "
            f"Timeframe={timeframe}, required bars are in {needed_label}."
        )
        return

    # Layout: chart + tables
    fig = make_rrg_figure(rrg_tails, symbol_to_display)

    st.plotly_chart(fig, use_container_width=True)

    # Summary tables
    st.markdown("### Rotation Summary (interpreted)")

    full_tbl, top_leading, top_improving = build_summary_tables(rrg_tails, symbol_to_display)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Top 3 Leading")
        if top_leading.empty:
            st.info("No symbols currently in the Leading quadrant.")
        else:
            st.dataframe(top_leading, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("#### Top 3 Improving")
        if top_improving.empty:
            st.info("No symbols currently in the Improving quadrant.")
        else:
            st.dataframe(top_improving, use_container_width=True, hide_index=True)

    with st.expander("Show full table (all selected symbols)", expanded=False):
        st.dataframe(full_tbl, use_container_width=True, hide_index=True)

    st.caption(
        "Interpretation notes: RS-Ratio and Momentum are labeled from the latest step (delta) with a small deadband. "
        "Direction arrow is the 8-way direction of travel. Rotation Speed uses percentile buckets across your selected symbols."
    )


if __name__ == "__main__":
    main()
