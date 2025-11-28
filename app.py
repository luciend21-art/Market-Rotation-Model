import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st


# ---------- Configuration ---------- #

# Predefined universes
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
        "Biotech (XBI)": "XBI",
        "Biotech 2 (BIB)": "BIB",
        "Pharmaceutical (PPH)": "PPH",
        "Drone (JEDI)": "JEDI",
        "Drone 2 (ARKQ)": "ARKQ",
        "Brokerage (RTH)": "RTH",
        "Brokerage 2 (IAI)": "IAI",
        "Retail Shopping (XRT)": "XRT",
        "Utilities (PUJ)": "PUJ",
        "SPAC (SPXC)": "SPXC",
        "Regional Banking (KRE)": "KRE",
        "Banking (KBE)": "KBE",
        "Airlines (JETS)": "JETS",
        "Rare Earth (REMX)": "REMX",
        "Quantum (QTUM)": "QTUM",
        "Cannabis (MSOS)": "MSOS",
    },
    "Commodity ETFs": {
        "Gold (GLD)": "GLD",
        "Gold 2 (IAU)": "IAU",
        "Silver (SLV)": "SLV",
        "Silver 2 (SIL)": "SIL",
        "Copper (CPER)": "CPER",
        "Bitcoin (BITO)": "BITO",
        "Ethereum (ETHA)": "ETHA",
        "Solana (BSOL)": "BSOL",
    },
    "Country ETFs": {
        "US (SPY)": "SPY",
        "World ex-US (VEU)": "VEU",
        "Emerging Mkts ex-China (EMXC)": "EMXC",
        "Brazil (EWZ)": "EWZ",
        "China (MCHI)": "MCHI",
        "China 2 (KWEB)": "KWEB",
        "Japan (EWJ)": "EWJ",
        "Germany (EWG)": "EWG",
        "Canada (EWC)": "EWC",
        "Singapore (EWS)": "EWS",
    },
}

DEFAULT_BENCHMARK = "SPY"


# ---------- Data utilities ---------- #

def compute_needed_years(lookback_wk: int, mom_wk: int, tail_wk: int) -> int:
    """
    Rough estimate of how many years of weekly data we need to
    support the requested lookback / momentum / tail.
    """
    needed_weeks = lookback_wk + mom_wk + tail_wk + 5  # small safety buffer
    return max(1, math.ceil(needed_weeks / 52))


@st.cache_data(show_spinner=False)
def download_daily(tickers, years: int) -> pd.DataFrame:
    end = datetime.today()
    start = end - timedelta(days=years * 365)

    data = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        raise Exception(f"No price data returned for tickers: {tickers}")

    # Multi-ticker result -> MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if "Close" in level0:
            close = data["Close"]
        elif "Adj Close" in level0:
            close = data["Adj Close"]
        else:
            raise Exception(
                f"Expected 'Close' or 'Adj Close' columns, got {sorted(set(level0))}"
            )
        close.columns = close.columns.astype(str)

    # Single-ticker result -> normal columns
    else:
        if "Close" in data.columns:
            close = data["Close"].to_frame()
        elif "Adj Close" in data.columns:
            close = data["Adj Close"].to_frame()
        else:
            raise Exception(
                f"Expected 'Close' or 'Adj Close' columns, got {list(data.columns)}"
            )

    # Drop tickers with no usable data at all
    close = close.dropna(axis=1, how="all")

    if close.empty:
        raise Exception(
            f"No usable 'Close' prices for tickers: {tickers} "
            "(they may be delisted or lack any history)."
        )

    return close

def resample_weekly(close_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily closes to weekly (Friday) closes."""
    weekly = close_df.resample("W-FRI").last().dropna(how="all")
    return weekly


# ---------- RRG calculations ---------- #

def make_rrg_series(
    weekly_prices: pd.DataFrame,
    benchmark: str,
    lookback_wk: int,
    mom_wk: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Compute standardized RS-Ratio and RS-Momentum series.
    Returns:
      rs_z: DataFrame of RS-Ratio (z-scored) by symbol
      mom_z: DataFrame of RS-Momentum (z-scored) by symbol
      dropped: dict of {symbol: (have_points, need_points)}
    """
    tickers = [c for c in weekly_prices.columns if c != benchmark]
    bench_series = weekly_prices[benchmark]

    # Relative strength vs benchmark
    rel = weekly_prices[tickers].div(bench_series, axis=0)

    # RS-Ratio: ratio vs lookback weeks ago
    rs = rel / rel.shift(lookback_wk)

    # RS-Momentum: change in RS-Ratio over the momentum window
    rs_mom = rs / rs.shift(mom_wk) - 1

    # Standardize (z-score) over time for each column
    rs_z = (rs - rs.mean()) / rs.std(ddof=0)
    mom_z = (rs_mom - rs_mom.mean()) / rs_mom.std(ddof=0)

    # Drop columns that don't have enough valid points near the end
    dropped = {}
    min_points = mom_wk + 5  # need at least a handful of points after momentum
    valid_cols = []
    for sym in tickers:
        non_na = mom_z[sym].dropna()
        have = len(non_na)
        if have >= min_points:
            valid_cols.append(sym)
        else:
            dropped[sym] = (have, min_points)

    if not valid_cols:
        return pd.DataFrame(), pd.DataFrame(), dropped

    rs_z = rs_z[valid_cols]
    mom_z = mom_z[valid_cols]

    return rs_z, mom_z, dropped


# ---------- Plotting ---------- #

def build_rrg_figure(
    rs_z: pd.DataFrame,
    mom_z: pd.DataFrame,
    labels: dict,
    tail_len: int,
) -> go.Figure:
    fig = go.Figure()

    # Fixed axis range for clean quadrants
    x_range = [-3, 3]
    y_range = [-3, 3]

    # Quadrant shading
    fig.update_layout(
        shapes=[
            # Leading (top-right)
            dict(
                type="rect",
                x0=0,
                x1=x_range[1],
                y0=0,
                y1=y_range[1],
                fillcolor="rgba(0, 200, 0, 0.05)",
                line_width=0,
                layer="below",
            ),
            # Weakening (bottom-right)
            dict(
                type="rect",
                x0=0,
                x1=x_range[1],
                y0=y_range[0],
                y1=0,
                fillcolor="rgba(255, 165, 0, 0.05)",
                line_width=0,
                layer="below",
            ),
            # Lagging (bottom-left)
            dict(
                type="rect",
                x0=x_range[0],
                x1=0,
                y0=y_range[0],
                y1=0,
                fillcolor="rgba(255, 0, 0, 0.04)",
                line_width=0,
                layer="below",
            ),
            # Improving (top-left)
            dict(
                type="rect",
                x0=x_range[0],
                x1=0,
                y0=0,
                y1=y_range[1],
                fillcolor="rgba(0, 0, 255, 0.04)",
                line_width=0,
                layer="below",
            ),
        ]
    )

    # Add tails
    for sym in rs_z.columns:
        x = rs_z[sym].dropna().tail(tail_len)
        y = mom_z[sym].dropna().tail(tail_len)
        if len(x) < 2 or len(y) < 2:
            continue

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=labels.get(sym, sym),
                text=[labels.get(sym, sym)] * len(x),
                hovertemplate=(
                    "%{text}<br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>"
                ),
            )
        )

    # Axis & quadrant labels
    fig.update_layout(
        xaxis=dict(
            title="RS-Ratio (standardized)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            range=x_range,
        ),
        yaxis=dict(
            title="RS-Momentum (standardized)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="gray",
            range=y_range,
        ),
        legend=dict(
            title="Tails (most recent point highlighted)",
            orientation="v",
            x=1.02,
            y=1.0,
        ),
        margin=dict(l=40, r=260, t=40, b=40),
        height=650,
    )

    # Quadrant text labels
    fig.add_annotation(x=2.3, y=1.8, text="Leading", showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=2.3, y=-1.8, text="Weakening", showarrow=False, font=dict(color="darkorange"))
    fig.add_annotation(x=-2.3, y=-1.8, text="Lagging", showarrow=False, font=dict(color="red"))
    fig.add_annotation(x=-2.3, y=1.8, text="Improving", showarrow=False, font=dict(color="blue"))

    return fig


# ---------- Streamlit app ---------- #

def main():
    st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")

    st.sidebar.header("RRG Settings")

    # Universe selection
    universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys()))
    universe = UNIVERSES[universe_name]

    # Benchmark
    all_tickers_in_uni = list(universe.values())
    default_bench = DEFAULT_BENCHMARK if DEFAULT_BENCHMARK in all_tickers_in_uni else DEFAULT_BENCHMARK
    benchmark = st.sidebar.text_input("Benchmark", value=default_bench).strip().upper()

    # ETF multiselect with free-form additions
    default_selection = list(universe.keys())
    chosen_labels = st.sidebar.multiselect(
        "Choose ETFs",
        options=list(universe.keys()),
        default=default_selection,
        help="You can add extra tickers below.",
    )

    extra_tickers_str = st.sidebar.text_input(
        "Extra tickers (comma-separated, e.g. 'QQQ, IWM, IHI')",
        value="",
    )
    extra_tickers = [t.strip().upper() for t in extra_tickers_str.split(",") if t.strip()]

    # Build ticker list & label map
    tickers = []
    label_map = {}
    for label in chosen_labels:
        sym = universe[label]
        tickers.append(sym)
        label_map[sym] = label

    for sym in extra_tickers:
        if sym and sym not in tickers:
            tickers.append(sym)
            label_map.setdefault(sym, sym)

    if benchmark not in tickers:
        tickers.append(benchmark)
        label_map.setdefault(benchmark, benchmark + " (benchmark)")

    # Sliders
    st.sidebar.markdown("---")
    history_years = st.sidebar.slider(
        "History (years, daily data)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Minimum number of years to download; app will auto-extend if math needs more data.",
    )
    lookback_wk = st.sidebar.slider(
        "Lookback window (weeks)",
        min_value=8,
        max_value=104,
        value=52,
        step=1,
        help="Lookback used for RS-Ratio calculation.",
    )
    mom_wk = st.sidebar.slider(
        "Momentum period (weeks)",
        min_value=4,
        max_value=26,
        value=13,
        step=1,
        help="Window used to smooth RS-Momentum.",
    )
    tail_len = st.sidebar.slider(
        "Tail length (weeks)",
        min_value=4,
        max_value=26,
        value=13,
        step=1,
        help="How many weekly points to show in each tail.",
    )

    st.title("Relative Rotation Graph (RRG)")
    st.caption(
        "Track sector, theme, commodity, and country rotation vs a benchmark. "
        "Approximation of JdK RS-Ratio and RS-Momentum using weekly data."
    )

    if len(tickers) == 0:
        st.warning("Please select at least one ETF.")
        return

    # Determine how much history we REALLY need
    needed_years = compute_needed_years(lookback_wk, mom_wk, tail_len)
    years_to_fetch = max(history_years, needed_years)

    if years_to_fetch > history_years:
        st.info(
            f"Based on lookback ({lookback_wk}w), momentum ({mom_wk}w), and tail ({tail_len}w), "
            f"the app automatically extended history from {history_years} to {years_to_fetch} years."
        )

    # Download & resample
    try:
        daily = download_daily(tickers, years_to_fetch)
    except Exception as e:
        st.error(f"Error downloading data from Yahoo Finance: {e}")
        return

    if daily.empty:
        st.error("No price data returned. Try different tickers or a shorter history window.")
        return

    weekly = resample_weekly(daily)

    if benchmark not in weekly.columns:
        st.error(f"Benchmark {benchmark} is missing from the downloaded data.")
        return

    # Compute RRG series
    rs_z, mom_z, dropped = make_rrg_series(weekly, benchmark, lookback_wk, mom_wk)

    if rs_z.empty or mom_z.empty:
        st.warning(
            "Not enough data to build RRG (try a shorter lookback / momentum window or remove very new ETFs)."
        )
        if dropped:
            with st.expander("Details: symbols dropped due to insufficient history"):
                for sym, (have, need) in dropped.items():
                    st.write(f"{sym}: have {have} weekly points after calculations, need ≥ {need}.")
        return

    # Build figure
    fig = build_rrg_figure(rs_z, mom_z, label_map, tail_len)
    st.plotly_chart(fig, use_container_width=True)

    # Show info about dropped symbols, if any
    if dropped:
        st.warning("Some symbols were dropped from the RRG due to insufficient history.")
        with st.expander("Symbols dropped"):
            for sym, (have, need) in dropped.items():
                st.write(f"{sym}: have {have} weekly points after calculations, need ≥ {need}.")


if __name__ == "__main__":
    main()
