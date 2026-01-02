import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

st.set_page_config(
    page_title="Market Rotation Model – RRG",
    layout="wide",
)

# Predefined universes
UNIVERSES = {
    "SPDR Sectors": {
        "etfs": {
            "XLB Materials": "XLB",
            "XLC Communication Services": "XLC",
            "XLE Energy": "XLE",
            "XLF Financials": "XLF",
            "XLI Industrials": "XLI",
            "XLP Consumer Staples": "XLP",
            "XLRE Real Estate": "XLRE",
            "XLU Utilities": "XLU",
            "XLV Health Care": "XLV",
            "XLY Consumer Discretionary": "XLY",
            "XLK Technology": "XLK",
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
            "Bitcoin Mining / HPC (WGMI)": "WGMI",
            "Bitcoin Mining / HPC 2 (STCE)": "STCE",
            "Home Construction (ITB)": "ITB",
            "Natural Gas (BOIL)": "BOIL",
            "Natural Gas 2 (XOP)": "XOP",
            "Robotics (ROBO)": "ROBO",
            "Robotics 2 (BOTZ)": "BOTZ",
            "Biotech (XBI)": "XBI",
            "Biotech 2 (BIB)": "BIB",
            "Pharmaceutical (PPH)": "PPH",
            "Drone (JEDI)": "JEDI",
            "Drone 2 (ARKQ)": "ARKQ",
            "Brokerage (IAI)": "IAI",
            "Retail Shopping (XRT)": "XRT",
            "Utilities (PUI)": "PUI",
            "SPAC (SPAK)": "SPAK",
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
            "Gold (GLD)": "GLD",
            "Gold 2 (IAU)": "IAU",
            "Silver (SLV)": "SLV",
            "Silver 2 (SIL)": "SIL",
            "Copper (CPER)": "CPER",
            "Oil (USO)": "USO",        # <- use USO, not OIL
            "Bitcoin (BITO)": "BITO",
            "Ethereum (ETHA)": "ETHA",
            "Solana (BSOL)": "BSOL",
        }
    },
    "Country ETFs": {
        "etfs": {
            "All World ex-US (VEU)": "VEU",
            "Emerging Mkts ex-China (EMXC)": "EMXC",
            "Brazil (EWZ)": "EWZ",
            "China (MCHI)": "MCHI",
            "China Internet (KWEB)": "KWEB",
            "Germany (EWG)": "EWG",
            "Canada (EWC)": "EWC",
            "Singapore (EWS)": "EWS",
        }
    },
}

DEFAULT_BENCHMARKS = ["SPY", "QQQ", "IWM", "ACWI", "DIA"]


# ------------------------------------------------------------
# Data download helpers
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def download_daily(tickers, years: int) -> pd.DataFrame:
    """
    Download daily prices for a list of tickers and return a wide
    DataFrame of Close prices. Robust to:
    - MultiIndex columns from yfinance
    - Tickers with no data (dropped)
    """
    tickers = list(dict.fromkeys(tickers))  # unique, preserve order

    end = datetime.today()
    start = end - timedelta(days=years * 365)

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",   # gives MultiIndex: (ticker, field)
        threads=True,
    )

    if data.empty:
        raise Exception(f"No price data returned for tickers: {tickers}")

    # Multi-ticker case -> MultiIndex (ticker, field)
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)   # tickers
        level1 = data.columns.get_level_values(1)   # fields (Open, High, Close,...)

        if "Close" in level1:
            close = data.xs("Close", axis=1, level=1)
        elif "Adj Close" in level1:
            close = data.xs("Adj Close", axis=1, level=1)
        else:
            raise Exception(
                f"Expected 'Close' or 'Adj Close' columns, got fields: {sorted(set(level1))}"
            )

        # Ensure column names are just ticker strings
        close.columns = close.columns.astype(str)

    # Single-ticker case -> normal columns ('Open', 'High', 'Close',...)
    else:
        cols = list(data.columns)
        if "Close" in cols:
            close = data["Close"].to_frame()
        elif "Adj Close" in cols:
            close = data["Adj Close"].to_frame()
        else:
            raise Exception(
                f"Expected 'Close' or 'Adj Close' columns, got {cols}"
            )
        close.columns = [tickers[0]]

    # Drop tickers with no usable prices
    close = close.dropna(axis=1, how="all")

    if close.empty:
        raise Exception(
            f"No usable Close prices for tickers: {tickers} "
            "(they may be delisted or lack any history)."
        )

    return close


def to_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily close prices to weekly (Friday) closes."""
    weekly = prices.resample("W-FRI").last()
    weekly = weekly.dropna(how="all")
    return weekly


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across columns (symbols)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def compute_rrg(
    weekly_prices: pd.DataFrame,
    benchmark: str,
    lookback_weeks: int,
    momentum_weeks: int,
    tail_length: int,
):
    """
    Compute RS-Ratio and RS-Momentum series for all assets vs benchmark.
    Returns:
        rrg_tails: dict[ticker] -> dict(x_series, y_series, x_last, y_last)
        meta: dict with dropped_symbols, needed_weeks, have_weeks
    """
    if benchmark not in weekly_prices.columns:
        raise ValueError(f"Benchmark {benchmark} not in price data.")

    # Align on benchmark (drop rows where benchmark is NaN)
    weekly = weekly_prices.dropna(subset=[benchmark])
    weekly = weekly.sort_index()

    assets = [c for c in weekly.columns if c != benchmark]

    if not assets:
        raise ValueError("No assets left after filtering benchmark.")

    needed_weeks = lookback_weeks + momentum_weeks + tail_length
    have_weeks = len(weekly)

    if have_weeks < needed_weeks + 5:
        raise ValueError(
            f"Not enough weekly data for selected windows. "
            f"Need at least ~{needed_weeks} weeks, have {have_weeks}."
        )

    bench = weekly[benchmark]

    # Relative strength (price / benchmark)
    rs = weekly[assets].div(bench, axis=0)

    # RS-Ratio approx: rolling mean of log RS
    rs_ratio_raw = np.log(rs).rolling(lookback_weeks).mean()

    # RS-Momentum approx: difference of RS-Ratio over momentum window
    rs_mom_raw = rs_ratio_raw.diff(momentum_weeks)

    # Cross-sectional z-scores
    rs_ratio_z = cross_sectional_zscore(rs_ratio_raw)
    rs_mom_z = cross_sectional_zscore(rs_mom_raw)

    # Use only last tail_length points for tails
    rs_ratio_tail = rs_ratio_z.iloc[-tail_length:]
    rs_mom_tail = rs_mom_z.iloc[-tail_length:]

    rrg_tails = {}
    dropped_symbols = []

    for sym in assets:
        x_series = rs_ratio_tail[sym]
        y_series = rs_mom_tail[sym]

        # Require at least some valid points
        if x_series.isna().all() or y_series.isna().all():
            dropped_symbols.append(sym)
            continue

        # Drop any leading NaNs inside tail
        valid = (~x_series.isna()) & (~y_series.isna())
        x_series = x_series[valid]
        y_series = y_series[valid]

        if len(x_series) == 0:
            dropped_symbols.append(sym)
            continue

        rrg_tails[sym] = {
            "x_series": x_series,
            "y_series": y_series,
            "x_last": x_series.iloc[-1],
            "y_last": y_series.iloc[-1],
        }

    meta = {
        "dropped_symbols": dropped_symbols,
        "needed_weeks": needed_weeks,
        "have_weeks": have_weeks,
    }

    return rrg_tails, meta


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

def make_rrg_figure(rrg_tails, display_names):
    """Build Plotly RRG figure from tails dict, highlighting the latest point."""

    if not rrg_tails:
        raise ValueError("No symbols with valid RRG data.")

    # Color palette so tail + head use the same color
    palette = px.colors.qualitative.Plotly

    # Collect all points to determine ranges
    all_x = np.concatenate([v["x_series"].values for v in rrg_tails.values()])
    all_y = np.concatenate([v["y_series"].values for v in rrg_tails.values()])

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

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

    # ---- Quadrant shading ----
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=x_max,
        y1=y_max,
        fillcolor="rgba(0, 200, 0, 0.07)",
        line=dict(width=0),
        layer="below",
    )  # Leading

    fig.add_shape(
        type="rect",
        x0=x_min,
        y0=0,
        x1=0,
        y1=y_max,
        fillcolor="rgba(0, 0, 200, 0.05)",
        line=dict(width=0),
        layer="below",
    )  # Improving

    fig.add_shape(
        type="rect",
        x0=x_min,
        y0=y_min,
        x1=0,
        y1=0,
        fillcolor="rgba(200, 0, 0, 0.06)",
        line=dict(width=0),
        layer="below",
    )  # Lagging

    fig.add_shape(
        type="rect",
        x0=0,
        y0=y_min,
        x1=x_max,
        y1=0,
        fillcolor="rgba(200, 200, 0, 0.06)",
        line=dict(width=0),
        layer="below",
    )  # Weakening

    # Axes lines
    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=y_min,
        y1=y_max,
        line=dict(color="gray", width=1),
    )
    fig.add_shape(
        type="line",
        x0=x_min,
        x1=x_max,
        y0=0,
        y1=0,
        line=dict(color="gray", width=1),
    )

    # Quadrant labels
    fig.add_annotation(
        x=x_max * 0.85,
        y=y_max * 0.85,
        text="Leading",
        showarrow=False,
        font=dict(color="green", size=12),
    )
    fig.add_annotation(
        x=x_min * 0.85,
        y=y_max * 0.85,
        text="Improving",
        showarrow=False,
        font=dict(color="blue", size=12),
    )
    fig.add_annotation(
        x=x_min * 0.85,
        y=y_min * 0.85,
        text="Lagging",
        showarrow=False,
        font=dict(color="red", size=12),
    )
    fig.add_annotation(
        x=x_max * 0.85,
        y=y_min * 0.85,
        text="Weakening",
        showarrow=False,
        font=dict(color="orange", size=12),
    )

    # ---- Tails + highlighted heads ----
    for idx, (sym, tail) in enumerate(rrg_tails.items()):
        x = tail["x_series"]
        y = tail["y_series"]
        name = display_names.get(sym, sym)
        color = palette[idx % len(palette)]

        # Tail: all but last point (small markers + line, no legend)
        if len(x) > 1:
            fig.add_trace(
                go.Scatter(
                    x=x.iloc[:-1],
                    y=y.iloc[:-1],
                    mode="lines+markers",
                    name=name,
                    legendgroup=sym,
                    showlegend=False,  # legend only for the head
                    line=dict(width=1.5, color=color),
                    marker=dict(size=4, color=color),
                    hovertemplate=(
                        f"{name}<br>"
                        "RS-Ratio: %{x:.2f}<br>"
                        "RS-Momentum: %{y:.2f}<extra></extra>"
                    ),
                )
            )

        # Head: most recent point (big marker with black outline, shows in legend)
        fig.add_trace(
            go.Scatter(
                x=[x.iloc[-1]],
                y=[y.iloc[-1]],
                mode="markers",
                name=name,
                legendgroup=sym,
                showlegend=True,
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=2, color="black"),
                    symbol="circle",
                ),
                hovertemplate=(
                    f"{name} (latest)<br>"
                    "RS-Ratio: %{x:.2f}<br>"
                    "RS-Momentum: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        xaxis_title="RS-Ratio (standardized)",
        yaxis_title="RS-Momentum (standardized)",
        xaxis=dict(range=[x_min, x_max], zeroline=False),
        yaxis=dict(range=[y_min, y_max], zeroline=False),
        legend=dict(
            title="Most recent point (large marker)",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
        margin=dict(l=40, r=220, t=60, b=40),
        template="plotly_white",
    )

    return fig


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

def main():
    st.title("Relative Rotation Graph (RRG)")

    st.caption(
        "Track sector, theme, commodity, and country rotation vs a benchmark. "
        "Approximation of JdK RS-Ratio and RS-Momentum using weekly data."
    )

    # Sidebar
    with st.sidebar:
        st.header("RRG Settings")

        universe_name = st.selectbox("Universe", list(UNIVERSES.keys()))
        uni = UNIVERSES[universe_name]["etfs"]

        benchmark = st.selectbox("Benchmark", DEFAULT_BENCHMARKS, index=0)

        # Choose ETFs from universe
        default_labels = list(uni.keys())
        chosen_labels = st.multiselect(
            "Choose ETFs",
            options=default_labels,
            default=default_labels,
            help="Add/remove items from this universe.",
        )

        extra_raw = st.text_input(
            "Extra tickers (comma-separated, e.g. 'QQQ, IWM, HII')"
        )

        history_years = st.slider(
            "History (years, daily data)",
            min_value=1,
            max_value=10,
            value=3,
        )

        lookback_weeks = st.slider(
            "Lookback window (weeks)",
            min_value=20,
            max_value=78,
            value=52,
        )

        momentum_weeks = st.slider(
            "Momentum period (weeks)",
            min_value=5,
            max_value=26,
            value=13,
        )

        tail_length = st.slider(
            "Tail length (weeks)",
            min_value=5,
            max_value=26,
            value=13,
        )

    # Build list of tickers and mapping to display names
    display_to_sym = {label: uni[label] for label in chosen_labels}

    # Extra tickers typed in
    extra_raw = extra_raw.strip()
    if extra_raw:
        for tok in extra_raw.split(","):
            sym = tok.strip().upper()
            if not sym:
                continue
            display_to_sym[sym] = sym  # display = symbol

    # Add benchmark to tickers list (even if not selected as asset)
    tickers = list(display_to_sym.values())
    if benchmark not in tickers:
        tickers.append(benchmark)

    if not display_to_sym:
        st.warning("Please select at least one ETF.")
        return

    # Download data
    try:
        daily_close = download_daily(tickers, history_years)
    except Exception as e:
        st.error(f"Error downloading data from Yahoo Finance: {e}")
        return

    # Weekly data
    weekly = to_weekly(daily_close)

    # Ensure benchmark is present
    if benchmark not in weekly.columns:
        st.error(
            f"Benchmark {benchmark} has no weekly price data "
            f"for the selected history window."
        )
        return

    # Restrict to symbols we actually want to show (excluding benchmark)
    assets_symbols = [sym for sym in display_to_sym.values() if sym != benchmark]
    weekly = weekly[[c for c in weekly.columns if c in assets_symbols + [benchmark]]]

    # Map ticker -> display name
    symbol_to_display = {v: k for k, v in display_to_sym.items()}

    # Compute RRG
    try:
        rrg_tails, meta = compute_rrg(
            weekly,
            benchmark=benchmark,
            lookback_weeks=lookback_weeks,
            momentum_weeks=momentum_weeks,
            tail_length=tail_length,
        )
    except Exception as e:
        st.warning(str(e))
        return

    dropped = meta["dropped_symbols"]
    if dropped:
        st.sidebar.info(
            "Some symbols were dropped due to insufficient history or missing data:\n\n"
            + ", ".join(dropped)
        )

    # Build figure
    try:
        fig = make_rrg_figure(rrg_tails, symbol_to_display)
    except Exception as e:
        st.error(str(e))
        return

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
