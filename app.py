import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ------------------------------------------------------------
# Universe definitions
# ------------------------------------------------------------

def get_universes():
    # SPDR sector ETFs
    sectors = {
        "XLB  Materials": "XLB",
        "XLC  Communication Services": "XLC",
        "XLE  Energy": "XLE",
        "XLF  Financials": "XLF",
        "XLI  Industrials": "XLI",
        "XLK  Technology": "XLK",
        "XLP  Consumer Staples": "XLP",
        "XLRE Real Estate": "XLRE",
        "XLU  Utilities": "XLU",
        "XLV  Health Care": "XLV",
        "XLY  Consumer Discretionary": "XLY",
    }

    # Theme ETFs (from your sheet; labels kept descriptive)
    themes = {
        "Semiconductors 1 (SOXX)": "SOXX",
        "Semiconductors 2 (SMH)": "SMH",
        "Cybersecurity 1 (CIBR)": "CIBR",
        "Cybersecurity 2 (HACK)": "HACK",
        "Cloud 1 (CLOU)": "CLOU",
        "Cloud 2 (SKYY)": "SKYY",
        "Software (IGV)": "IGV",
        "Defense & Aerospace 1 (ITA)": "ITA",
        "Defense & Aerospace 2 (XAR)": "XAR",
        "European Defense (EUAD)": "EUAD",
        "Clean Energy (ICLN)": "ICLN",
        "Solar (TAN)": "TAN",
        "Fintech / Innovation (ARKF)": "ARKF",
        "Infrastructure (PAVE)": "PAVE",
        "Digital Infrastructure 1 (DTCR)": "DTCR",
        "Digital Infrastructure 2 (TCAI)": "TCAI",
        "Bitcoin Mining / HPC 1 (WGMI)": "WGMI",
        "Bitcoin Mining / HPC 2 (STCE)": "STCE",
        "Home Construction (ITB)": "ITB",
        "Natural Gas 1 (BOIL)": "BOIL",
        "Natural Gas 2 (XOP)": "XOP",
        "Robotics 1 (ROBO)": "ROBO",
        "Robotics 2 (BOTZ)": "BOTZ",
        "Nuclear 1 (NLR)": "NLR",
        "Nuclear 2 (NUKZ)": "NUKZ",
        "Biotech 1 (XBI)": "XBI",
        "Biotech 2 (BIB)": "BIB",
        "Pharmaceutical (PPH)": "PPH",
        "Drone 1 (JEDI)": "JEDI",
        "Drone 2 (ARKQ)": "ARKQ",
        "Brokerage 1 (IAI)": "IAI",
        "Retail Shopping (XRT)": "XRT",
        "Utilities (PUI)": "PUI",
        "Regional Banking (KRE)": "KRE",
        "Banking (KBE)": "KBE",
        "Airlines (JETS)": "JETS",
        "Rare Earth (REMX)": "REMX",
        "Quantum (QTUM)": "QTUM",
        "Cannabis (MSOS)": "MSOS",
    }

    # Commodity ETFs
    commodities = {
        "Gold 1 (RING)": "RING",
        "Gold 2 (IAU)": "IAU",
        "Silver 1 (SLV)": "SLV",
        "Silver 2 (SIL)": "SIL",
        "Copper (COPX)": "COPX",
        "Oil 1 (OIL)": "OIL",
        "Oil 2 (BNO)": "BNO",
        "Ethereum (ETHA)": "ETHA",
        "Solana (BSOL)": "BSOL",
    }

    # Country / region ETFs
    countries = {
        "All World ex US (VEU)": "VEU",
        "Emerging Markets ex China (EMXC)": "EMXC",
        "Brazil (EWZ)": "EWZ",
        "India (INDA)": "INDA",
        "China 1 (MCHI)": "MCHI",
        "China 2 (KWEB)": "KWEB",
        "Chile (ECH)": "ECH",
        "Germany (EWG)": "EWG",
        "Canada (EWC)": "EWC",
        "Singapore (EWS)": "EWS",
    }

    universes = {
        "SPDR Sectors": sectors,
        "Themes ETFs": themes,
        "Commodity ETFs": commodities,
        "Country ETFs": countries,
    }

    default_benchmarks = {
        "SPDR Sectors": "SPY",
        "Themes ETFs": "SPY",
        "Commodity ETFs": "DBC",
        "Country ETFs": "ACWI",
    }

    return universes, default_benchmarks


# ------------------------------------------------------------
# Data + RRG calculations
# ------------------------------------------------------------

def download_prices(symbols, start, end):
    data = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(data, pd.DataFrame) and "Close" in data.columns:
        prices = data["Close"]
    else:
        # yfinance returns a Series when there is only one symbol
        prices = data
    return prices


def build_rrg_data(prices_w, symbols, benchmark, lookback, momentum, tail_len):
    """
    prices_w: weekly prices DataFrame (columns = symbols + benchmark)
    Returns:
        rrg_dict: {symbol: DataFrame with RS_Ratio, RS_Momentum}
        snapshot: DataFrame with last point per symbol
        skipped: list of (symbol, available_points, required_points)
    """
    min_points = lookback + momentum + tail_len + 4

    rrg_dict = {}
    snapshots = []
    skipped = []

    for sym in symbols:
        if sym not in prices_w.columns:
            skipped.append((sym, 0, min_points))
            continue

        df = pd.DataFrame({"asset": prices_w[sym], "bench": prices_w[benchmark]}).dropna()
        n = len(df)
        if n < min_points:
            skipped.append((sym, n, min_points))
            continue

        rs = df["asset"] / df["bench"]

        # RS-Ratio: z-score of relative strength over rolling lookback
        rolling_mean = rs.rolling(lookback).mean()
        rolling_std = rs.rolling(lookback).std()
        rs_ratio = (rs - rolling_mean) / rolling_std

        # RS-Momentum: change in RS-Ratio over "momentum" window
        rs_mom = rs_ratio.diff(momentum)

        rrg = pd.DataFrame(
            {"RS_Ratio": rs_ratio, "RS_Momentum": rs_mom}
        ).dropna()

        if len(rrg) < tail_len:
            skipped.append((sym, len(rrg), tail_len))
            continue

        rrg_tail = rrg.iloc[-tail_len:].copy()
        rrg_dict[sym] = rrg_tail

        last = rrg_tail.iloc[-1]
        snapshots.append(
            {
                "Symbol": sym,
                "RS_Ratio": float(last["RS_Ratio"]),
                "RS_Momentum": float(last["RS_Momentum"]),
            }
        )

    if not snapshots:
        return {}, None, skipped

    snapshot_df = pd.DataFrame(snapshots)
    snapshot_df["Quadrant"] = snapshot_df.apply(
        lambda r: classify_quadrant(r["RS_Ratio"], r["RS_Momentum"]), axis=1
    )

    return rrg_dict, snapshot_df, skipped


def classify_quadrant(x, y):
    if x >= 0 and y >= 0:
        return "Leading"
    elif x >= 0 and y < 0:
        return "Weakening"
    elif x < 0 and y < 0:
        return "Lagging"
    else:
        return "Improving"


def make_rrg_figure(rrg_dict, label_map):
    # Collect all points to set symmetric axis ranges
    all_x, all_y = [], []
    for sym, df in rrg_dict.items():
        all_x.extend(df["RS_Ratio"].values)
        all_y.extend(df["RS_Momentum"].values)

    if not all_x:
        return go.Figure()

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    max_extent = float(max(np.max(np.abs(all_x)), np.max(np.abs(all_y))))
    pad = 0.5
    axis_range = [-max_extent - pad, max_extent + pad]

    fig = go.Figure()

    # Quadrant shading
    r0, r1 = axis_range
    quad_shapes = [
        # Leading (top-right)
        dict(
            type="rect",
            x0=0,
            x1=r1,
            y0=0,
            y1=r1,
            fillcolor="rgba(0, 200, 0, 0.05)",
            line_width=0,
        ),
        # Weakening (bottom-right)
        dict(
            type="rect",
            x0=0,
            x1=r1,
            y0=r0,
            y1=0,
            fillcolor="rgba(255, 165, 0, 0.05)",
            line_width=0,
        ),
        # Lagging (bottom-left)
        dict(
            type="rect",
            x0=r0,
            x1=0,
            y0=r0,
            y1=0,
            fillcolor="rgba(200, 0, 0, 0.05)",
            line_width=0,
        ),
        # Improving (top-left)
        dict(
            type="rect",
            x0=r0,
            x1=0,
            y0=0,
            y1=r1,
            fillcolor="rgba(0, 0, 200, 0.05)",
            line_width=0,
        ),
    ]
    fig.update_layout(shapes=quad_shapes)

    # Quadrant labels
    fig.add_annotation(
        x=r1 * 0.7,
        y=r1 * 0.7,
        text="Leading",
        showarrow=False,
        font=dict(color="green"),
    )
    fig.add_annotation(
        x=r1 * 0.7,
        y=r0 * 0.7,
        text="Weakening",
        showarrow=False,
        font=dict(color="orange"),
    )
    fig.add_annotation(
        x=r0 * 0.7,
        y=r0 * 0.7,
        text="Lagging",
        showarrow=False,
        font=dict(color="red"),
    )
    fig.add_annotation(
        x=r0 * 0.7,
        y=r1 * 0.7,
        text="Improving",
        showarrow=False,
        font=dict(color="blue"),
    )

    # Add tails
    for sym, df in rrg_dict.items():
        label = label_map.get(sym, sym)
        fig.add_trace(
            go.Scatter(
                x=df["RS_Ratio"],
                y=df["RS_Momentum"],
                mode="lines+markers",
                name=label,
                text=[label] * len(df),
                hovertemplate=(
                    "%{text}<br>RS-Ratio: %{x:.2f}<br>"
                    "RS-Momentum: %{y:.2f}<extra></extra>"
                ),
            )
        )

        # Highlight last point
        last = df.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last["RS_Ratio"]],
                y=[last["RS_Momentum"]],
                mode="markers",
                marker=dict(size=10, symbol="circle", line=dict(width=1, color="black")),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_xaxes(
        title="RS-Ratio (standardized)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="black",
        range=axis_range,
    )
    fig.update_yaxes(
        title="RS-Momentum (standardized)",
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="black",
        range=axis_range,
    )

    fig.update_layout(
        title="Relative Rotation Graph (RRG)",
        legend=dict(
            title="Tails (most recent point highlighted)", orientation="v", x=1.02, y=1
        ),
        margin=dict(l=40, r=220, t=60, b=40),
    )

    return fig


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="RRG Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Relative Rotation Graph (RRG)")
    st.caption(
        "Track sector, theme, commodity, and country rotation vs a benchmark. "
        "Rough approximation of JdK RS-Ratio and RS-Momentum using weekly data."
    )

    universes, default_benchmarks = get_universes()

    # ---- Sidebar controls ----
    st.sidebar.header("RRG Settings")

    universe_name = st.sidebar.selectbox("Universe", list(universes.keys()))

    universe_map = universes[universe_name]
    labels = list(universe_map.keys())
    tickers = list(universe_map.values())

    default_bench = default_benchmarks[universe_name]
    bench = st.sidebar.text_input("Benchmark", value=default_bench)

    selected_labels = st.sidebar.multiselect(
        "Choose ETFs",
        labels,
        default=labels,
        help="You can deselect items here if you want a smaller universe.",
    )

    if not selected_labels:
        st.warning("Please select at least one ETF.")
        return

    selected_symbols = [universe_map[l] for l in selected_labels]
    label_map = {universe_map[l]: l for l in selected_labels}

    history_years = st.sidebar.slider(
        "History (years, daily data)",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="How many years of daily data to download before resampling to weekly.",
    )

    lookback_weeks = st.sidebar.slider(
        "Lookback window (weeks)",
        min_value=26,
        max_value=104,
        value=52,
        step=1,
        help="Longer = smoother RS-Ratio.",
    )

    momentum_weeks = st.sidebar.slider(
        "Momentum period (weeks)",
        min_value=4,
        max_value=26,
        value=13,
        step=1,
        help="Change in RS-Ratio over this window.",
    )

    tail_len = st.sidebar.slider(
        "Tail length (weeks)",
        min_value=4,
        max_value=26,
        value=13,
        step=1,
        help="How many weeks to plot in each tail.",
    )

    # ---- Download data ----
    end = datetime.today()
    start = end - relativedelta(years=history_years)

    all_symbols = sorted(set(selected_symbols + [bench]))

    with st.spinner("Downloading price data from Yahoo Finance..."):
        prices_d = download_prices(all_symbols, start, end)

    if prices_d is None or len(prices_d) == 0:
        st.error("No data returned from Yahoo Finance. Please check tickers.")
        return

    # Ensure DataFrame
    if isinstance(prices_d, pd.Series):
        prices_d = prices_d.to_frame()

    prices_d = prices_d.dropna(how="all")

    # Weekly resample (Friday close)
    prices_w = prices_d.resample("W-FRI").last().dropna(how="all")

    if bench not in prices_w.columns:
        st.error(f"Benchmark {bench} has no data for the selected period.")
        return

    # ---- Build RRG data ----
    rrg_dict, snapshot_df, skipped = build_rrg_data(
        prices_w, selected_symbols, bench, lookback_weeks, momentum_weeks, tail_len
    )

    # History warnings
    if skipped:
        skipped_msg = ", ".join(
            f"{sym} (have {have} wk, need ≥ {need})" for sym, have, need in skipped
        )
        st.sidebar.warning(
            "Some symbols were dropped due to insufficient history:\n\n" + skipped_msg
        )

    if not rrg_dict:
        st.warning(
            "Not enough data to build RRG (try a shorter lookback / momentum window "
            "or remove very new ETFs)."
        )
        return

    # ---- Main chart ----
    fig = make_rrg_figure(rrg_dict, label_map)
    st.plotly_chart(fig, use_container_width=True)

    # ---- Snapshot & summary ----
    st.subheader("Latest RRG Snapshot")

    sort_choice = st.radio(
        "Sort summary by:",
        options=("RS-Ratio", "RS-Momentum"),
        horizontal=True,
    )

    if snapshot_df is not None and not snapshot_df.empty:
        # Map back to labels for readability
        snapshot_df = snapshot_df.copy()
        snapshot_df["Name"] = snapshot_df["Symbol"].map(label_map)

        sort_col = sort_choice

        leading = (
            snapshot_df[snapshot_df["Quadrant"] == "Leading"]
            .sort_values(sort_col, ascending=False)
            .head(5)
        )
        improving = (
            snapshot_df[snapshot_df["Quadrant"] == "Improving"]
            .sort_values(sort_col, ascending=False)
            .head(3)
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Leading quadrant (top 5)**")
            if leading.empty:
                st.write("None currently in Leading.")
            else:
                st.table(
                    leading[["Name", "Symbol", "RS-Ratio", "RS-Momentum"]].reset_index(
                        drop=True
                    )
                )

        with col2:
            st.markdown("**Top 3 Improving (moving toward Leading)**")
            if improving.empty:
                st.write("None currently in Improving.")
            else:
                st.table(
                    improving[
                        ["Name", "Symbol", "RS-Ratio", "RS-Momentum"]
                    ].reset_index(drop=True)
                )


if __name__ == "__main__":
    main()
