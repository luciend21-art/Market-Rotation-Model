# rrg_app.py  –  Relative Rotation Graph (RRG) with Sectors, Themes, Commodities & Countries

import datetime as dt
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RRG – Sector & Theme Rotation", layout="wide")

# -------------------------
# 1. Universe definitions
# -------------------------

SECTOR_ETFS = {
    "XLY  Consumer Discretionary": "XLY",
    "XLP  Consumer Staples": "XLP",
    "XLE  Energy": "XLE",
    "XLF  Financials": "XLF",
    "XLV  Health Care": "XLV",
    "XLI  Industrials": "XLI",
    "XLB  Materials": "XLB",
    "XLK  Technology": "XLK",
    "XLU  Utilities": "XLU",
    "XLC  Communication Services": "XLC",
    "XLRE Real Estate": "XLRE",
}

THEME_ETFS = {
    "Semiconductors SOXX": "SOXX",
    "Semiconductors 2 SMH": "SMH",
    "Cybersecurity CIBR": "CIBR",
    "Cybersecurity 2 HACK": "HACK",
    "Cloud CLOU": "CLOU",
    "Cloud 2 SKYY": "SKYY",
    "Software IGV": "IGV",
    "Defense & Aerospace ITA": "ITA",
    "Defense & Aerospace 2 XAR": "XAR",
    "European Defense EUAD": "EUAD",
    "Clean Energy ICLN": "ICLN",
    "Solar TAN": "TAN",
    "Fintech / Innovation ARKF": "ARKF",
    "Infrastructure PAVE": "PAVE",
    "Digital Infrastructure DTCR": "DTCR",
    "Digital Infrastructure 2 TCAI": "TCAI",
    "Bitcoin Mining / HPC STCE": "STCE",
    "Bitcoin Mining / HPC 2 WGMI": "WGMI",
    "Home Construction ITB": "ITB",
    "Natural Gas BOIL": "BOIL",
    "Natural Gas 2 XOP": "XOP",
    "Robotics ROBO": "ROBO",
    "Robotics 2 BOTZ": "BOTZ",
    "Nuclear NLR": "NLR",
    "Nuclear 2 NUKZ": "NUKZ",
    "Biotech XBI": "XBI",
    "Biotech 2 BIB": "BIB",
    "Pharmaceutical PPH": "PPH",
    "Drone JEDI": "JEDI",
    "Drone 2 ARKQ": "ARKQ",
    "Brokerage IAI": "IAI",
    "Brokerage 2 RTH": "RTH",
    "Retail Shopping XRT": "XRT",
    "Utilities PUI": "PUI",
    "SPAC SPUC": "SPUC",
    "Regional Banking KRE": "KRE",
    "Banking KBE": "KBE",
    "Airlines JETS": "JETS",
    "Rare Earth REMX": "REMX",
    "Quantum QTUM": "QTUM",
    "Cannabis MSOS": "MSOS",
}

COMMODITY_ETFS = {
    "Gold RING": "RING",
    "Gold 2 IAU": "IAU",
    "Silver SLV": "SLV",
    "Silver 2 SIL": "SIL",
    "Copper COPX": "COPX",
    "Oil USO": "USO",
    "Bitcoin BITO": "BITO",
    "Ethereum ETHA": "ETHA",
    "Solana BSOL": "BSOL",
}

COUNTRY_ETFS = {
    "All World ex US VEU": "VEU",
    "Emerging Mkts ex China EMXC": "EMXC",
    "Brazil EWZ": "EWZ",
    "China MCHI": "MCHI",
    "China 2 KWEB": "KWEB",
    "China 3 CQQQ": "CQQQ",
    "Germany EWG": "EWG",
    "Canada EWC": "EWC",
    "Singapore EWS": "EWS",
}

UNIVERSES = {
    "SPDR Sectors": SECTOR_ETFS,
    "Themes ETFs": THEME_ETFS,
    "Commodity ETFs": COMMODITY_ETFS,
    "Country ETFs": COUNTRY_ETFS,
}

# default benchmark choices for each universe
UNIVERSE_BENCHMARKS = {
    "SPDR Sectors": ["SPY"],
    "Themes ETFs": ["SPY", "QQQ"],
    "Commodity ETFs": ["SPY", "DBC"],
    "Country ETFs": ["SPY", "ACWX", "VEU"],
}


# -------------------------
# 2. Data + RRG maths
# -------------------------

@st.cache_data(show_spinner=False)
def fetch_weekly_close(tickers, benchmark, years: int = 3) -> pd.DataFrame:
    """Download adjusted daily prices, convert to weekly closes."""
    end = dt.date.today()
    start = end - relativedelta(years=years)

    all_tickers = sorted(set(tickers + [benchmark]))
    data = yf.download(
        all_tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No price data returned from yfinance.")

    # Handle both single- and multi-index columns robustly
    if isinstance(data.columns, pd.MultiIndex):
        # first level is OHLCV, second is ticker
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"].copy()
        else:
            # fall back to last level if structure changes
            close = data.xs("Close", axis=1, level=0)
    else:
        if "Close" in data.columns:
            close = data[["Close"]].copy()
            close.columns = all_tickers  # if yfinance flattens columns oddly
        else:
            raise ValueError("Could not find 'Close' prices in downloaded data.")

    # Weekly closes on Friday
    weekly = close.resample("W-FRI").last()
    weekly = weekly.dropna(how="all")

    return weekly


def compute_rrg_coordinates(
    weekly_close: pd.DataFrame,
    tickers,
    benchmark: str,
    lookback_weeks: int = 52,
    momentum_period: int = 13,
):
    """
    Compute normalized RS-Ratio (X) and RS-Momentum (Y) series for RRG.
    """
    # Relative strength vs benchmark
    rs = weekly_close[tickers].div(weekly_close[benchmark], axis=0)
    rs = rs.dropna(how="all")

    log_rs = np.log(rs)

    # RS-Ratio: z-score of log RS over lookback window
    mean_rs = log_rs.rolling(lookback_weeks).mean()
    std_rs = log_rs.rolling(lookback_weeks).std()
    rs_ratio = (log_rs - mean_rs) / std_rs

    # RS-Momentum: z-score of momentum (change in log RS over momentum_period)
    mom_raw = log_rs.diff(momentum_period)
    mean_mom = mom_raw.rolling(lookback_weeks).mean()
    std_mom = mom_raw.rolling(lookback_weeks).std()
    rs_mom = (mom_raw - mean_mom) / std_mom

    return rs_ratio, rs_mom


def build_rrg_tail(
    rs_ratio: pd.DataFrame,
    rs_mom: pd.DataFrame,
    tickers,
    tail_length: int = 13,
) -> pd.DataFrame:
    """
    Convert RS-Ratio & RS-Momentum into a long-form dataframe
    of tail points for plotting.
    """
    valid = rs_ratio.dropna().index.intersection(rs_mom.dropna().index)

    if len(valid) == 0:
        return pd.DataFrame(columns=["date", "symbol", "step", "rs_ratio", "rs_momentum"])

    tail_dates = valid[-tail_length:]

    rows = []
    for step_idx, d in enumerate(tail_dates):
        for sym in tickers:
            x = rs_ratio.loc[d, sym]
            y = rs_mom.loc[d, sym]
            if pd.isna(x) or pd.isna(y):
                continue
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "step": step_idx,
                    "rs_ratio": x,
                    "rs_momentum": y,
                }
            )

    return pd.DataFrame(rows)


def make_rrg_figure(rrg_df: pd.DataFrame, label_map: dict) -> go.Figure:
    """
    Plot the RRG with tails using Plotly.
    """
    fig = go.Figure()

    if rrg_df.empty:
        fig.update_layout(
            title="No data to display",
            xaxis_title="RS-Ratio (standardized)",
            yaxis_title="RS-Momentum (standardized)",
        )
        return fig

    # Color palette
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for i, sym in enumerate(sorted(rrg_df["symbol"].unique())):
        df_sym = rrg_df[rrg_df["symbol"] == sym].sort_values("step")
        name = label_map.get(sym, sym)

        color = palette[i % len(palette)]

        # Tail line
        fig.add_trace(
            go.Scatter(
                x=df_sym["rs_ratio"],
                y=df_sym["rs_momentum"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6, color=color),
                name=name,
                text=[f"{name}<br>{d.date()}" for d in df_sym["date"]],
                hovertemplate="RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<br>%{text}<extra></extra>",
            )
        )

        # Highlight last point with a larger marker
        last = df_sym.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last["rs_ratio"]],
                y=[last["rs_momentum"]],
                mode="markers",
                marker=dict(size=12, color=color, symbol="circle-open-dot", line=dict(width=2, color="black")),
                showlegend=False,
                hovertemplate="Latest<br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>",
            )
        )

    # Quadrant lines
    fig.add_shape(type="line", x0=0, x1=0, y0=rrg_df["rs_momentum"].min() - 0.5, y1=rrg_df["rs_momentum"].max() + 0.5,
                  line=dict(color="lightgray", width=1))
    fig.add_shape(type="line", x0=rrg_df["rs_ratio"].min() - 0.5, x1=rrg_df["rs_ratio"].max() + 0.5, y0=0, y1=0,
                  line=dict(color="lightgray", width=1))

    # Quadrant labels
    x_min, x_max = rrg_df["rs_ratio"].min(), rrg_df["rs_ratio"].max()
    y_min, y_max = rrg_df["rs_momentum"].min(), rrg_df["rs_momentum"].max()
    x_mid_left = (x_min + 0) / 2
    x_mid_right = (x_max + 0) / 2
    y_mid_low = (y_min + 0) / 2
    y_mid_high = (y_max + 0) / 2

    fig.add_annotation(x=x_mid_right, y=y_mid_high, text="Leading", showarrow=False, font=dict(color="green"))
    fig.add_annotation(x=x_mid_right, y=y_mid_low, text="Weakening", showarrow=False, font=dict(color="orange"))
    fig.add_annotation(x=x_mid_left, y=y_mid_low, text="Lagging", showarrow=False, font=dict(color="red"))
    fig.add_annotation(x=x_mid_left, y=y_mid_high, text="Improving", showarrow=False, font=dict(color="blue"))

    fig.update_layout(
        xaxis_title="RS-Ratio (standardized)",
        yaxis_title="RS-Momentum (standardized)",
        legend_title="Tails (most recent point highlighted)",
        hovermode="closest",
        template="plotly_white",
    )

    return fig


# -------------------------
# 3. Streamlit UI
# -------------------------

def main():
    st.title("Relative Rotation Graph (RRG)")
    st.caption("Track sector, theme, commodity, and country rotation vs a benchmark.")

    with st.sidebar:
        st.header("RRG Settings")

        universe_name = st.selectbox(
            "Universe",
            list(UNIVERSES.keys()),
            index=0,
        )

        universe = UNIVERSES[universe_name]

        tickers_labels = list(universe.keys())
        default_selection = tickers_labels  # by default include all
        selected_labels = st.multiselect(
            "Choose ETFs",
            options=tickers_labels,
            default=default_selection,
        )

        if not selected_labels:
            st.warning("Select at least one ETF to plot.")
            return

        tickers = [universe[label] for label in selected_labels]

        # benchmark options based on universe
        bench_choices = UNIVERSE_BENCHMARKS.get(universe_name, ["SPY"])
        benchmark = st.selectbox("Benchmark", bench_choices, index=0)

        years = st.slider("History (years)", min_value=1, max_value=10, value=3)
        lookback = st.slider("Lookback window (weeks)", min_value=26, max_value=104, value=52, step=2)
        mom_period = st.slider("Momentum period (weeks)", min_value=4, max_value=26, value=13, step=1)
        tail_len = st.slider("Tail length (weeks)", min_value=4, max_value=26, value=13, step=1)

    # Fetch and compute
    try:
        weekly = fetch_weekly_close(tickers, benchmark, years=years)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    rs_ratio, rs_mom = compute_rrg_coordinates(
        weekly_close=weekly,
        tickers=tickers,
        benchmark=benchmark,
        lookback_weeks=lookback,
        momentum_period=mom_period,
    )

    rrg_df = build_rrg_tail(rs_ratio, rs_mom, tickers=tickers, tail_length=tail_len)

    if rrg_df.empty:
        st.warning("Not enough data to build RRG (try shorter lookback or different symbols).")
        return

    # Label map for pretty names in legend
    label_map = {universe[label]: label for label in selected_labels}

    fig = make_rrg_figure(rrg_df, label_map)
    st.plotly_chart(fig, use_container_width=True)

    # Snapshot table of latest points
    latest_step = rrg_df["step"].max()
    latest = (
        rrg_df[rrg_df["step"] == latest_step][["symbol", "rs_ratio", "rs_momentum"]]
        .replace({"symbol": {v: k for k, v in label_map.items()}})
        .rename(columns={"symbol": "ETF", "rs_ratio": "RS-Ratio", "rs_momentum": "RS-Momentum"})
        .sort_values("RS-Ratio", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Latest RRG Snapshot")
    st.dataframe(latest, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
