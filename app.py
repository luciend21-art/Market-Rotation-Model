import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================
# App config
# =========================
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")


# =========================
# Predefined universes
# =========================
def predefined_universes() -> dict:
    # Format: { universe_name: DataFrame(columns=["Group","Ticker","Name"]) }
    sectors = pd.DataFrame(
        [
            ("SPDR Sectors", "XLB", "Materials"),
            ("SPDR Sectors", "XLC", "Communication Services"),
            ("SPDR Sectors", "XLE", "Energy"),
            ("SPDR Sectors", "XLF", "Financials"),
            ("SPDR Sectors", "XLI", "Industrials"),
            ("SPDR Sectors", "XLK", "Technology"),
            ("SPDR Sectors", "XLP", "Consumer Staples"),
            ("SPDR Sectors", "XLRE", "Real Estate"),
            ("SPDR Sectors", "XLU", "Utilities"),
            ("SPDR Sectors", "XLV", "Health Care"),
            ("SPDR Sectors", "XLY", "Consumer Discretionary"),
        ],
        columns=["Group", "Ticker", "Name"],
    )

    themes = pd.DataFrame(
        [
            ("Themes ETFs", "SOXX", "Semiconductors"),
            ("Themes ETFs", "SMH", "Semiconductors 2"),
            ("Themes ETFs", "CIBR", "Cybersecurity"),
            ("Themes ETFs", "HACK", "Cybersecurity 2"),
            ("Themes ETFs", "CLOU", "Cloud"),
            ("Themes ETFs", "SKYY", "Cloud 2"),
            ("Themes ETFs", "IGV", "Software"),
            ("Themes ETFs", "ITA", "Defense & Aerospace"),
            ("Themes ETFs", "XAR", "Defense & Aerospace 2"),
            ("Themes ETFs", "EUAD", "European Defense"),
            ("Themes ETFs", "ICLN", "Clean Energy"),
            ("Themes ETFs", "TAN", "Solar"),
            ("Themes ETFs", "ARKF", "Fintech / Innovation"),
            ("Themes ETFs", "PAVE", "Infrastructure"),
            ("Themes ETFs", "DTC R", "Digital Infrastructure"),  # note: will be filtered if invalid
            ("Themes ETFs", "TCAI", "Digital Infrastructure 2"),
            ("Themes ETFs", "WGMI", "Bitcoin Mining / HPC"),
            ("Themes ETFs", "ITB", "Home Construction"),
            ("Themes ETFs", "BOIL", "Natural Gas"),
            ("Themes ETFs", "XOP", "Natural Gas 2"),
            ("Themes ETFs", "ROBO", "Robotics"),
            ("Themes ETFs", "BOTZ", "Robotics 2"),
            ("Themes ETFs", "NLR", "Nuclear"),
            ("Themes ETFs", "NUKZ", "Nuclear 2"),
            ("Themes ETFs", "ARKG", "Biotech"),
            ("Themes ETFs", "BIB", "Biotech 2"),
            ("Themes ETFs", "PPH", "Pharmaceutical"),
            ("Themes ETFs", "ARKQ", "Drone"),
            ("Themes ETFs", "JEDI", "Drone 2"),
            ("Themes ETFs", "IAI", "Brokerage"),
            ("Themes ETFs", "RTH", "Brokerage 2"),
            ("Themes ETFs", "XRT", "Retail Shopping"),
            ("Themes ETFs", "UFO", "Space"),
            ("Themes ETFs", "KRE", "Regional Banking"),
            ("Themes ETFs", "KBE", "Banking"),
            ("Themes ETFs", "JETS", "Airlines"),
            ("Themes ETFs", "REMX", "Rare Earth"),
            ("Themes ETFs", "QTUM", "Quantum"),
            ("Themes ETFs", "MSOS", "Cannabis"),
        ],
        columns=["Group", "Ticker", "Name"],
    )

    commodities = pd.DataFrame(
        [
            ("Commodity ETFs", "RING", "Gold"),
            ("Commodity ETFs", "IAU", "Gold 2"),
            ("Commodity ETFs", "SIL", "Silver"),
            ("Commodity ETFs", "SLV", "Silver 2"),
            ("Commodity ETFs", "COPX", "Copper"),
            ("Commodity ETFs", "USO", "Oil"),
            ("Commodity ETFs", "BTC-USD", "Bitcoin"),
            ("Commodity ETFs", "ETH-USD", "Ethereum"),
            ("Commodity ETFs", "BSOL", "Solana"),
        ],
        columns=["Group", "Ticker", "Name"],
    )

    countries = pd.DataFrame(
        [
            ("Country ETFs", "VEU", "All World ex-US"),
            ("Country ETFs", "EMXC", "Emerging Mkts ex-China"),
            ("Country ETFs", "EEM", "Emerging Markets"),
            ("Country ETFs", "EWZ", "Brazil"),
            ("Country ETFs", "MCHI", "China"),
            ("Country ETFs", "KWEB", "China Internet"),
            ("Country ETFs", "EWG", "Germany"),
            ("Country ETFs", "EWC", "Canada"),
            ("Country ETFs", "EWS", "Singapore"),
        ],
        columns=["Group", "Ticker", "Name"],
    )

    # Clean tickers (strip, upper)
    def clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
        df["Name"] = df["Name"].astype(str).str.strip()
        df["Group"] = df["Group"].astype(str).str.strip()
        # remove obvious bad placeholders like "DTC R"
        df = df[df["Ticker"].str.match(r"^[A-Z0-9\-\.\=]+$")]
        df = df[df["Ticker"].str.len() > 0]
        return df.drop_duplicates(subset=["Ticker"], keep="first")

    return {
        "SPDR Sectors": clean_df(sectors),
        "Themes ETFs": clean_df(themes),
        "Commodity ETFs": clean_df(commodities),
        "Country ETFs": clean_df(countries),
    }


# =========================
# Upload / Manage Universe
# =========================
def parse_uploaded_universe(file) -> pd.DataFrame:
    """
    Expected columns (case-insensitive):
      - Group (required)
      - Ticker (required)
      - Name (optional)
    """
    if file is None:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"])

    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Upload must be a CSV or Excel file (.csv, .xlsx).")

    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    required = ["group", "ticker"]
    for r in required:
        if r not in cols:
            raise ValueError("File must include columns: Group, Ticker (Name optional).")

    group_col = cols["group"]
    ticker_col = cols["ticker"]
    name_col = cols.get("name", None)

    out = pd.DataFrame()
    out["Group"] = df[group_col].astype(str).str.strip()
    out["Ticker"] = df[ticker_col].astype(str).str.strip().str.upper()
    out["Name"] = df[name_col].astype(str).str.strip() if name_col else out["Ticker"]

    out = out[out["Ticker"].str.len() > 0]
    out = out[out["Group"].str.len() > 0]
    out = out[out["Ticker"].str.match(r"^[A-Z0-9\-\.\=]+$")]

    # first group wins within the uploaded file
    out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
    return out


def merge_universes(predef_df: pd.DataFrame, uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Option A: First group wins (uploaded first, then predef excluding duplicates).
    That means: if ticker exists in uploaded, we keep uploaded's Group/Name.
    """
    if uploaded_df is None or uploaded_df.empty:
        return predef_df.copy()

    uploaded_df = uploaded_df.copy()
    predef_df = predef_df.copy()

    uploaded_tickers = set(uploaded_df["Ticker"].tolist())
    predef_df = predef_df[~predef_df["Ticker"].isin(uploaded_tickers)]
    merged = pd.concat([uploaded_df, predef_df], ignore_index=True)
    return merged.reset_index(drop=True)


def template_csv_bytes() -> bytes:
    df = pd.DataFrame(
        [
            {"Group": "Themes ETFs", "Ticker": "SOXX", "Name": "Semiconductors"},
            {"Group": "Themes ETFs", "Ticker": "CIBR", "Name": "Cybersecurity"},
            {"Group": "Commodity ETFs", "Ticker": "IAU", "Name": "Gold"},
            {"Group": "Country ETFs", "Ticker": "EWG", "Name": "Germany"},
        ]
    )
    return df.to_csv(index=False).encode("utf-8")


# =========================
# Data fetch helpers
# =========================
@st.cache_data(show_spinner=False)
def yf_download_close(tickers: tuple, period: str, interval: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns as tickers and values as Adj Close (preferred) or Close.
    """
    tickers_list = list(dict.fromkeys([t for t in tickers if isinstance(t, str) and t.strip()]))
    if not tickers_list:
        return pd.DataFrame()

    df = yf.download(
        tickers_list,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # yfinance returns:
    # - single ticker: columns like ["Open","High","Low","Close","Adj Close","Volume"]
    # - multi ticker: columns multiindex [field, ticker]
    if isinstance(df.columns, pd.MultiIndex):
        fields = df.columns.get_level_values(0).unique().tolist()
        preferred = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else None)
        if preferred is None:
            raise ValueError("Expected 'Close' or 'Adj Close' from Yahoo Finance.")
        out = df[preferred].copy()
    else:
        preferred = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
        if preferred is None:
            raise ValueError("Expected 'Close' or 'Adj Close' from Yahoo Finance.")
        out = df[[preferred]].copy()
        out.columns = [tickers_list[0]]

    out.index = pd.to_datetime(out.index)
    out = out.dropna(how="all")
    return out


def to_weekly(close_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily close to weekly (Friday close).
    """
    if close_daily.empty:
        return close_daily
    weekly = close_daily.resample("W-FRI").last()
    return weekly.dropna(how="all")


# =========================
# RRG math (approximation)
# =========================
def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std()
    return (series - mean) / std


def compute_rrg_components(
    prices: pd.DataFrame,
    benchmark: str,
    lookback_weeks: int,
    momentum_weeks: int,
) -> pd.DataFrame:
    """
    Given price series (weekly or daily already aligned), compute:
      - RS series: price / benchmark_price
      - RS-Ratio (standardized): zscore of RS (lookback)
      - RS-Momentum (standardized): zscore of ROC of RS (momentum)
    """
    if benchmark not in prices.columns:
        raise ValueError("Benchmark price series missing from downloaded data.")

    df = prices.copy().dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # align
    bench = df[benchmark].dropna()
    df = df.loc[bench.index].copy()
    df = df.dropna(subset=[benchmark])

    out_rows = []
    for sym in df.columns:
        if sym == benchmark:
            continue
        s = df[sym].dropna()
        common_idx = s.index.intersection(bench.index)
        if len(common_idx) < max(lookback_weeks, momentum_weeks) + 5:
            continue

        rs = (s.loc[common_idx] / bench.loc[common_idx]).rename("RS")
        rs_ratio = zscore(rs, lookback_weeks).rename("RS_Ratio")
        rs_roc = rs.pct_change().rename("RS_ROC")
        rs_mom = zscore(rs_roc, momentum_weeks).rename("RS_Momentum")

        tmp = pd.concat([rs_ratio, rs_mom], axis=1).dropna()
        tmp["Ticker"] = sym
        tmp = tmp.reset_index().rename(columns={"index": "Date"})
        out_rows.append(tmp)

    if not out_rows:
        return pd.DataFrame(columns=["Date", "RS_Ratio", "RS_Momentum", "Ticker"])

    out = pd.concat(out_rows, ignore_index=True)
    return out


# =========================
# Interpretation helpers
# =========================
def classify_rs_ratio(x: float) -> str:
    return "Improving" if x >= 0 else "Weakening"


def classify_momentum(y: float) -> str:
    # 3-bucket for readability
    if y > 0.5:
        return "Improving"
    if y < -0.5:
        return "Weakening"
    return "Flat"


def quadrant(x: float, y: float) -> str:
    if x >= 0 and y >= 0:
        return "Leading"
    if x < 0 and y >= 0:
        return "Improving"
    if x < 0 and y < 0:
        return "Lagging"
    return "Weakening"


def angle_to_arrow(dx: float, dy: float) -> str:
    # 8-direction arrows based on angle
    ang = math.degrees(math.atan2(dy, dx))  # -180..180
    # bins centered on cardinal/intercardinal
    dirs = [
        (22.5, "→"),
        (67.5, "↗"),
        (112.5, "↑"),
        (157.5, "↖"),
        (180.0, "←"),
        (-157.5, "←"),
        (-112.5, "↙"),
        (-67.5, "↓"),
        (-22.5, "↘"),
    ]
    for th, ar in dirs:
        if ang <= th:
            return ar
    return "→"


def speed_bucket_from_percentile(p: float) -> str:
    # 4 buckets
    if p >= 0.75:
        return "Hot/Climactic"
    if p >= 0.50:
        return "Fast"
    if p >= 0.25:
        return "Medium"
    return "Slow"


# =========================
# Plotting
# =========================
def make_rrg_figure(
    rrg_df: pd.DataFrame,
    meta: pd.DataFrame,
    tail_len: int,
    title: str,
) -> go.Figure:
    """
    rrg_df: columns [Date, RS_Ratio, RS_Momentum, Ticker]
    meta: columns [Ticker, Name, Group, _COLORHEX]
    """
    fig = go.Figure()

    # Quadrant shading (light)
    # We'll set axis ranges dynamically from data with padding
    x_all = rrg_df["RS_Ratio"].values
    y_all = rrg_df["RS_Momentum"].values

    if len(x_all) == 0:
        fig.update_layout(title=title)
        return fig

    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)

    # padding so points outside don’t get clipped
    pad_x = max(0.75, 0.15 * (x_max - x_min + 1e-9))
    pad_y = max(0.75, 0.15 * (y_max - y_min + 1e-9))

    x0, x1 = x_min - pad_x, x_max + pad_x
    y0, y1 = y_min - pad_y, y_max + pad_y

    # shaded rectangles
    fig.add_shape(type="rect", x0=x0, x1=0, y0=0, y1=y1, fillcolor="rgba(80,120,255,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=x1, y0=0, y1=y1, fillcolor="rgba(80,200,120,0.10)", line_width=0)
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0, fillcolor="rgba(255,80,80,0.08)", line_width=0)
    fig.add_shape(type="rect", x0=0, x1=x1, y0=y0, y1=0, fillcolor="rgba(255,200,80,0.10)", line_width=0)

    # axes cross
    fig.add_shape(type="line", x0=0, x1=0, y0=y0, y1=y1, line=dict(color="rgba(0,0,0,0.35)", width=1))
    fig.add_shape(type="line", x0=x0, x1=x1, y0=0, y1=0, line=dict(color="rgba(0,0,0,0.35)", width=1))

    # quadrant labels (corners)
    fig.add_annotation(x=x0 + 0.15 * (0 - x0), y=y1 - 0.10 * (y1 - 0), text="Improving", showarrow=False,
                       font=dict(color="rgba(0,0,255,0.75)", size=12))
    fig.add_annotation(x=x1 - 0.15 * (x1 - 0), y=y1 - 0.10 * (y1 - 0), text="Leading", showarrow=False,
                       font=dict(color="rgba(0,140,60,0.85)", size=12))
    fig.add_annotation(x=x0 + 0.15 * (0 - x0), y=y0 + 0.10 * (0 - y0), text="Lagging", showarrow=False,
                       font=dict(color="rgba(180,0,0,0.75)", size=12))
    fig.add_annotation(x=x1 - 0.15 * (x1 - 0), y=y0 + 0.10 * (0 - y0), text="Weakening", showarrow=False,
                       font=dict(color="rgba(200,120,0,0.85)", size=12))

    # plot each ticker tail as ONE connected trace (so it always connects head/tail)
    for sym, g in rrg_df.groupby("Ticker"):
        g = g.sort_values("Date").tail(tail_len)
        if len(g) < 2:
            continue

        row = meta[meta["Ticker"] == sym]
        name = sym if row.empty else row.iloc[0]["Name"]
        color = "#1f77b4" if row.empty else row.iloc[0]["_COLORHEX"]

        # tail line + small markers
        fig.add_trace(
            go.Scatter(
                x=g["RS_Ratio"],
                y=g["RS_Momentum"],
                mode="lines+markers",
                name=f"{sym} ({name})",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
                hovertemplate=(
                    f"<b>{sym}</b> ({name})<br>"
                    "Date=%{customdata}<br>"
                    "RS-Ratio=%{x:.2f}<br>"
                    "RS-Momentum=%{y:.2f}<extra></extra>"
                ),
                customdata=g["Date"].dt.strftime("%Y-%m-%d"),
                showlegend=False,  # keep legend clean; we show “Most recent point” legend separately
            )
        )

        # head marker (most recent point) — diamond, larger, black outline
        head = g.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[head["RS_Ratio"]],
                y=[head["RS_Momentum"]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    size=14,
                    color=color,
                    line=dict(color="black", width=2),
                ),
                name=f"{sym} ({name})",
                hovertemplate=(
                    f"<b>{sym}</b> ({name})<br>"
                    "Most recent<br>"
                    "Date=%{customdata}<br>"
                    "RS-Ratio=%{x:.2f}<br>"
                    "RS-Momentum=%{y:.2f}<extra></extra>"
                ),
                customdata=[pd.to_datetime(head["Date"]).strftime("%Y-%m-%d")],
                showlegend=True,
            )
        )

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
        legend_title_text="Most recent point",
        xaxis=dict(title="RS-Ratio (standardized)", range=[x0, x1], zeroline=False),
        yaxis=dict(title="RS-Momentum (standardized)", range=[y0, y1], zeroline=False),
    )
    return fig


# =========================
# Snapshot tables (with color swatches)
# =========================
def assign_colors(meta: pd.DataFrame) -> pd.DataFrame:
    # stable color assignment based on ticker order
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    meta = meta.copy()
    colors = []
    for i, _ in enumerate(meta["Ticker"].tolist()):
        colors.append(palette[i % len(palette)])
    meta["_COLORHEX"] = colors
    return meta


def compute_snapshot(rrg_df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sym, g in rrg_df.groupby("Ticker"):
        g = g.sort_values("Date")
        if len(g) < 2:
            continue
        last = g.iloc[-1]
        prev = g.iloc[-2]
        dx = float(last["RS_Ratio"] - prev["RS_Ratio"])
        dy = float(last["RS_Momentum"] - prev["RS_Momentum"])
        spd = float(math.sqrt(dx * dx + dy * dy))

        rows.append(
            {
                "Ticker": sym,
                "RS_Ratio_val": float(last["RS_Ratio"]),
                "RS_Momentum_val": float(last["RS_Momentum"]),
                "dx": dx,
                "dy": dy,
                "RotationSpeedRaw": spd,
            }
        )

    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap

    # percentiles for speed buckets
    snap["SpeedPct"] = snap["RotationSpeedRaw"].rank(pct=True)

    snap["RS-Ratio"] = snap["RS_Ratio_val"].apply(classify_rs_ratio)
    snap["Momentum"] = snap["RS_Momentum_val"].apply(classify_momentum)
    snap["Direction"] = [angle_to_arrow(dx, dy) for dx, dy in zip(snap["dx"], snap["dy"])]
    snap["Rotation Speed"] = snap["SpeedPct"].apply(speed_bucket_from_percentile)
    snap["Quadrant"] = [quadrant(x, y) for x, y in zip(snap["RS_Ratio_val"], snap["RS_Momentum_val"])]

    snap = snap.merge(meta[["Ticker", "Name", "Group", "_COLORHEX"]], on="Ticker", how="left")
    snap["Name"] = snap["Name"].fillna(snap["Ticker"])
    snap["Group"] = snap["Group"].fillna("")

    # Display-only: keep swatch, hide _COLORHEX later
    return snap


def render_color_table(df: pd.DataFrame, hide_cols: list[str] | None = None):
    if df is None or df.empty:
        st.info("No snapshot rows to display.")
        return

    hide_cols = hide_cols or []
    show = df.copy()

    # build visible “Color” swatch from _COLORHEX
    def swatch(hexcode: str) -> str:
        return f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;background:{hexcode};border:1px solid rgba(0,0,0,0.25)'></span>"

    show.insert(0, "Color", show["_COLORHEX"].fillna("#999999").apply(swatch))

    for c in hide_cols:
        if c in show.columns:
            show = show.drop(columns=[c])

    # also hide internal columns if present
    for c in ["_COLORHEX", "RS_Ratio_val", "RS_Momentum_val", "dx", "dy", "RotationSpeedRaw", "SpeedPct"]:
        if c in show.columns:
            show = show.drop(columns=[c])

    html = (
        show.to_html(escape=False, index=False)
        .replace("<table", "<table style='width:100%; border-collapse:collapse;'")
        .replace("<th>", "<th style='text-align:left; padding:8px; border-bottom:1px solid #eee; font-size:12px;'>")
        .replace("<td>", "<td style='padding:8px; border-bottom:1px solid #f3f3f3; font-size:12px;'>")
    )
    st.markdown(html, unsafe_allow_html=True)


# =========================
# UI
# =========================
st.title("Relative Rotation Graph (RRG)")
st.caption("Track sector, theme, commodity, and country rotation vs a benchmark. Approximation of JdK RS-Ratio and RS-Momentum.")

universes = predefined_universes()

with st.sidebar:
    st.header("RRG Settings")

    universe_name = st.selectbox("Universe", list(universes.keys()), index=0)
    default_benchmark = "SPY"
    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)

    timeframe = st.radio("Timeframe", ["Daily", "Weekly"], index=1)

    # Enforce your rule:
    # - Daily uses 1 year
    # - Weekly uses 3 years
    fixed_years = 1 if timeframe == "Daily" else 3
    st.caption(f"History: {fixed_years} year(s) (enforced)")

    lookback_weeks = st.slider("Lookback (weeks)", min_value=8, max_value=104, value=52, step=1)
    momentum_weeks = st.slider("Momentum (weeks)", min_value=4, max_value=52, value=13, step=1)
    tail_len = st.slider("Tail length (weeks)", min_value=4, max_value=30, value=13, step=1)

    st.markdown("---")
    st.subheader("Manage Universe")

    st.download_button(
        "Download CSV template",
        data=template_csv_bytes(),
        file_name="rrg_universe_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded = st.file_uploader("Upload Universe (CSV or XLSX)", type=["csv", "xlsx", "xls"])

    use_uploaded = st.checkbox("Use uploaded universe", value=False)

    extra_tickers_raw = st.text_input("Extra tickers (comma-separated)", value="", placeholder="e.g. QQQ, IWM, HYG")

    colA, colB = st.columns(2)
    with colA:
        clear_cache = st.button("Refresh / Clear cache", use_container_width=True)
    with colB:
        pass

if clear_cache:
    st.cache_data.clear()
    st.rerun()

# Parse uploaded file (if any)
uploaded_df = pd.DataFrame(columns=["Group", "Ticker", "Name"])
upload_error = None
if uploaded is not None:
    try:
        uploaded_df = parse_uploaded_universe(uploaded)
    except Exception as e:
        upload_error = str(e)

if upload_error:
    st.error(f"Upload error: {upload_error}")

# Build active universe table
base_universe = universes[universe_name]
if use_uploaded and not uploaded_df.empty:
    active_universe = merge_universes(base_universe, uploaded_df)
else:
    active_universe = base_universe.copy()

# Filter active_universe to current selected Universe group label:
# (we keep Group column for display; the Universe drop-down is the “view”)
active_universe["Group"] = active_universe["Group"].fillna(universe_name)

# Allow choose ETFs
choices = (
    active_universe.assign(Label=lambda d: d["Ticker"] + " — " + d["Name"])
    .sort_values("Label")["Label"]
    .tolist()
)
default_labels = (
    base_universe.assign(Label=lambda d: d["Ticker"] + " — " + d["Name"])
    .sort_values("Label")["Label"]
    .tolist()
)

with st.sidebar:
    selected_labels = st.multiselect(
        "Choose ETFs",
        options=choices,
        default=default_labels[: min(11, len(default_labels))],
    )

# Convert labels to tickers
label_to_ticker = (
    active_universe.assign(Label=lambda d: d["Ticker"] + " — " + d["Name"])
    .set_index("Label")["Ticker"]
    .to_dict()
)

selected_tickers = [label_to_ticker[lbl] for lbl in selected_labels if lbl in label_to_ticker]
extra_tickers = [t.strip().upper() for t in extra_tickers_raw.split(",") if t.strip()]
tickers_all = list(dict.fromkeys([benchmark] + selected_tickers + extra_tickers))

# Build meta for colors & names
meta = active_universe[active_universe["Ticker"].isin(selected_tickers)].copy()
# add extra tickers (if any) as self-named
for t in extra_tickers:
    if t not in meta["Ticker"].tolist():
        meta = pd.concat(
            [meta, pd.DataFrame([{"Group": "Extra", "Ticker": t, "Name": t}])],
            ignore_index=True,
        )

meta = meta.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)
meta = assign_colors(meta)

# Data period/interval
if timeframe == "Daily":
    period = "1y"
    interval = "1d"
else:
    period = "3y"
    interval = "1d"  # download daily then resample for consistent handling

# Download prices
try:
    close_daily = yf_download_close(tuple(tickers_all), period=period, interval=interval)
    if close_daily.empty:
        st.warning("No data returned from Yahoo Finance. Try fewer symbols or refresh.")
        st.stop()

    if timeframe == "Weekly":
        prices = to_weekly(close_daily)
    else:
        prices = close_daily

    # Ensure benchmark exists
    if benchmark not in prices.columns:
        st.error(f"Benchmark {benchmark} did not return price data from Yahoo Finance.")
        st.stop()

except Exception as e:
    st.error(f"Error downloading data from Yahoo Finance: {e}")
    st.stop()

# Compute RRG data
rrg = compute_rrg_components(prices, benchmark=benchmark, lookback_weeks=lookback_weeks, momentum_weeks=momentum_weeks)

if rrg.empty:
    st.warning("Not enough data to build RRG (try shorter lookback / momentum window or remove very new ETFs).")
    st.stop()

# Title
title = f"{universe_name} vs {benchmark} ({timeframe})"

# Plot
fig = make_rrg_figure(rrg, meta=meta, tail_len=tail_len, title=title)
st.plotly_chart(fig, use_container_width=True)

# Snapshot tables
st.subheader("Latest RRG Snapshot (interpreted)")

snap = compute_snapshot(rrg, meta)
if snap.empty:
    st.info("No snapshot available (need at least 2 points per symbol).")
    st.stop()

# Top 3 Leading & Top 3 Improving
leading = snap[snap["Quadrant"] == "Leading"].copy()
improving = snap[snap["Quadrant"] == "Improving"].copy()

# rank: Leading by RS_Ratio_val descending; Improving by RS_Ratio_val descending (closest to 0 is stronger)
leading = leading.sort_values(["RS_Ratio_val", "RS_Momentum_val"], ascending=[False, False]).head(3)
improving = improving.sort_values(["RS_Ratio_val", "RS_Momentum_val"], ascending=[False, False]).head(3)

left, right = st.columns(2)

with left:
    st.markdown("**Top 3 Leading**")
    render_color_table(
        leading[
            ["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "_COLORHEX"]
        ],
        hide_cols=["_COLORHEX"],
    )

with right:
    st.markdown("**Top 3 Improving**")
    render_color_table(
        improving[
            ["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "_COLORHEX"]
        ],
        hide_cols=["_COLORHEX"],
    )

# Full universe snapshot (expander)
with st.expander("Universe Snapshot Table (all selected symbols)", expanded=True):
    render_color_table(
        snap[
            ["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant", "_COLORHEX"]
        ],
        hide_cols=["_COLORHEX"],
    )

# Upload status help
if uploaded is not None:
    if use_uploaded and not uploaded_df.empty and not upload_error:
        st.success("Uploaded universe is active. Your file is being used (uploaded tickers take precedence).")
    elif not use_uploaded and not upload_error:
        st.info("File uploaded, but 'Use uploaded universe' is OFF. Turn it on to apply your file.")
