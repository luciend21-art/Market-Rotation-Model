import math
from io import BytesIO
from typing import Dict, List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="Relative Rotation Graph (RRG)",
    layout="wide",
)

# ----------------------------
# Predefined universes (baseline)
# ----------------------------
def predefined_universes() -> Dict[str, pd.DataFrame]:
    # Columns: Ticker, Name, Group
    sectors = [
        ("XLB", "Materials", "SPDR Sectors"),
        ("XLC", "Communication Services", "SPDR Sectors"),
        ("XLE", "Energy", "SPDR Sectors"),
        ("XLF", "Financials", "SPDR Sectors"),
        ("XLI", "Industrials", "SPDR Sectors"),
        ("XLK", "Technology", "SPDR Sectors"),
        ("XLP", "Consumer Staples", "SPDR Sectors"),
        ("XLRE", "Real Estate", "SPDR Sectors"),
        ("XLU", "Utilities", "SPDR Sectors"),
        ("XLV", "Health Care", "SPDR Sectors"),
        ("XLY", "Consumer Discretionary", "SPDR Sectors"),
    ]

    # You can keep these minimal because you’ll upload your own file over time
    themes = [
        ("SOXX", "Semiconductors", "Themes"),
        ("SMH", "Semiconductors 2", "Themes"),
        ("CIBR", "Cybersecurity", "Themes"),
        ("HACK", "Cybersecurity 2", "Themes"),
        ("CLOU", "Cloud", "Themes"),
        ("SKYY", "Cloud 2", "Themes"),
        ("IGV", "Software", "Themes"),
        ("ITA", "Defense & Aerospace", "Themes"),
        ("XAR", "Defense & Aerospace 2", "Themes"),
        ("ICLN", "Clean Energy", "Themes"),
        ("TAN", "Solar", "Themes"),
        ("ARKF", "Fintech / Innovation", "Themes"),
        ("PAVE", "Infrastructure", "Themes"),
        ("ROBO", "Robotics", "Themes"),
        ("BOTZ", "Robotics 2", "Themes"),
        ("NLR", "Nuclear", "Themes"),
        ("NUKZ", "Nuclear 2", "Themes"),
        ("ARKG", "Biotech", "Themes"),
        ("BIB", "Biotech 2", "Themes"),
        ("PPH", "Pharmaceutical", "Themes"),
        ("JETS", "Airlines", "Themes"),
        ("REMX", "Rare Earth", "Themes"),
        ("QTUM", "Quantum", "Themes"),
        ("MSOS", "Cannabis", "Themes"),
    ]

    commodities = [
        ("RING", "Gold Miners", "Commodities"),
        ("IAU", "Gold", "Commodities"),
        ("SLV", "Silver", "Commodities"),
        ("COPX", "Copper Miners", "Commodities"),
        ("USO", "Oil", "Commodities"),
    ]

    countries = [
        ("VEU", "All-World ex US", "Countries"),
        ("EMXC", "Emerging Markets ex China", "Countries"),
        ("EEM", "Emerging Markets", "Countries"),
        ("MCHI", "China", "Countries"),
        ("KWEB", "China Internet", "Countries"),
        ("EWZ", "Brazil", "Countries"),
        ("EWG", "Germany", "Countries"),
        ("EWC", "Canada", "Countries"),
        ("EWS", "Singapore", "Countries"),
    ]

    def df(rows):
        out = pd.DataFrame(rows, columns=["Ticker", "Name", "Group"])
        out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
        out["Name"] = out["Name"].astype(str).str.strip()
        out["Group"] = out["Group"].astype(str).str.strip()
        return out

    return {
        "SPDR Sectors": df(sectors),
        "Themes ETFs": df(themes),
        "Commodity ETFs": df(commodities),
        "Country ETFs": df(countries),
    }


# ----------------------------
# Upload parsing
# ----------------------------
def sample_universe_template() -> pd.DataFrame:
    # Group is optional but recommended since you want organization.
    return pd.DataFrame(
        [
            {"Ticker": "XLK", "Name": "Technology", "Group": "SPDR Sectors"},
            {"Ticker": "SOXX", "Name": "Semiconductors", "Group": "Themes"},
            {"Ticker": "IAU", "Name": "Gold", "Group": "Commodities"},
            {"Ticker": "EWG", "Name": "Germany", "Group": "Countries"},
        ]
    )


def read_uploaded_universe(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df, error_message). error_message="" if ok.
    Required column: Ticker
    Optional columns: Name, Group
    """
    if uploaded_file is None:
        return pd.DataFrame(), ""

    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            try:
                import openpyxl  # noqa: F401
            except Exception:
                return pd.DataFrame(), "Missing optional dependency 'openpyxl'. Add openpyxl to requirements.txt or upload a CSV instead."
            df = pd.read_excel(uploaded_file)
        else:
            return pd.DataFrame(), "Unsupported file type. Upload CSV or XLSX."
    except Exception as e:
        return pd.DataFrame(), f"Could not read file: {e}"

    if df.empty:
        return pd.DataFrame(), "Uploaded file is empty."

    # Normalize columns
    cols = {c.strip(): c for c in df.columns}
    ticker_col = None
    for c in df.columns:
        if str(c).strip().lower() == "ticker":
            ticker_col = c
            break
    if ticker_col is None:
        return pd.DataFrame(), "Upload file must include a 'Ticker' column."

    out = pd.DataFrame()
    out["Ticker"] = df[ticker_col].astype(str).str.upper().str.strip()

    # Optional columns
    name_col = None
    group_col = None
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "name":
            name_col = c
        if cl == "group":
            group_col = c

    out["Name"] = df[name_col].astype(str).str.strip() if name_col else ""
    out["Group"] = df[group_col].astype(str).str.strip() if group_col else ""

    # Drop blanks / junk
    out = out.replace({"Ticker": {"": np.nan, "NAN": np.nan, "NONE": np.nan}})
    out = out.dropna(subset=["Ticker"])
    out = out[out["Ticker"].str.len() > 0]
    out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)

    return out, ""


def merge_universe(base: pd.DataFrame, uploaded: pd.DataFrame) -> pd.DataFrame:
    """
    Option A: first group wins (base wins). Dedupe by ticker.
    New tickers from upload are appended.
    If upload has Name/Group missing, fill where possible.
    """
    if uploaded is None or uploaded.empty:
        return base.copy()

    base2 = base.copy()
    up2 = uploaded.copy()

    base2["Ticker"] = base2["Ticker"].astype(str).str.upper().str.strip()
    up2["Ticker"] = up2["Ticker"].astype(str).str.upper().str.strip()

    # Append only tickers not already present
    existing = set(base2["Ticker"].tolist())
    add = up2[~up2["Ticker"].isin(existing)].copy()

    # If Group missing in add, default to "Uploaded"
    if "Group" not in add.columns:
        add["Group"] = "Uploaded"
    add["Group"] = add["Group"].replace("", "Uploaded")

    # If Name missing in add, set to Ticker
    if "Name" not in add.columns:
        add["Name"] = add["Ticker"]
    add["Name"] = add["Name"].replace("", add["Ticker"])

    out = pd.concat([base2, add[["Ticker", "Name", "Group"]]], ignore_index=True)
    out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)

    return out


# ----------------------------
# Data fetch & transforms
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_prices(tickers: List[str], period_years: int) -> pd.DataFrame:
    """
    Returns daily Adj Close (or Close) for tickers in columns.
    Robust to yfinance multiindex output.
    """
    tickers = [t.upper().strip() for t in tickers if t and isinstance(t, str)]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    period = f"{int(period_years)}y"

    df = yf.download(
        tickers=tickers,
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # If single ticker, columns are single level
    if isinstance(df.columns, pd.Index) and ("Close" in df.columns or "Adj Close" in df.columns):
        px = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
        px = px.to_frame(name=tickers[0])
        px.index = pd.to_datetime(px.index)
        return px.dropna(how="all")

    # MultiIndex: (field, ticker) or (ticker, field) depending on version
    if isinstance(df.columns, pd.MultiIndex):
        # Try common shapes
        # Shape A: first level is OHLC field
        if "Adj Close" in df.columns.get_level_values(0) or "Close" in df.columns.get_level_values(0):
            field = "Adj Close" if "Adj Close" in df.columns.get_level_values(0) else "Close"
            px = df[field].copy()
            px.index = pd.to_datetime(px.index)
            px = px.dropna(how="all")
            # px columns are tickers
            return px

        # Shape B: second level is OHLC field
        if "Adj Close" in df.columns.get_level_values(1) or "Close" in df.columns.get_level_values(1):
            field = "Adj Close" if "Adj Close" in df.columns.get_level_values(1) else "Close"
            px = df.xs(field, level=1, axis=1).copy()
            px.index = pd.to_datetime(px.index)
            px = px.dropna(how="all")
            return px

    return pd.DataFrame()


def to_weekly(px_daily: pd.DataFrame) -> pd.DataFrame:
    if px_daily.empty:
        return px_daily
    # Weekly last (Friday anchor helps consistency)
    wk = px_daily.resample("W-FRI").last()
    wk = wk.dropna(how="all")
    return wk


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(5, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z


def compute_rrg(prices: pd.DataFrame, benchmark: str, lookback: int, momentum: int) -> Dict[str, pd.DataFrame]:
    """
    Build RRG series for each symbol:
    - RS = (symbol / benchmark) * 100
    - RS-Ratio ~ rolling z-score of RS
    - RS-Momentum ~ rolling z-score of change in RS-Ratio
    Returns dict symbol -> DataFrame(index=date, columns=["rs_ratio","rs_mom"])
    """
    if prices.empty or benchmark not in prices.columns:
        return {}

    bench = prices[benchmark].dropna()
    out = {}

    for sym in prices.columns:
        if sym == benchmark:
            continue
        s = prices[sym].dropna()
        df = pd.concat([s, bench], axis=1, join="inner")
        df.columns = ["sym", "bench"]
        df = df.dropna()
        if len(df) < (lookback + momentum + 5):
            continue

        rs = (df["sym"] / df["bench"]) * 100.0

        # RS-Ratio: z-score of RS over lookback
        rs_ratio = rolling_zscore(rs, lookback)

        # RS-Momentum: z-score of delta in RS-Ratio over momentum window
        rs_ratio_delta = rs_ratio - rs_ratio.shift(momentum)
        rs_mom = rolling_zscore(rs_ratio_delta, lookback)

        z = pd.DataFrame({"rs_ratio": rs_ratio, "rs_mom": rs_mom}, index=df.index).dropna()
        if not z.empty:
            out[sym] = z

    return out


# ----------------------------
# Interpretation helpers
# ----------------------------
def direction_arrow(dx: float, dy: float) -> str:
    if dx == 0 and dy == 0:
        return "•"
    ang = math.degrees(math.atan2(dy, dx))  # -180..180
    # Map to 8 sectors
    if -22.5 <= ang < 22.5:
        return "→"
    if 22.5 <= ang < 67.5:
        return "↗"
    if 67.5 <= ang < 112.5:
        return "↑"
    if 112.5 <= ang < 157.5:
        return "↖"
    if ang >= 157.5 or ang < -157.5:
        return "←"
    if -157.5 <= ang < -112.5:
        return "↙"
    if -112.5 <= ang < -67.5:
        return "↓"
    if -67.5 <= ang < -22.5:
        return "↘"
    return "•"


def label_change(val: float, prev: float, thr: float, down_label: str) -> str:
    d = val - prev
    if d > thr:
        return "Improving"
    if d < -thr:
        return down_label
    return "Flat"


def quadrant(x: float, y: float) -> str:
    if x >= 0 and y >= 0:
        return "Leading"
    if x < 0 and y >= 0:
        return "Improving"
    if x < 0 and y < 0:
        return "Lagging"
    return "Weakening"


def build_snapshot(
    rrg_series: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per symbol with latest coordinates and interpreted fields.
    """
    rows = []
    for sym, df in rrg_series.items():
        if df.shape[0] < 2:
            continue
        x = float(df["rs_ratio"].iloc[-1])
        y = float(df["rs_mom"].iloc[-1])
        x0 = float(df["rs_ratio"].iloc[-2])
        y0 = float(df["rs_mom"].iloc[-2])
        dx = x - x0
        dy = y - y0
        spd = math.sqrt(dx * dx + dy * dy)

        m = meta[meta["Ticker"] == sym]
        nm = m["Name"].iloc[0] if not m.empty else sym
        grp = m["Group"].iloc[0] if not m.empty else "—"

        rows.append(
            {
                "Ticker": sym,
                "Name": nm,
                "Group": grp,
                "_x": x,
                "_y": y,
                "_dx": dx,
                "_dy": dy,
                "_speed": spd,
                "Quadrant": quadrant(x, y),
            }
        )
    snap = pd.DataFrame(rows)
    if snap.empty:
        return snap

    # thresholds for "Flat" detection based on distribution
    # Use robust threshold = median absolute delta / 2 (bounded)
    dx_thr = float(np.clip(np.nanmedian(np.abs(snap["_dx"])) / 2.0, 0.03, 0.15))
    dy_thr = float(np.clip(np.nanmedian(np.abs(snap["_dy"])) / 2.0, 0.03, 0.15))

    snap["RS-Ratio"] = snap.apply(lambda r: label_change(r["_x"], r["_x"] - r["_dx"], dx_thr, "Weakening"), axis=1)
    snap["Momentum"] = snap.apply(lambda r: label_change(r["_y"], r["_y"] - r["_dy"], dy_thr, "Falling"), axis=1)
    snap["Direction"] = snap.apply(lambda r: direction_arrow(r["_dx"], r["_dy"]), axis=1)

    # Speed buckets via percentiles (4 buckets)
    p25, p50, p75 = np.nanpercentile(snap["_speed"], [25, 50, 75])
    def speed_bucket(v: float) -> str:
        if v <= p25:
            return "Slow"
        if v <= p50:
            return "Medium"
        if v <= p75:
            return "Fast"
        return "Hot/Climactic"
    snap["Rotation Speed"] = snap["_speed"].apply(speed_bucket)

    return snap


# ----------------------------
# Plotting
# ----------------------------
def color_palette(n: int) -> List[str]:
    # A bright, readable palette. (No need for seaborn.)
    base = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
        "#0b3d91", "#ff4d4d", "#00a878", "#ff70e6",
        "#00c2ff", "#ffa600",
    ]
    if n <= len(base):
        return base[:n]
    # Repeat if needed
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out


def make_rrg_figure(
    rrg_series: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    tail_len: int,
    title: str,
    auto_zoom: bool = True,
    fixed_range: float = 4.0,
) -> Tuple[alt.Chart, pd.DataFrame]:
    """
    Returns (chart, color_map_df)
    color_map_df: Ticker, Name, Group, ColorHex
    """
    if not rrg_series:
        return alt.Chart(pd.DataFrame()), pd.DataFrame()

    symbols = list(rrg_series.keys())
    colors = color_palette(len(symbols))
    cmap = {sym: colors[i] for i, sym in enumerate(symbols)}

    meta2 = meta.copy()
    meta2["Ticker"] = meta2["Ticker"].str.upper()

    # Build long dataframe of tail points
    pts = []
    heads = []
    for sym, df in rrg_series.items():
        d = df.copy()
        if d.empty:
            continue
        d = d.tail(max(3, tail_len))
        d = d.reset_index().rename(columns={"index": "Date"})
        d["Ticker"] = sym
        d["ColorHex"] = cmap[sym]
        pts.append(d)

        # Head is last point
        last = d.iloc[-1].copy()
        heads.append(last)

    points = pd.concat(pts, ignore_index=True)
    head_df = pd.DataFrame(heads)

    # Join names/groups
    join = meta2[["Ticker", "Name", "Group"]].drop_duplicates()
    points = points.merge(join, on="Ticker", how="left")
    head_df = head_df.merge(join, on="Ticker", how="left")

    # Axis handling
    if auto_zoom:
        xmin = float(np.nanmin(points["rs_ratio"]))
        xmax = float(np.nanmax(points["rs_ratio"]))
        ymin = float(np.nanmin(points["rs_mom"]))
        ymax = float(np.nanmax(points["rs_mom"]))
        pad_x = max(0.4, (xmax - xmin) * 0.10)
        pad_y = max(0.4, (ymax - ymin) * 0.10)
        # Force include 0 and quadrant boundaries for readability
        xmin = min(xmin - pad_x, -0.2)
        xmax = max(xmax + pad_x, 0.2)
        ymin = min(ymin - pad_y, -0.2)
        ymax = max(ymax + pad_y, 0.2)
    else:
        xmin, xmax = -fixed_range, fixed_range
        ymin, ymax = -fixed_range, fixed_range

    # Quadrant shading backgrounds
    bg = pd.DataFrame(
        [
            {"x0": xmin, "x1": 0, "y0": 0, "y1": ymax, "q": "Improving", "c": "#eef2ff"},
            {"x0": 0, "x1": xmax, "y0": 0, "y1": ymax, "q": "Leading", "c": "#ecfdf5"},
            {"x0": xmin, "x1": 0, "y0": ymin, "y1": 0, "q": "Lagging", "c": "#fef2f2"},
            {"x0": 0, "x1": xmax, "y0": ymin, "y1": 0, "q": "Weakening", "c": "#fff7ed"},
        ]
    )

    base = alt.Chart(points).properties(
        width="container",
        height=420,
        title=title,
    )

    quad = (
        alt.Chart(bg)
        .mark_rect(opacity=0.55)
        .encode(
            x=alt.X("x0:Q", scale=alt.Scale(domain=[xmin, xmax]), title="RS-Ratio (standardized)"),
            x2="x1:Q",
            y=alt.Y("y0:Q", scale=alt.Scale(domain=[ymin, ymax]), title="RS-Momentum (standardized)"),
            y2="y1:Q",
            color=alt.Color("c:N", scale=None, legend=None),
        )
    )

    # Axes cross lines
    xline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#2d2d2d", opacity=0.6).encode(x="x:Q")
    yline = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#2d2d2d", opacity=0.6).encode(y="y:Q")

    # Tail path lines (connect by Date per Ticker)
    line = (
        base.mark_line(point=False)
        .encode(
            x="rs_ratio:Q",
            y="rs_mom:Q",
            detail="Ticker:N",
            color=alt.Color("Ticker:N", scale=alt.Scale(domain=symbols, range=[cmap[s] for s in symbols]), legend=None),
            order=alt.Order("Date:T", sort="ascending"),
            tooltip=[
                alt.Tooltip("Ticker:N"),
                alt.Tooltip("Name:N"),
                alt.Tooltip("Group:N"),
                alt.Tooltip("Date:T"),
                alt.Tooltip("rs_ratio:Q", title="RS-Ratio", format=".2f"),
                alt.Tooltip("rs_mom:Q", title="RS-Mom", format=".2f"),
            ],
        )
    )

    dots = (
        base.mark_circle(size=24, opacity=0.95)
        .encode(
            x="rs_ratio:Q",
            y="rs_mom:Q",
            color=alt.Color("Ticker:N", scale=alt.Scale(domain=symbols, range=[cmap[s] for s in symbols]), legend=None),
            order=alt.Order("Date:T", sort="ascending"),
        )
    )

    # Head markers (big diamond w/ black outline)
    head = (
        alt.Chart(head_df)
        .mark_point(shape="diamond", size=180, filled=True, stroke="black", strokeWidth=1.6)
        .encode(
            x=alt.X("rs_ratio:Q", scale=alt.Scale(domain=[xmin, xmax]), title="RS-Ratio (standardized)"),
            y=alt.Y("rs_mom:Q", scale=alt.Scale(domain=[ymin, ymax]), title="RS-Momentum (standardized)"),
            color=alt.Color("Ticker:N", scale=alt.Scale(domain=symbols, range=[cmap[s] for s in symbols]), legend=None),
            tooltip=[
                alt.Tooltip("Ticker:N"),
                alt.Tooltip("Name:N"),
                alt.Tooltip("Group:N"),
                alt.Tooltip("rs_ratio:Q", title="RS-Ratio", format=".2f"),
                alt.Tooltip("rs_mom:Q", title="RS-Mom", format=".2f"),
            ],
        )
    )

    # Corner quadrant labels
    labels = pd.DataFrame(
        [
            {"x": xmin + (xmax - xmin) * 0.03, "y": ymax - (ymax - ymin) * 0.06, "t": "Improving", "c": "#1d4ed8"},
            {"x": xmax - (xmax - xmin) * 0.12, "y": ymax - (ymax - ymin) * 0.06, "t": "Leading", "c": "#047857"},
            {"x": xmin + (xmax - xmin) * 0.03, "y": ymin + (ymax - ymin) * 0.08, "t": "Lagging", "c": "#b91c1c"},
            {"x": xmax - (xmax - xmin) * 0.16, "y": ymin + (ymax - ymin) * 0.08, "t": "Weakening", "c": "#c2410c"},
        ]
    )
    qtext = (
        alt.Chart(labels)
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="t:N", color=alt.Color("c:N", scale=None, legend=None))
    )

    chart = (quad + xline + yline + line + dots + head + qtext).configure_title(
        fontSize=16, anchor="start"
    )

    color_map_df = meta2.merge(
        pd.DataFrame({"Ticker": list(cmap.keys()), "ColorHex": list(cmap.values())}),
        on="Ticker",
        how="right",
    )[["Ticker", "Name", "Group", "ColorHex"]]

    return chart, color_map_df


# ----------------------------
# Table rendering with swatches
# ----------------------------
def style_color_swatch(df: pd.DataFrame, hex_col: str = "ColorHex") -> pd.io.formats.style.Styler:
    """
    Create a 'Color' column with a swatch, use ColorHex for background.
    """
    d = df.copy()
    d["Color"] = "■"
    if hex_col not in d.columns:
        d[hex_col] = "#444444"

    # Order columns (no _COLORHEX)
    show_cols = [c for c in ["Color", "Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant"] if c in d.columns]
    d = d[show_cols + ([hex_col] if hex_col in d.columns else [])]

    def color_cells(col):
        if col.name != "Color":
            return [""] * len(col)
        # Use underlying ColorHex
        colors = df[hex_col].tolist() if hex_col in df.columns else ["#444444"] * len(df)
        return [f"background-color: {h}; color: {h};" for h in colors]  # hide the ■ text by matching foreground

    sty = d.style.apply(color_cells, axis=0)
    # Hide the ColorHex helper column if present
    if hex_col in d.columns:
        sty = sty.hide(columns=[hex_col])
    return sty


# ----------------------------
# UI
# ----------------------------
st.title("Relative Rotation Graph (RRG)")
st.caption("Track sector/theme/commodity/country rotation vs a benchmark. Approximation of JdK RS-Ratio and RS-Momentum.")

universes = predefined_universes()

with st.sidebar:
    st.header("RRG Settings")
    universe_name = st.selectbox("Universe", list(universes.keys()), index=0)

    benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)

    timeframe = st.radio("Timeframe", ["Daily", "Weekly"], index=1)
    if timeframe == "Daily":
        period_years = 1
        st.caption("History: 1 year(s) (enforced)")
    else:
        period_years = 3
        st.caption("History: 3 year(s) (enforced)")

    lookback = st.slider("Lookback (weeks)", 12, 104, 52, step=1)
    momentum = st.slider("Momentum (weeks)", 4, 26, 13, step=1)
    tail_len = st.slider("Tail length (weeks)", 4, 26, 13, step=1)

    st.divider()
    st.subheader("Manage Universe")

    # Template download
    tmpl = sample_universe_template()
    buf = BytesIO()
    tmpl.to_csv(buf, index=False)
    st.download_button(
        "Download CSV template",
        data=buf.getvalue(),
        file_name="rrg_universe_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Universe (CSV or XLSX)", type=["csv", "xlsx", "xls"])
    use_upload = st.checkbox("Use uploaded universe (merge with selected universe)", value=True, disabled=(uploaded is None))

# Build base universe meta
base_meta = universes[universe_name].copy()

# Read upload and merge
upload_df, upload_err = read_uploaded_universe(uploaded)
if upload_err:
    st.error(f"Upload error: {upload_err}")

if uploaded is not None and use_upload and upload_err == "":
    meta = merge_universe(base_meta, upload_df)
    meta["Group"] = meta["Group"].replace("", universe_name)
    source_note = f"Using **{universe_name}** merged with uploaded file."
else:
    meta = base_meta
    source_note = f"Using **{universe_name}**."

st.markdown(source_note)

# ETF selector
# Keep selectable list to manage chaos if user uploads huge universes
options = [f"{r['Ticker']} ({r['Name']})" for _, r in meta.iterrows()]
label_to_ticker = {f"{r['Ticker']} ({r['Name']})": r["Ticker"] for _, r in meta.iterrows()}

default_count = min(10, len(options))
default_sel = options[:default_count]

selected_labels = st.sidebar.multiselect("Choose ETFs", options, default=default_sel)
selected = [label_to_ticker[x] for x in selected_labels]
selected = [t for t in selected if t != benchmark]

if not selected:
    st.warning("Select at least 1 ETF to plot.")
    st.stop()

# Pull prices
tickers = sorted(list(dict.fromkeys(selected + [benchmark])))

with st.spinner("Downloading prices..."):
    px_daily = fetch_prices(tickers, period_years=period_years)

if px_daily.empty:
    st.error("No price data returned. Try fewer symbols or a different benchmark.")
    st.stop()

# Ensure all requested columns exist
missing = [t for t in tickers if t not in px_daily.columns]
if missing:
    st.warning(f"Dropped missing tickers (no daily data): {', '.join(missing)}")
    tickers = [t for t in tickers if t in px_daily.columns]
    px_daily = px_daily[tickers]

# Timeframe transform
if timeframe == "Weekly":
    px = to_weekly(px_daily)
else:
    # For "Daily" view, convert to weekly-like cadence for RRG calculations,
    # but based on daily history. This keeps RRG interpretable while still reacting faster.
    px = to_weekly(px_daily)

# Compute RRG series
rrg_series = compute_rrg(px, benchmark=benchmark, lookback=lookback, momentum=momentum)

# Warn dropped symbols for insufficient history
dropped = [t for t in selected if t not in rrg_series]
if dropped:
    st.warning(
        "Some symbols were dropped due to insufficient history for your settings: "
        + ", ".join(dropped)
    )

if not rrg_series:
    st.error("Not enough data to build RRG (try shorter lookback/momentum or fewer symbols).")
    st.stop()

# Limit to selected tickers that computed successfully
rrg_series = {k: v for k, v in rrg_series.items() if k in selected}

# Chart
auto_zoom = st.toggle("Auto-zoom chart", value=True, help="Auto-fit axes to plotted points to reduce clutter.")
title = f"{universe_name} vs {benchmark} ({timeframe})"

chart, color_map = make_rrg_figure(
    rrg_series=rrg_series,
    meta=meta,
    tail_len=tail_len,
    title=title,
    auto_zoom=auto_zoom,
    fixed_range=4.0,
)

st.altair_chart(chart, use_container_width=True)

# Snapshot tables
snap = build_snapshot(rrg_series, meta=meta)
if snap.empty:
    st.warning("Could not build snapshot table (not enough points).")
    st.stop()

# Merge colors onto snapshot
snap = snap.merge(color_map[["Ticker", "ColorHex"]], on="Ticker", how="left")
snap["ColorHex"] = snap["ColorHex"].fillna("#444444")

st.subheader("Latest RRG Snapshot (interpreted)")

# Top 3 Leading / Improving
leading = snap[snap["Quadrant"] == "Leading"].copy()
improving = snap[snap["Quadrant"] == "Improving"].copy()

leading["Score"] = leading["_x"] + leading["_y"]
improving["Score"] = improving["_y"] - improving["_x"].abs()

top_leading = leading.sort_values(["Score"], ascending=False).head(3)
top_improving = improving.sort_values(["Score"], ascending=False).head(3)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Top 3 Leading")
    if top_leading.empty:
        st.caption("No symbols currently in the Leading quadrant.")
    else:
        show = top_leading[["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant", "ColorHex"]].copy()
        st.dataframe(style_color_swatch(show), use_container_width=True, hide_index=True)

with c2:
    st.markdown("### Top 3 Improving")
    if top_improving.empty:
        st.caption("No symbols currently in the Improving quadrant.")
    else:
        show = top_improving[["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant", "ColorHex"]].copy()
        st.dataframe(style_color_swatch(show), use_container_width=True, hide_index=True)

# Full universe snapshot
with st.expander("Universe Snapshot Table (all selected symbols)", expanded=True):
    full = snap.copy()
    # Display columns only (no internals)
    full_show = full[["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed", "Quadrant", "ColorHex"]].copy()
    st.dataframe(style_color_swatch(full_show), use_container_width=True, hide_index=True)

# Helpful note about XLSX
if uploaded is not None and uploaded.name.lower().endswith((".xlsx", ".xls")) and upload_err:
    st.info("Fix: add **openpyxl** to requirements.txt, redeploy, then try uploading XLSX again. CSV uploads work without it.")
