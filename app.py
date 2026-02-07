import math
from io import StringIO
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go


# ----------------------------
# Predefined universes
# ----------------------------
SPDR_SECTORS = [
    ("XLB", "Materials"),
    ("XLC", "Communication Services"),
    ("XLE", "Energy"),
    ("XLF", "Financials"),
    ("XLI", "Industrials"),
    ("XLK", "Technology"),
    ("XLP", "Consumer Staples"),
    ("XLRE", "Real Estate"),
    ("XLU", "Utilities"),
    ("XLV", "Health Care"),
    ("XLY", "Consumer Discretionary"),
]

THEMES = [
    ("SOXX", "Semiconductors"),
    ("SMH", "Semiconductors 2"),
    ("CIBR", "Cybersecurity"),
    ("HACK", "Cybersecurity 2"),
    ("CLOU", "Cloud"),
    ("SKYY", "Cloud 2"),
    ("IGV", "Software"),
    ("ITA", "Defense & Aerospace"),
    ("XAR", "Defense & Aerospace 2"),
    ("EUAD", "European Defense"),
    ("ICLN", "Clean Energy"),
    ("TAN", "Solar"),
    ("ARKF", "Fintech / Innovation"),
    ("PAVE", "Infrastructure"),
    ("DTCR", "Digital Infrastructure"),
    ("TCAI", "Digital Infrastructure 2"),
    ("STCE", "Bitcoin Mining / HPC"),
    ("WGMI", "Bitcoin Mining / HPC 2"),
    ("ITB", "Home Construction"),
    ("BOIL", "Natural Gas"),
    ("XOP", "Natural Gas 2"),
    ("ROBO", "Robotics"),
    ("BOTZ", "Robotics 2"),
    ("NLR", "Nuclear"),
    ("NUKZ", "Nuclear 2"),
    ("ARKG", "Biotech"),
    ("BIB", "Biotech 2"),
    ("PPH", "Pharmaceutical"),
    ("JEDI", "Drone"),
    ("ARKQ", "Drone 2"),
    ("RTH", "Brokerage"),
    ("IAI", "Brokerage 2"),
    ("XRT", "Retail Shopping"),
    ("PUI", "Utilities"),
    ("UFO", "Space"),
    ("KRE", "Regional Banking"),
    ("KBE", "Banking"),
    ("JETS", "Airlines"),
    ("REMX", "Rare Earth"),
    ("QTUM", "Quantum"),
    ("MSOS", "Cannabis"),
]

COMMODITIES = [
    ("RING", "Gold Miners"),
    ("IAU", "Gold"),
    ("SLV", "Silver"),
    ("SIL", "Silver Miners"),
    ("COPX", "Copper Miners"),
    ("USO", "Oil"),
    ("BTC-USD", "Bitcoin"),
    ("ETH-USD", "Ethereum"),
    ("BSOL", "Solana (proxy)"),
]

COUNTRIES = [
    ("VEU", "All World ex US"),
    ("EMXC", "Emerging Mkts ex-China"),
    ("EM", "Emerging Markets"),
    ("EWC", "Canada"),
    ("EWG", "Germany"),
    ("EWS", "Singapore"),
    ("EWZ", "Brazil"),
    ("MCHI", "China"),
    ("KWEB", "China Internet"),
]

UNIVERSES = {
    "SPDR Sectors": {"group": "Sectors", "items": SPDR_SECTORS},
    "Themes ETFs": {"group": "Themes", "items": THEMES},
    "Commodity ETFs": {"group": "Commodities", "items": COMMODITIES},
    "Country ETFs": {"group": "Countries", "items": COUNTRIES},
}


# ----------------------------
# Helpers
# ----------------------------
def make_template_csv() -> bytes:
    template = (
        "Group,Ticker,Name,Include\n"
        "Sectors,XLB,Materials,TRUE\n"
        "Themes,SMH,Semiconductors,TRUE\n"
        "Commodities,GLD,Gold,TRUE\n"
        "Countries,EWC,Canada,TRUE\n"
        "Countries,MCHI,China,TRUE\n"
    )
    return template.encode("utf-8")


def _to_bool(x, default=True):
    if pd.isna(x):
        return default
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    return default


def parse_uploaded_csv(file) -> pd.DataFrame:
    """
    Accepts CSV with columns:
      - Group, Ticker, Name (required)
      - Include (optional, default TRUE)
    """
    if file is None:
        return pd.DataFrame(columns=["Group", "Ticker", "Name", "Include"])

    raw = file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(raw))

    cols = {c.strip().lower(): c for c in df.columns}
    required = ["group", "ticker", "name"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Required columns are Group,Ticker,Name (Include optional).")

    out = df[[cols["group"], cols["ticker"], cols["name"]]].copy()
    out.columns = ["Group", "Ticker", "Name"]

    if "include" in cols:
        out["Include"] = df[cols["include"]].apply(_to_bool)
    else:
        out["Include"] = True

    out["Group"] = out["Group"].astype(str).str.strip()
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    out["Name"] = out["Name"].astype(str).str.strip()

    out = out.dropna(subset=["Ticker"])
    out = out[out["Ticker"].str.len() > 0]
    return out


def build_predefined_df() -> pd.DataFrame:
    rows = []
    for uni_name, u in UNIVERSES.items():
        group = u["group"]
        for t, nm in u["items"]:
            rows.append({"Group": group, "Ticker": t.upper().strip(), "Name": nm, "Include": True, "Source": "predefined"})
    df = pd.DataFrame(rows).drop_duplicates(subset=["Ticker"], keep="first")  # Option A: first wins
    return df


def merge_predefined_with_csv(predef: pd.DataFrame, csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Option A: predefined wins on duplicates.
    - Keep predefined rows
    - Add CSV rows only when ticker not already present
    """
    if csv_df is None or csv_df.empty:
        return predef.copy()

    predef_t = set(predef["Ticker"].tolist())
    add = csv_df[~csv_df["Ticker"].isin(predef_t)].copy()
    add["Source"] = "csv"
    merged = pd.concat([predef, add], ignore_index=True)
    # If CSV provides Include=False, it will still be honored (for the added ones).
    return merged


@st.cache_data(show_spinner=False, ttl=60 * 30)
def download_prices(tickers, years: int, interval: str = "1d") -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    period = f"{int(years)}y"
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return df


def extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    t = ticker.upper().strip()

    if isinstance(df.columns, pd.MultiIndex):
        if t in df.columns.get_level_values(0):
            sub = df[t]
            if "Close" in sub.columns:
                return sub["Close"].rename(t)
            if "Adj Close" in sub.columns:
                return sub["Adj Close"].rename(t)

        # try reverse style
        if "Close" in df.columns.get_level_values(0):
            try:
                return df["Close"][t].rename(t)
            except Exception:
                pass
        if "Adj Close" in df.columns.get_level_values(0):
            try:
                return df["Adj Close"][t].rename(t)
            except Exception:
                pass
        return pd.Series(dtype=float)

    if "Close" in df.columns:
        return df["Close"].rename(t)
    if "Adj Close" in df.columns:
        return df["Adj Close"].rename(t)
    return pd.Series(dtype=float)


def to_weekly(close: pd.Series) -> pd.Series:
    if close is None or close.empty:
        return close
    close = close.dropna()
    if close.empty:
        return close
    return close.resample("W-FRI").last().dropna()


def zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 3)).std(ddof=0)
    return (s - m) / sd


def compute_rrg_series(price: pd.Series, benchmark: pd.Series, lookback: int, momentum: int) -> pd.DataFrame:
    df = pd.DataFrame({"price": price, "bench": benchmark}).dropna()
    if df.empty:
        return pd.DataFrame()

    rs = df["price"] / df["bench"]
    ratio = rs / rs.rolling(lookback, min_periods=max(10, lookback // 3)).mean()
    rs_ratio = zscore(ratio, lookback)
    mom_raw = rs_ratio.diff(momentum)
    rs_mom = zscore(mom_raw, lookback)

    out = pd.DataFrame({"rs_ratio": rs_ratio, "rs_mom": rs_mom}).dropna()
    return out


def angle_to_arrow(dx: float, dy: float) -> str:
    if not np.isfinite(dx) or not np.isfinite(dy):
        return "•"
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "•"
    ang = math.degrees(math.atan2(dy, dx))
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
    return "↘"


def slope_label(v: float, thr: float = 0.15) -> str:
    if not np.isfinite(v):
        return "Flat"
    if v > thr:
        return "Improving"
    if v < -thr:
        return "Weakening"
    return "Flat"


def momentum_label(v: float, thr: float = 0.15) -> str:
    if not np.isfinite(v):
        return "Flat"
    if v > thr:
        return "Rising"
    if v < -thr:
        return "Falling"
    return "Flat"


def speed_bucket_from_percentile(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 25:
        return "Slow"
    if p < 50:
        return "Medium"
    if p < 80:
        return "Fast"
    return "Hot/Climactic"


def make_rrg_figure(tails: dict, title: str) -> go.Figure:
    fig = go.Figure()

    xs, ys = [], []
    for _, df in tails.items():
        if df is None or df.empty:
            continue
        xs.extend(df["rs_ratio"].values.tolist())
        ys.extend(df["rs_mom"].values.tolist())

    if len(xs) == 0:
        fig.update_layout(title=title)
        return fig

    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)

    xpad = max(0.5, (xmax - xmin) * 0.15)
    ypad = max(0.5, (ymax - ymin) * 0.15)
    x0, x1 = xmin - xpad, xmax + xpad
    y0, y1 = ymin - ypad, ymax + ypad

    quad_opacity = 0.10
    fig.add_shape(type="rect", x0=x0, x1=0, y0=0, y1=y1, fillcolor="rgba(90,120,255,1)", opacity=quad_opacity, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=x1, y0=0, y1=y1, fillcolor="rgba(70,200,120,1)", opacity=quad_opacity, line_width=0)
    fig.add_shape(type="rect", x0=x0, x1=0, y0=y0, y1=0, fillcolor="rgba(255,120,120,1)", opacity=quad_opacity, line_width=0)
    fig.add_shape(type="rect", x0=0, x1=x1, y0=y0, y1=0, fillcolor="rgba(255,210,80,1)", opacity=quad_opacity, line_width=0)

    fig.add_shape(type="line", x0=x0, x1=x1, y0=0, y1=0, line=dict(width=1, color="rgba(0,0,0,0.35)"))
    fig.add_shape(type="line", x0=0, x1=0, y0=y0, y1=y1, line=dict(width=1, color="rgba(0,0,0,0.35)"))

    fig.add_annotation(x=x0 + 0.02 * (x1 - x0), y=y1 - 0.02 * (y1 - y0), text="Improving", showarrow=False, font=dict(size=12, color="rgba(0,0,160,0.8)"))
    fig.add_annotation(x=x1 - 0.02 * (x1 - x0), y=y1 - 0.02 * (y1 - y0), text="Leading", showarrow=False, xanchor="right", font=dict(size=12, color="rgba(0,120,0,0.8)"))
    fig.add_annotation(x=x0 + 0.02 * (x1 - x0), y=y0 + 0.02 * (y1 - y0), text="Lagging", showarrow=False, yanchor="bottom", font=dict(size=12, color="rgba(160,0,0,0.8)"))
    fig.add_annotation(x=x1 - 0.02 * (x1 - x0), y=y0 + 0.02 * (y1 - y0), text="Weakening", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12, color="rgba(180,120,0,0.9)"))

    for t, df in tails.items():
        if df is None or df.empty:
            continue
        df = df.dropna()
        if df.empty:
            continue

        x = df["rs_ratio"].values
        y = df["rs_mom"].values

        # Tail connects to head because this includes all points
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=t,
                line=dict(width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{t}</b><br>RS-Ratio=%{{x:.2f}}<br>RS-Mom=%{{y:.2f}}<extra></extra>",
                showlegend=False,
            )
        )

        # Head marker
        fig.add_trace(
            go.Scatter(
                x=[x[-1]],
                y=[y[-1]],
                mode="markers",
                name=t,
                marker=dict(symbol="diamond", size=16, line=dict(width=2, color="black")),
                hovertemplate=f"<b>{t} (latest)</b><br>RS-Ratio=%{{x:.2f}}<br>RS-Mom=%{{y:.2f}}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        title=title,
        height=560,
        margin=dict(l=30, r=30, t=60, b=40),
        xaxis=dict(title="RS-Ratio (standardized)", range=[x0, x1], zeroline=False),
        yaxis=dict(title="RS-Momentum (standardized)", range=[y0, y1], zeroline=False),
    )
    return fig


# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")
st.title("Relative Rotation Graph (RRG)")
st.caption("Track sector, theme, commodity, and country rotation vs a benchmark. Approximation of JdK RS-Ratio and RS-Momentum.")


# --- Sidebar: Upload + Manage Universe ---
with st.sidebar:
    st.header("RRG Settings")

    st.subheader("Manage Universe (persistent via CSV)")

    # 1) Optional: Managed universe CSV (overrides everything)
    managed_file = st.file_uploader("Upload Managed Universe CSV (Group,Ticker,Name,Include)", type=["csv"], key="managed_csv")
    st.download_button("Download managed CSV template", data=make_template_csv(), file_name="rrg_managed_template.csv", mime="text/csv")

    # 2) Optional: Add-on CSV (only adds tickers not in predefined) — used when no managed CSV
    st.caption("Optional: Add-on CSV (Group,Ticker,Name[,Include]) — used only if no managed CSV is uploaded.")
    addon_file = st.file_uploader("Upload Add-on CSV", type=["csv"], key="addon_csv")

    # Build baseline
    predef_df = build_predefined_df()

    if managed_file is not None:
        try:
            managed_df = parse_uploaded_csv(managed_file)
            managed_df["Source"] = "managed"
            universe_df = managed_df.copy()
        except Exception as e:
            st.error(str(e))
            universe_df = predef_df.copy()
    else:
        try:
            addon_df = parse_uploaded_csv(addon_file) if addon_file is not None else pd.DataFrame(columns=["Group", "Ticker", "Name", "Include"])
        except Exception as e:
            st.error(str(e))
            addon_df = pd.DataFrame(columns=["Group", "Ticker", "Name", "Include"])
        universe_df = merge_predefined_with_csv(predef_df, addon_df)

    # Normalize / de-dupe tickers (first wins)
    universe_df["Ticker"] = universe_df["Ticker"].astype(str).str.upper().str.strip()
    universe_df["Group"] = universe_df["Group"].astype(str).str.strip()
    universe_df["Name"] = universe_df["Name"].astype(str).str.strip()
    universe_df["Include"] = universe_df["Include"].apply(_to_bool)
    universe_df = universe_df.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)

    # Editable grid (this is your Manage Universe UI)
    with st.expander("Open Manage Universe Editor", expanded=False):
        st.caption("Tip: Uncheck Include to remove a ticker from the app. Edit Group/Name to reorganize.")
        edited = st.data_editor(
            universe_df[["Group", "Ticker", "Name", "Include"]],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="universe_editor",
        )

        # Save into session + export
        edited = edited.copy()
        edited["Ticker"] = edited["Ticker"].astype(str).str.upper().str.strip()
        edited["Group"] = edited["Group"].astype(str).str.strip()
        edited["Name"] = edited["Name"].astype(str).str.strip()
        edited["Include"] = edited["Include"].apply(_to_bool)
        edited = edited.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)

        st.download_button(
            "Download Managed Universe CSV (save this!)",
            data=edited.to_csv(index=False).encode("utf-8"),
            file_name="rrg_managed_universe.csv",
            mime="text/csv",
        )

        universe_df = edited  # use edited version for the rest of the app

    # --- RRG Settings ---
    universe_name = st.selectbox("Universe (group filter)", list(UNIVERSES.keys()), index=0)
    bench = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], index=0)

    st.subheader("Timeframe")
    tf = st.radio(" ", ["Daily", "Weekly"], index=0)

    default_years = 1 if tf == "Daily" else 3
    years = st.slider("History (years, daily data)", min_value=1, max_value=10, value=default_years, step=1)

    lookback_weeks = st.slider("Lookback (weeks)", min_value=12, max_value=104, value=52, step=1)
    momentum_weeks = st.slider("Momentum (weeks)", min_value=4, max_value=52, value=13, step=1)
    tail_weeks = st.slider("Tail length (weeks)", min_value=4, max_value=52, value=13, step=1)

    # Filter tickers shown in picker by the selected universe group label
    universe_group = UNIVERSES[universe_name]["group"]

    # Candidates: Included only
    base = universe_df[universe_df["Include"] == True].copy()
    # If user picks SPDR Sectors, show only Group == "Sectors" by default
    candidates = base[base["Group"].astype(str).str.strip().str.lower() == universe_group.lower()].copy()

    # fall back if no matches
    if candidates.empty:
        candidates = base.copy()

    candidates["Label"] = candidates["Name"].astype(str) + " (" + candidates["Ticker"].astype(str) + ")"
    label_to_ticker = dict(zip(candidates["Label"], candidates["Ticker"]))
    ticker_to_label = dict(zip(candidates["Ticker"], candidates["Label"]))

    default_tickers = candidates["Ticker"].head(min(12, len(candidates))).tolist()
    default_labels = [ticker_to_label[t] for t in default_tickers if t in ticker_to_label]

    st.subheader("Choose ETFs")
    selected_labels = st.multiselect(" ", options=candidates["Label"].tolist(), default=default_labels)
    selected_tickers = [label_to_ticker[l] for l in selected_labels]

    extra = st.text_input("Extra tickers (comma-separated)", value="")
    extra_tickers = [x.strip().upper() for x in extra.split(",") if x.strip()] if extra.strip() else []

    # De-dupe
    tickers = []
    seen = set()
    for t in [*selected_tickers, *extra_tickers]:
        if t not in seen:
            seen.add(t)
            tickers.append(t)


# ----------------------------
# Main: build RRG
# ----------------------------
if not tickers:
    st.warning("Select at least one ETF/ticker.")
    st.stop()

need_points_weeks = lookback_weeks + momentum_weeks + tail_weeks + 10

all_symbols = sorted(set([bench] + tickers))

try:
    raw = download_prices(all_symbols, years=years, interval="1d")
except Exception as e:
    st.error(f"Error downloading data from Yahoo Finance: {e}")
    st.stop()

close_map = {}
missing = []
for sym in all_symbols:
    s = extract_close_series(raw, sym).dropna()
    if s.empty:
        missing.append(sym)
        continue
    close_map[sym] = s

if bench not in close_map:
    st.error(f"Benchmark '{bench}' has no data.")
    if missing:
        st.caption(f"Missing: {missing}")
    st.stop()

bench_close = close_map[bench]
price_series = {t: close_map[t] for t in tickers if t in close_map}

if tf == "Weekly":
    bench_close = to_weekly(bench_close)
    price_series = {t: to_weekly(s) for t, s in price_series.items()}

if tf == "Daily":
    lookback_n = int(lookback_weeks * 5)
    momentum_n = int(momentum_weeks * 5)
    tail_n = int(tail_weeks * 5)
else:
    lookback_n = int(lookback_weeks)
    momentum_n = int(momentum_weeks)
    tail_n = int(tail_weeks)

tails = {}
dropped = []
latest_snapshot = []

for t, p in price_series.items():
    series = pd.DataFrame({"p": p, "b": bench_close}).dropna()
    if series.shape[0] < max(lookback_n + momentum_n + tail_n + 20, need_points_weeks):
        dropped.append((t, series.shape[0]))
        continue

    out = compute_rrg_series(series["p"], series["b"], lookback=lookback_n, momentum=momentum_n)
    if out.empty:
        dropped.append((t, series.shape[0]))
        continue

    tail_df = out.tail(tail_n).dropna()
    if tail_df.shape[0] < max(10, min(30, tail_n // 2)):
        dropped.append((t, tail_df.shape[0]))
        continue

    tails[t] = tail_df

    x_last = float(tail_df["rs_ratio"].iloc[-1])
    y_last = float(tail_df["rs_mom"].iloc[-1])

    if tail_df.shape[0] >= 2:
        dx = float(tail_df["rs_ratio"].iloc[-1] - tail_df["rs_ratio"].iloc[-2])
        dy = float(tail_df["rs_mom"].iloc[-1] - tail_df["rs_mom"].iloc[-2])
    else:
        dx, dy = np.nan, np.nan

    latest_snapshot.append({"Ticker": t, "RS_Ratio": x_last, "RS_Momentum": y_last, "dx": dx, "dy": dy})

snap = pd.DataFrame(latest_snapshot)

if dropped:
    with st.sidebar:
        st.warning("Some symbols were dropped due to insufficient data:")
        for t, n in dropped[:40]:
            st.caption(f"- {t} (usable points: {n})")

if not tails:
    st.warning("Not enough data to build RRG (try shorter windows or remove very new ETFs).")
    st.stop()

# Speed buckets
snap["speed"] = np.sqrt(snap["dx"] ** 2 + snap["dy"] ** 2)
snap["speed_pct"] = snap["speed"].rank(pct=True) * 100.0

# Meta from universe_df (edited)
meta_map = {r["Ticker"]: {"Name": r["Name"], "Group": r["Group"]} for _, r in universe_df.iterrows()}
snap["Name"] = [meta_map.get(t, {"Name": t}).get("Name", t) for t in snap["Ticker"].tolist()]
snap["Group"] = [meta_map.get(t, {"Group": "Custom"}).get("Group", "Custom") for t in snap["Ticker"].tolist()]

# Interpreted fields
snap["RS-Ratio"] = snap["dx"].apply(lambda v: slope_label(v, thr=0.15))
snap["Momentum"] = snap["dy"].apply(lambda v: momentum_label(v, thr=0.15))
snap["Direction"] = [angle_to_arrow(dx, dy) for dx, dy in zip(snap["dx"], snap["dy"])]
snap["Rotation Speed"] = snap["speed_pct"].apply(speed_bucket_from_percentile)

# Quadrants
snap["Quadrant"] = np.where(
    (snap["RS_Ratio"] >= 0) & (snap["RS_Momentum"] >= 0),
    "Leading",
    np.where(
        (snap["RS_Ratio"] >= 0) & (snap["RS_Momentum"] < 0),
        "Weakening",
        np.where((snap["RS_Ratio"] < 0) & (snap["RS_Momentum"] < 0), "Lagging", "Improving"),
    ),
)

tf_label = "Daily" if tf == "Daily" else "Weekly"
fig_title = f"{universe_name} vs {bench} ({tf_label})"
fig = make_rrg_figure(tails, title=fig_title)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Latest RRG Snapshot (interpreted)")

colA, colB = st.columns(2)

with colA:
    st.subheader("Top 3 Leading")
    lead = snap[snap["Quadrant"] == "Leading"].copy()
    lead = lead.sort_values(["RS_Ratio", "speed_pct"], ascending=[False, False]).head(3)
    if lead.empty:
        st.caption("No tickers currently in Leading quadrant.")
    else:
        st.dataframe(
            lead[["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]],
            use_container_width=True,
            hide_index=True,
        )

with colB:
    st.subheader("Top 3 Improving")
    imp = snap[snap["Quadrant"] == "Improving"].copy()
    imp = imp.sort_values(["RS_Momentum", "speed_pct"], ascending=[False, False]).head(3)
    if imp.empty:
        st.caption("No tickers currently in Improving quadrant.")
    else:
        st.dataframe(
            imp[["Ticker", "Name", "Group", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]],
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Full Snapshot Table (all selected tickers)"):
    full = snap.copy().sort_values(["Quadrant", "speed_pct"], ascending=[True, False])
    show = full[["Ticker", "Name", "Group", "Quadrant", "RS-Ratio", "Momentum", "Direction", "Rotation Speed"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

st.caption(
    "Notes: RS-Ratio/Momentum labels are based on the last-step slope (dx/dy). "
    "Rotation Speed is bucketed by percentile of latest movement magnitude among displayed tickers."
)
