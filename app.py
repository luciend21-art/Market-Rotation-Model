def _read_universe_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """
    Expected columns:
      Group, Ticker, Name
    Name optional. Group and Ticker required.
    """
    if uploaded_file is None:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"]), ""

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            try:
                import openpyxl  # noqa: F401
            except Exception:
                return pd.DataFrame(columns=["Group", "Ticker", "Name"]), (
                    "Missing optional dependency 'openpyxl'. "
                    "Add openpyxl to requirements.txt for Excel uploads, or upload CSV instead."
                )
            df = pd.read_excel(uploaded_file)
        else:
            return pd.DataFrame(columns=["Group", "Ticker", "Name"]), "Unsupported upload type. Use CSV or XLSX."
    except Exception as e:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"]), f"Could not read uploaded file: {e}"

    df.columns = [str(c).strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in col_map:
                return col_map[n]
        return None

    c_group = pick("group", "universe", "category")
    c_ticker = pick("ticker", "symbol")
    c_name = pick("name", "description", "label")

    if c_group is None or c_ticker is None:
        return pd.DataFrame(columns=["Group", "Ticker", "Name"]), (
            "Upload must include columns: Group and Ticker. Name is optional."
        )

    out = pd.DataFrame()
    out["Group"] = df[c_group].astype(str).str.strip()
    out["Ticker"] = df[c_ticker].astype(str).str.strip().str.upper()

    if c_name is not None:
        out["Name"] = df[c_name]
    else:
        out["Name"] = np.nan

    # normalize blanks
    out["Group"] = out["Group"].replace("", np.nan)
    out["Ticker"] = out["Ticker"].replace("", np.nan)

    out = out.dropna(subset=["Group", "Ticker"]).copy()
    out = out.drop_duplicates(subset=["Ticker"], keep="first").reset_index(drop=True)

    # clean names after dropping bad rows
    out["Name"] = out["Name"].astype("string").str.strip()
    out["Name"] = out["Name"].where(out["Name"].notna() & (out["Name"] != ""), out["Ticker"])

    return out[["Group", "Ticker", "Name"]], ""
