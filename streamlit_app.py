from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Finnhub Zone Opportunity Scanner", layout="wide")
st.title("Finnhub Zone Opportunity Scanner")
st.caption("Trend + support/resistance + supply/demand + pullback watch")

# =========================================================
# CONSTANTS
# =========================================================
FINNHUB_BASE = "https://finnhub.io/api/v1"
DEFAULT_WATCHLIST = "SPY, QQQ, IWM, AAPL, NVDA, AMD, TSLA, META"

# =========================================================
# SECRETS
# =========================================================
def get_secret(*names: str) -> str:
    for name in names:
        if name in st.secrets and st.secrets[name]:
            return str(st.secrets[name]).strip()
    return ""

FINNHUB_API_KEY = get_secret("FINNHUB_API_KEY", "FINNHUB_KEY", "API_KEY")

# =========================================================
# API HELPERS
# =========================================================
def finnhub_get(path: str, params: Optional[dict] = None) -> dict:
    if not FINNHUB_API_KEY:
        raise ValueError("Missing FINNHUB_API_KEY in Streamlit secrets.")

    p = dict(params or {})
    p["token"] = FINNHUB_API_KEY

    url = f"{FINNHUB_BASE}{path}"
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

def get_candles(symbol: str, resolution: str = "D", bars: int = 260) -> pd.DataFrame:
    if resolution in {"5", "15", "30", "60"}:
        lookback_days = 120
    elif resolution == "W":
        lookback_days = 365 * 5
    else:
        lookback_days = 365 * 2

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=lookback_days)

    data = finnhub_get(
        "/stock/candle",
        {
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start_dt.timestamp()),
            "to": int(end_dt.timestamp()),
        },
    )

    if data.get("s") != "ok":
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(data["t"], unit="s", utc=True).tz_convert(None),
            "Open": pd.to_numeric(data["o"], errors="coerce"),
            "High": pd.to_numeric(data["h"], errors="coerce"),
            "Low": pd.to_numeric(data["l"], errors="coerce"),
            "Close": pd.to_numeric(data["c"], errors="coerce"),
            "Volume": pd.to_numeric(data["v"], errors="coerce"),
        }
    )

    df = df.dropna().set_index("Timestamp").sort_index().tail(bars)
    return df

def get_quote(symbol: str) -> Optional[dict]:
    try:
        data = finnhub_get("/quote", {"symbol": symbol})
        return {
            "current": data.get("c", np.nan),
            "high": data.get("h", np.nan),
            "low": data.get("l", np.nan),
            "open": data.get("o", np.nan),
            "prev_close": data.get("pc", np.nan),
            "timestamp": data.get("t"),
        }
    except Exception:
        return None

# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class Zone:
    low: float
    high: float
    kind: str   # demand / supply
    touches: int
    source_indices: List[int]

# =========================================================
# TECHNICALS
# =========================================================
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def detect_trend(df: pd.DataFrame, fast_len: int = 20, slow_len: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["EMA_FAST"] = ema(out["Close"], fast_len)
    out["EMA_SLOW"] = ema(out["Close"], slow_len)
    out["FAST_SLOPE"] = out["EMA_FAST"].diff()
    out["SLOW_SLOPE"] = out["EMA_SLOW"].diff()

    def label(row):
        if (
            row["Close"] > row["EMA_FAST"] > row["EMA_SLOW"]
            and row["FAST_SLOPE"] > 0
            and row["SLOW_SLOPE"] > 0
        ):
            return "Bullish"
        if (
            row["Close"] < row["EMA_FAST"] < row["EMA_SLOW"]
            and row["FAST_SLOPE"] < 0
            and row["SLOW_SLOPE"] < 0
        ):
            return "Bearish"
        return "Neutral"

    out["Trend"] = out.apply(label, axis=1)
    return out

# =========================================================
# PIVOTS / SUPPORT / RESISTANCE / ZONES
# =========================================================
def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i - left:i + right + 1]) and highs[i] > max(highs[i - left:i]):
            pivot_highs.append(i)

        if lows[i] == min(lows[i - left:i + right + 1]) and lows[i] < min(lows[i - left:i]):
            pivot_lows.append(i)

    return pivot_highs, pivot_lows

def build_raw_zones(
    df: pd.DataFrame,
    pivot_indices: List[int],
    zone_type: str,
    atr_values: pd.Series,
    width_atr_mult: float = 0.7,
) -> List[Zone]:
    zones = []

    for idx in pivot_indices:
        atr_here = atr_values.iloc[idx]
        if pd.isna(atr_here) or atr_here <= 0:
            continue

        price = df["High"].iloc[idx] if zone_type == "supply" else df["Low"].iloc[idx]
        half = atr_here * width_atr_mult / 2.0

        zones.append(
            Zone(
                low=float(price - half),
                high=float(price + half),
                kind=zone_type,
                touches=1,
                source_indices=[idx],
            )
        )

    return zones

def merge_zones(zones: List[Zone], overlap_threshold: float = 0.35) -> List[Zone]:
    if not zones:
        return []

    zones = sorted(zones, key=lambda z: (z.kind, z.low))
    merged = []

    for z in zones:
        if not merged or merged[-1].kind != z.kind:
            merged.append(z)
            continue

        prev = merged[-1]
        overlap_low = max(prev.low, z.low)
        overlap_high = min(prev.high, z.high)
        overlap = max(0.0, overlap_high - overlap_low)

        prev_size = prev.high - prev.low
        z_size = z.high - z.low
        min_size = max(min(prev_size, z_size), 1e-9)

        if overlap / min_size >= overlap_threshold:
            prev.low = min(prev.low, z.low)
            prev.high = max(prev.high, z.high)
            prev.touches += z.touches
            prev.source_indices.extend(z.source_indices)
        else:
            merged.append(z)

    return merged

def score_zones(df: pd.DataFrame, zones: List[Zone], lookback_bars: int = 150) -> List[Zone]:
    recent = df.tail(lookback_bars)
    scored = []

    for z in zones:
        touches = 0
        for _, row in recent.iterrows():
            if row["High"] >= z.low and row["Low"] <= z.high:
                touches += 1
        z.touches = max(z.touches, touches)
        scored.append(z)

    return sorted(scored, key=lambda x: (x.kind, -x.touches))

def build_zones(df: pd.DataFrame, pivot_left: int, pivot_right: int, atr_width_mult: float) -> List[Zone]:
    ph, pl = find_pivots(df, left=pivot_left, right=pivot_right)

    raw_supply = build_raw_zones(df, ph, "supply", df["ATR"], atr_width_mult)
    raw_demand = build_raw_zones(df, pl, "demand", df["ATR"], atr_width_mult)

    supply = merge_zones(raw_supply)
    demand = merge_zones(raw_demand)

    return score_zones(df, supply + demand)

def zone_center(z: Zone) -> float:
    return (z.low + z.high) / 2.0

def nearest_zones(price: float, zones: List[Zone], kind: str, n: int = 5) -> List[Zone]:
    filtered = [z for z in zones if z.kind == kind]
    return sorted(filtered, key=lambda z: abs(zone_center(z) - price))[:n]

def price_in_zone(price: float, z: Zone) -> bool:
    return z.low <= price <= z.high

def zone_distance_pct(price: float, z: Zone) -> float:
    return abs(price - zone_center(z)) / max(price, 1e-9) * 100.0

# =========================================================
# OPPORTUNITY LOGIC
# =========================================================
def classify_setup(df: pd.DataFrame, zones: List[Zone], atr_mult_near: float = 0.8) -> dict:
    if len(df) < 60:
        return {"state": "Not enough data"}

    row = df.iloc[-1]
    close = float(row["Close"])
    low = float(row["Low"])
    high = float(row["High"])
    trend = row["Trend"]
    current_atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else 0.0
    ema_fast = float(row["EMA_FAST"])

    demand = nearest_zones(close, zones, "demand", n=5)
    supply = nearest_zones(close, zones, "supply", n=5)

    if trend == "Bullish":
        for z in demand:
            near_zone = low <= z.high + current_atr * atr_mult_near and close >= z.low - current_atr * atr_mult_near
            near_ema = abs(close - ema_fast) <= current_atr * atr_mult_near

            if price_in_zone(close, z):
                return {
                    "state": "Bullish Pullback In Demand",
                    "idea": "Look for bullish continuation if price holds demand.",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }
            if near_zone or near_ema:
                return {
                    "state": "Bullish Pullback Watch",
                    "idea": "Wait for reaction near demand / EMA, not a chase entry.",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }

        nearest_supply = nearest_zones(close, zones, "supply", n=1)
        if nearest_supply and zone_distance_pct(close, nearest_supply[0]) < 1.0:
            return {
                "state": "Bullish But Near Resistance",
                "idea": "Be careful. Price is getting close to overhead supply/resistance.",
                "zone_low": nearest_supply[0].low,
                "zone_high": nearest_supply[0].high,
            }

        return {"state": "Bullish Trend - No Pullback", "idea": "Wait for a pullback instead of chasing."}

    if trend == "Bearish":
        for z in supply:
            near_zone = high >= z.low - current_atr * atr_mult_near and close <= z.high + current_atr * atr_mult_near
            near_ema = abs(close - ema_fast) <= current_atr * atr_mult_near

            if price_in_zone(close, z):
                return {
                    "state": "Bearish Pullback In Supply",
                    "idea": "Look for bearish continuation if price rejects supply.",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }
            if near_zone or near_ema:
                return {
                    "state": "Bearish Pullback Watch",
                    "idea": "Wait for rejection near supply / EMA, not a chase entry.",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }

        nearest_demand = nearest_zones(close, zones, "demand", n=1)
        if nearest_demand and zone_distance_pct(close, nearest_demand[0]) < 1.0:
            return {
                "state": "Bearish But Near Support",
                "idea": "Be careful. Price is getting close to demand/support.",
                "zone_low": nearest_demand[0].low,
                "zone_high": nearest_demand[0].high,
            }

        return {"state": "Bearish Trend - No Pullback", "idea": "Wait for a pullback instead of chasing."}

    nearest_demand = nearest_zones(close, zones, "demand", n=1)
    nearest_supply = nearest_zones(close, zones, "supply", n=1)

    if nearest_demand and price_in_zone(close, nearest_demand[0]):
        return {"state": "Neutral In Demand", "idea": "Possible bounce area, but trend confirmation is weak."}
    if nearest_supply and price_in_zone(close, nearest_supply[0]):
        return {"state": "Neutral In Supply", "idea": "Possible rejection area, but trend confirmation is weak."}

    return {"state": "Neutral / No Setup", "idea": "No clean edge right now."}

# =========================================================
# CHART
# =========================================================
def make_chart(df: pd.DataFrame, zones: List[Zone], bars: int = 180) -> go.Figure:
    plot_df = df.tail(bars).copy()
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="Price",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["EMA_FAST"],
            mode="lines",
            name="EMA Fast",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df.index,
            y=plot_df["EMA_SLOW"],
            mode="lines",
            name="EMA Slow",
        )
    )

    recent_index = plot_df.index
    x0 = recent_index.min()
    x1 = recent_index.max()

    for z in zones:
        label = "Demand" if z.kind == "demand" else "Supply"
        fig.add_hrect(
            y0=z.low,
            y1=z.high,
            line_width=0,
            opacity=min(0.10 + z.touches * 0.01, 0.25),
            annotation_text=f"{label} ({z.touches})",
            annotation_position="top left",
        )

    fig.update_layout(
        height=760,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# =========================================================
# TABLE HELPERS
# =========================================================
def zone_table(zs: List[Zone], price: float) -> pd.DataFrame:
    rows = []
    for z in zs:
        rows.append(
            {
                "Type": z.kind,
                "Low": round(z.low, 2),
                "High": round(z.high, 2),
                "Center": round(zone_center(z), 2),
                "Touches": z.touches,
                "Distance %": round(zone_distance_pct(price, z), 2),
            }
        )
    return pd.DataFrame(rows)

# =========================================================
# UI
# =========================================================
a, b, c, d = st.columns([1.2, 1, 1, 1])

with a:
    symbol = st.text_input("Symbol", value="SPY").upper().strip()

with b:
    resolution = st.selectbox("Chart Resolution", ["D", "60", "30", "15", "5", "W"], index=0)

with c:
    fast_ema = st.number_input("Fast EMA", min_value=5, max_value=50, value=20, step=1)

with d:
    slow_ema = st.number_input("Slow EMA", min_value=10, max_value=100, value=50, step=1)

e, f, g = st.columns(3)
with e:
    pivot_left = st.number_input("Pivot Left", min_value=2, max_value=10, value=3, step=1)
with f:
    pivot_right = st.number_input("Pivot Right", min_value=2, max_value=10, value=3, step=1)
with g:
    atr_zone_mult = st.number_input("Zone Width ATR", min_value=0.2, max_value=2.0, value=0.7, step=0.1)

if not FINNHUB_API_KEY:
    st.error("Missing FINNHUB_API_KEY in Streamlit secrets.")
    st.stop()

# =========================================================
# LOAD DATA
# =========================================================
try:
    df = get_candles(symbol, resolution=resolution, bars=260)
except Exception as e:
    st.error(f"Failed to load Finnhub candles: {e}")
    st.stop()

if df.empty:
    st.error("No candle data returned.")
    st.stop()

df = detect_trend(df, int(fast_ema), int(slow_ema))
df["ATR"] = atr(df, 14)

zones = build_zones(
    df=df,
    pivot_left=int(pivot_left),
    pivot_right=int(pivot_right),
    atr_width_mult=float(atr_zone_mult),
)

setup = classify_setup(df, zones)

last = df.iloc[-1]
spot = float(last["Close"])
trend = str(last["Trend"])
atr_val = float(last["ATR"]) if pd.notna(last["ATR"]) else np.nan
quote = get_quote(symbol)

# =========================================================
# SUMMARY
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"{spot:.2f}")
m2.metric("Trend", trend)
m3.metric("ATR(14)", f"{atr_val:.2f}" if pd.notna(atr_val) else "n/a")
m4.metric("Setup", setup.get("state", "n/a"))

if quote and quote.get("timestamp"):
    try:
        qdt = datetime.utcfromtimestamp(int(quote["timestamp"]))
        st.caption(f"Finnhub quote timestamp (UTC): {qdt}")
    except Exception:
        pass

# =========================================================
# CHART
# =========================================================
st.plotly_chart(make_chart(df, zones), use_container_width=True)

# =========================================================
# READOUT
# =========================================================
st.subheader("Current Read")
st.write(
    f"""
**State:** {setup.get('state', 'n/a')}  
**Idea:** {setup.get('idea', 'n/a')}
"""
)

if "zone_low" in setup and "zone_high" in setup:
    st.write(f"**Relevant Zone:** {setup['zone_low']:.2f} to {setup['zone_high']:.2f}")

# =========================================================
# ZONES
# =========================================================
z1, z2 = st.columns(2)

with z1:
    st.markdown("### Demand / Support Areas")
    st.dataframe(zone_table(nearest_zones(spot, zones, "demand", n=5), spot), use_container_width=True)

with z2:
    st.markdown("### Supply / Resistance Areas")
    st.dataframe(zone_table(nearest_zones(spot, zones, "supply", n=5), spot), use_container_width=True)

# =========================================================
# WATCHLIST SCAN
# =========================================================
st.subheader("Watchlist Scanner")
watchlist_text = st.text_area("Symbols", value=DEFAULT_WATCHLIST, height=100)

if st.button("Scan Watchlist"):
    rows = []
    symbols = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]

    for sym in symbols:
        try:
            h = get_candles(sym, resolution="D", bars=220)
            if h.empty or len(h) < 80:
                continue

            h = detect_trend(h, int(fast_ema), int(slow_ema))
            h["ATR"] = atr(h, 14)
            z = build_zones(h, int(pivot_left), int(pivot_right), float(atr_zone_mult))
            s = classify_setup(h, z)
            last_row = h.iloc[-1]

            rows.append(
                {
                    "Symbol": sym,
                    "Close": round(float(last_row["Close"]), 2),
                    "Trend": str(last_row["Trend"]),
                    "Setup": s.get("state", "n/a"),
                    "Idea": s.get("idea", "n/a"),
                }
            )
        except Exception:
            continue

    scan_df = pd.DataFrame(rows)

    if scan_df.empty:
        st.info("No symbols scanned successfully.")
    else:
        priority = {
            "Bullish Pullback In Demand": 1,
            "Bearish Pullback In Supply": 2,
            "Bullish Pullback Watch": 3,
            "Bearish Pullback Watch": 4,
            "Bullish But Near Resistance": 5,
            "Bearish But Near Support": 6,
            "Bullish Trend - No Pullback": 7,
            "Bearish Trend - No Pullback": 8,
            "Neutral In Demand": 9,
            "Neutral In Supply": 10,
            "Neutral / No Setup": 11,
        }
        scan_df["Sort"] = scan_df["Setup"].map(priority).fillna(99)
        scan_df = scan_df.sort_values(["Sort", "Symbol"]).drop(columns=["Sort"])
        st.dataframe(scan_df, use_container_width=True)