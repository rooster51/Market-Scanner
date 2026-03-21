from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Zone Pullback Spread Scanner", layout="wide")
st.title("Zone Pullback Spread Scanner")
st.caption("Market Data candles + Market Data options")

# =========================================================
# CONSTANTS
# =========================================================
BASE_URL = "https://api.marketdata.app/v1"
DEFAULT_WATCHLIST = "SPY, QQQ, IWM, AAPL, NVDA, AMD, TSLA, META"

# =========================================================
# SECRETS / AUTH
# =========================================================
def get_api_key() -> str:
    for key in ["MARKETDATA_API_KEY", "MARKET_DATA_API_KEY", "API_KEY"]:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key]).strip()
    return ""

API_KEY = get_api_key()

def md_headers() -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }

def md_get(path: str, params: Optional[dict] = None) -> dict:
    if not API_KEY:
        raise ValueError("Missing MARKETDATA_API_KEY in Streamlit secrets.")

    url = f"{BASE_URL}{path}"
    r = requests.get(url, headers=md_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()

# =========================================================
# DATA STRUCTURES
# =========================================================
@dataclass
class Zone:
    low: float
    high: float
    kind: str
    touches: int
    source_indices: List[int]

# =========================================================
# HELPERS
# =========================================================
def to_series_df(data: dict) -> pd.DataFrame:
    """
    Convert a Market Data column-array style response into row-based DataFrame.
    """
    if not isinstance(data, dict):
        return pd.DataFrame()

    list_cols = [k for k, v in data.items() if isinstance(v, list)]
    if not list_cols:
        return pd.DataFrame()

    max_len = max(len(data[k]) for k in list_cols)
    rows = []
    for i in range(max_len):
        row = {}
        for k in list_cols:
            vals = data.get(k, [])
            row[k] = vals[i] if i < len(vals) else np.nan
        rows.append(row)

    return pd.DataFrame(rows)

# =========================================================
# MARKET DATA STOCKS
# =========================================================
def get_md_stock_candles(symbol: str, resolution: str = "D", bars: int = 260) -> pd.DataFrame:
    end_dt = date.today()

    if resolution in {"1", "5", "15", "30", "60"}:
        lookback_days = 120
    elif resolution == "W":
        lookback_days = 365 * 5
    else:
        lookback_days = 365 * 2

    start_dt = end_dt - timedelta(days=lookback_days)

    data = md_get(
        f"/stocks/candles/{resolution}/{symbol}/",
        params={
            "from": start_dt.isoformat(),
            "to": end_dt.isoformat(),
        },
    )

    df = to_series_df(data)
    if df.empty:
        return df

    rename_map = {
        "t": "Timestamp",
        "date": "Timestamp",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
    }
    df = df.rename(columns=rename_map)

    if "Timestamp" not in df.columns:
        return pd.DataFrame()

    # Handle either unix timestamps or date strings
    ts = pd.to_numeric(df["Timestamp"], errors="coerce")
    if ts.notna().sum() > 0:
        df["Timestamp"] = pd.to_datetime(ts, unit="s", errors="coerce")
    else:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.set_index("Timestamp").sort_index()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[needed].dropna().tail(bars)

def get_md_stock_quote(symbol: str) -> Optional[dict]:
    try:
        data = md_get(f"/stocks/quotes/{symbol}/")

        if isinstance(data.get("symbol"), list):
            return {
                "symbol": data.get("symbol", [None])[0],
                "last": data.get("last", [np.nan])[0] if "last" in data else np.nan,
                "bid": data.get("bid", [np.nan])[0] if "bid" in data else np.nan,
                "ask": data.get("ask", [np.nan])[0] if "ask" in data else np.nan,
                "mid": data.get("mid", [np.nan])[0] if "mid" in data else np.nan,
                "volume": data.get("volume", [np.nan])[0] if "volume" in data else np.nan,
                "updated": data.get("updated", [None])[0] if "updated" in data else None,
            }

        return data
    except Exception:
        return None

# =========================================================
# MARKET DATA OPTIONS
# =========================================================
def get_expirations(symbol: str) -> List[str]:
    try:
        data = md_get(f"/options/expirations/{symbol}/")

        if isinstance(data.get("expiration"), list):
            return [str(x) for x in data["expiration"]]

        df = to_series_df(data)
        if "expiration" in df.columns:
            return [str(x) for x in df["expiration"].dropna().tolist()]

        return []
    except Exception:
        return []

def choose_expiration(expirations: List[str], min_dte: int = 7, max_dte: int = 35) -> Optional[str]:
    today = date.today()
    ranked = []

    for exp in expirations:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (d - today).days
            if min_dte <= dte <= max_dte:
                ranked.append((abs(dte - 14), dte, exp))
        except Exception:
            continue

    if not ranked:
        return None

    ranked.sort()
    return ranked[0][2]

def get_option_chain(symbol: str, expiration: str) -> pd.DataFrame:
    try:
        data = md_get(
            f"/options/chain/{symbol}/",
            params={"expiration": expiration},
        )

        df = to_series_df(data)
        if df.empty:
            return df

        rename_map = {
            "optionSymbol": "contract_symbol",
            "underlying": "underlying",
            "expiration": "expiration",
            "side": "option_type",
            "strike": "strike",
            "bid": "bid",
            "ask": "ask",
            "mid": "mid",
            "last": "last",
            "volume": "volume",
            "openInterest": "open_interest",
            "delta": "delta",
            "gamma": "gamma",
            "theta": "theta",
            "vega": "vega",
            "rho": "rho",
            "iv": "iv",
            "updated": "updated",
        }
        df = df.rename(columns=rename_map)

        if "option_type" in df.columns:
            df["option_type"] = (
                df["option_type"]
                .astype(str)
                .str.lower()
                .replace({"call": "call", "put": "put", "c": "call", "p": "put"})
            )

        numeric_cols = [
            "strike", "bid", "ask", "mid", "last", "volume", "open_interest",
            "delta", "gamma", "theta", "vega", "rho", "iv"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    except Exception:
        return pd.DataFrame()

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
# ZONES
# =========================================================
def find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        if highs[i] == max(highs[i - left:i + right + 1]):
            if highs[i] > max(highs[i - left:i]):
                pivot_highs.append(i)

        if lows[i] == min(lows[i - left:i + right + 1]):
            if lows[i] < min(lows[i - left:i]):
                pivot_lows.append(i)

    return pivot_highs, pivot_lows

def build_raw_zones(
    df: pd.DataFrame,
    pivot_indices: List[int],
    zone_type: str,
    atr_values: pd.Series,
    width_atr_mult: float = 0.7,
) -> List[Zone]:
    zones: List[Zone] = []

    for idx in pivot_indices:
        atr_here = atr_values.iloc[idx]
        if pd.isna(atr_here) or atr_here <= 0:
            continue

        px = df["High"].iloc[idx] if zone_type == "supply" else df["Low"].iloc[idx]
        half = atr_here * width_atr_mult / 2.0

        zones.append(
            Zone(
                low=float(px - half),
                high=float(px + half),
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
    merged: List[Zone] = []

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

    supply = merge_zones(raw_supply, overlap_threshold=0.35)
    demand = merge_zones(raw_demand, overlap_threshold=0.35)

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
# PULLBACK SIGNAL
# =========================================================
def pullback_signal(df: pd.DataFrame, zones: List[Zone], atr_mult_near: float = 0.8) -> dict:
    if len(df) < 60:
        return {"state": "Not enough data"}

    row = df.iloc[-1]
    close = float(row["Close"])
    low = float(row["Low"])
    high = float(row["High"])
    current_trend = row["Trend"]
    current_atr = float(row["ATR"]) if not pd.isna(row["ATR"]) else 0.0
    ema_fast = float(row["EMA_FAST"])

    demand = nearest_zones(close, zones, "demand", n=5)
    supply = nearest_zones(close, zones, "supply", n=5)

    if current_trend == "Bullish":
        for z in demand:
            near_zone = low <= z.high + current_atr * atr_mult_near and close >= z.low - current_atr * atr_mult_near
            near_ema = abs(close - ema_fast) <= current_atr * atr_mult_near

            if price_in_zone(close, z):
                return {
                    "state": "Bullish Pullback In Zone",
                    "trend": current_trend,
                    "zone_type": "demand",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }
            if near_zone or near_ema:
                return {
                    "state": "Bullish Pullback Watch",
                    "trend": current_trend,
                    "zone_type": "demand",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }
        return {"state": "Bullish Trend - No Pullback", "trend": current_trend}

    if current_trend == "Bearish":
        for z in supply:
            near_zone = high >= z.low - current_atr * atr_mult_near and close <= z.high + current_atr * atr_mult_near
            near_ema = abs(close - ema_fast) <= current_atr * atr_mult_near

            if price_in_zone(close, z):
                return {
                    "state": "Bearish Pullback In Zone",
                    "trend": current_trend,
                    "zone_type": "supply",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }
            if near_zone or near_ema:
                return {
                    "state": "Bearish Pullback Watch",
                    "trend": current_trend,
                    "zone_type": "supply",
                    "zone_low": z.low,
                    "zone_high": z.high,
                }
        return {"state": "Bearish Trend - No Pullback", "trend": current_trend}

    return {"state": "Neutral / No Setup", "trend": current_trend}

# =========================================================
# OPTIONS HELPERS
# =========================================================
def safe_mid(bid, ask, last, mid):
    try:
        bid = float(bid) if pd.notna(bid) else np.nan
        ask = float(ask) if pd.notna(ask) else np.nan
        last = float(last) if pd.notna(last) else np.nan
        mid = float(mid) if pd.notna(mid) else np.nan
    except Exception:
        return np.nan

    if pd.notna(mid) and mid > 0:
        return mid
    if pd.notna(bid) and pd.notna(ask) and ask >= bid and ask > 0:
        return (bid + ask) / 2.0
    if pd.notna(last):
        return last
    return np.nan

def add_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["mid_calc"] = out.apply(
        lambda r: safe_mid(r.get("bid"), r.get("ask"), r.get("last"), r.get("mid")),
        axis=1,
    )
    return out

def filter_chain_for_puts(chain: pd.DataFrame) -> pd.DataFrame:
    out = chain.copy()
    out = out[out["option_type"] == "put"].copy()
    out = add_mid_price(out)
    out = out[pd.notna(out["strike"]) & pd.notna(out["mid_calc"])]
    return out

def filter_chain_for_calls(chain: pd.DataFrame) -> pd.DataFrame:
    out = chain.copy()
    out = out[out["option_type"] == "call"].copy()
    out = add_mid_price(out)
    out = out[pd.notna(out["strike"]) & pd.notna(out["mid_calc"])]
    return out

def build_vertical_spreads(short_df: pd.DataFrame, long_df: pd.DataFrame, spread_type: str) -> pd.DataFrame:
    rows = []

    short_df = short_df.sort_values("strike")
    long_df = long_df.sort_values("strike")

    short_records = short_df.to_dict("records")
    long_records = long_df.to_dict("records")

    for s in short_records:
        s_strike = float(s["strike"])
        s_mid = float(s["mid_calc"])

        for l in long_records:
            l_strike = float(l["strike"])
            l_mid = float(l["mid_calc"])

            if spread_type == "bull_put":
                if l_strike >= s_strike:
                    continue
                width = s_strike - l_strike
                credit = s_mid - l_mid
                if width <= 0 or credit <= 0:
                    continue
                max_loss = width - credit
                breakeven = s_strike - credit

            elif spread_type == "bear_call":
                if l_strike <= s_strike:
                    continue
                width = l_strike - s_strike
                credit = s_mid - l_mid
                if width <= 0 or credit <= 0:
                    continue
                max_loss = width - credit
                breakeven = s_strike + credit

            else:
                continue

            if max_loss <= 0:
                continue

            short_delta = float(s.get("delta", np.nan)) if pd.notna(s.get("delta", np.nan)) else np.nan
            roc = credit / max_loss

            rows.append(
                {
                    "short_strike": round(s_strike, 2),
                    "long_strike": round(l_strike, 2),
                    "width": round(width, 2),
                    "credit": round(credit, 2),
                    "max_loss": round(max_loss, 2),
                    "breakeven": round(breakeven, 2),
                    "roc": round(roc, 4),
                    "short_delta": round(short_delta, 3) if pd.notna(short_delta) else np.nan,
                    "short_bid": round(float(s.get("bid", np.nan)), 2) if pd.notna(s.get("bid", np.nan)) else np.nan,
                    "short_ask": round(float(s.get("ask", np.nan)), 2) if pd.notna(s.get("ask", np.nan)) else np.nan,
                    "long_bid": round(float(l.get("bid", np.nan)), 2) if pd.notna(l.get("bid", np.nan)) else np.nan,
                    "long_ask": round(float(l.get("ask", np.nan)), 2) if pd.notna(l.get("ask", np.nan)) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

# =========================================================
# SPREAD RECOMMENDATIONS
# =========================================================
def recommend_bull_put_spreads(
    chain: pd.DataFrame,
    spot: float,
    relevant_demand: Optional[Zone],
    max_short_delta: float = 0.30,
    min_credit: float = 0.20,
    max_width: float = 10.0,
) -> pd.DataFrame:
    puts = filter_chain_for_puts(chain)
    if puts.empty:
        return pd.DataFrame()

    shorts = puts[puts["strike"] < spot].copy()

    if "delta" in shorts.columns:
        shorts["abs_delta"] = shorts["delta"].abs()
        shorts = shorts[(pd.isna(shorts["abs_delta"])) | (shorts["abs_delta"] <= max_short_delta)]

    if relevant_demand is not None:
        shorts = shorts[shorts["strike"] <= relevant_demand.low]

    longs = puts.copy()

    spreads = build_vertical_spreads(shorts, longs, "bull_put")
    if spreads.empty:
        return spreads

    spreads = spreads[
        (spreads["credit"] >= min_credit) &
        (spreads["width"] <= max_width)
    ].copy()

    spreads["dist_from_spot_pct"] = ((spot - spreads["short_strike"]) / spot) * 100.0
    spreads["zone_buffer"] = (
        spreads["short_strike"] - relevant_demand.low if relevant_demand is not None else np.nan
    )

    spreads["score"] = (
        spreads["dist_from_spot_pct"] * 3
        + spreads["roc"] * 100
        + spreads["credit"] * 2
    )

    return spreads.sort_values(["score", "credit"], ascending=[False, False]).head(12)

def recommend_bear_call_spreads(
    chain: pd.DataFrame,
    spot: float,
    relevant_supply: Optional[Zone],
    max_short_delta: float = 0.30,
    min_credit: float = 0.20,
    max_width: float = 10.0,
) -> pd.DataFrame:
    calls = filter_chain_for_calls(chain)
    if calls.empty:
        return pd.DataFrame()

    shorts = calls[calls["strike"] > spot].copy()

    if "delta" in shorts.columns:
        shorts["abs_delta"] = shorts["delta"].abs()
        shorts = shorts[(pd.isna(shorts["abs_delta"])) | (shorts["abs_delta"] <= max_short_delta)]

    if relevant_supply is not None:
        shorts = shorts[shorts["strike"] >= relevant_supply.high]

    longs = calls.copy()

    spreads = build_vertical_spreads(shorts, longs, "bear_call")
    if spreads.empty:
        return spreads

    spreads = spreads[
        (spreads["credit"] >= min_credit) &
        (spreads["width"] <= max_width)
    ].copy()

    spreads["dist_from_spot_pct"] = ((spreads["short_strike"] - spot) / spot) * 100.0
    spreads["zone_buffer"] = (
        spreads["short_strike"] - relevant_supply.high if relevant_supply is not None else np.nan
    )

    spreads["score"] = (
        spreads["dist_from_spot_pct"] * 3
        + spreads["roc"] * 100
        + spreads["credit"] * 2
    )

    return spreads.sort_values(["score", "credit"], ascending=[False, False]).head(12)

def label_spreads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy().reset_index(drop=True)
    out["style"] = "Candidate"

    if len(out) >= 1:
        out.loc[0, "style"] = "Best Balanced"

    if len(out) >= 2:
        safest_idx = out["dist_from_spot_pct"].idxmax()
        out.loc[safest_idx, "style"] = "Best Conservative"

    if len(out) >= 3:
        aggressive_idx = out["credit"].idxmax()
        out.loc[aggressive_idx, "style"] = "Best Aggressive"

    return out

# =========================================================
# CHART / TABLE HELPERS
# =========================================================
def make_chart(df: pd.DataFrame, zones: List[Zone], bars: int = 160) -> go.Figure:
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

    for z in zones:
        fig.add_hrect(
            y0=z.low,
            y1=z.high,
            line_width=0,
            opacity=min(0.10 + z.touches * 0.01, 0.25),
            annotation_text=f"{z.kind.title()} ({z.touches})",
            annotation_position="top left",
        )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

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
col_a, col_b, col_c, col_d = st.columns([1.2, 1, 1, 1])

with col_a:
    symbol = st.text_input("Symbol", value="SPY").upper().strip()

with col_b:
    resolution = st.selectbox("Chart Resolution", ["D", "60", "30", "15", "5", "W"], index=0)

with col_c:
    min_dte = st.number_input("Min DTE", min_value=1, max_value=90, value=7, step=1)

with col_d:
    max_dte = st.number_input("Max DTE", min_value=2, max_value=120, value=35, step=1)

col_e, col_f, col_g, col_h = st.columns(4)
with col_e:
    fast_ema = st.number_input("Fast EMA", min_value=5, max_value=50, value=20, step=1)
with col_f:
    slow_ema = st.number_input("Slow EMA", min_value=10, max_value=100, value=50, step=1)
with col_g:
    pivot_left = st.number_input("Pivot Left", min_value=2, max_value=10, value=3, step=1)
with col_h:
    pivot_right = st.number_input("Pivot Right", min_value=2, max_value=10, value=3, step=1)

col_i, col_j, col_k = st.columns(3)
with col_i:
    atr_zone_mult = st.number_input("Zone Width ATR", min_value=0.2, max_value=2.0, value=0.7, step=0.1)
with col_j:
    max_short_delta = st.number_input("Max Short Delta", min_value=0.05, max_value=0.80, value=0.30, step=0.05)
with col_k:
    min_credit = st.number_input("Min Spread Credit", min_value=0.05, max_value=5.0, value=0.20, step=0.05)

if not API_KEY:
    st.error("Missing MARKETDATA_API_KEY in Streamlit secrets.")
    st.stop()

# =========================================================
# MAIN LOAD
# =========================================================
try:
    df = get_md_stock_candles(symbol, resolution=resolution, bars=260)
except Exception as e:
    st.error(f"Failed to load chart candles: {e}")
    st.stop()

if df.empty:
    st.error("No candle data returned. Double-check the symbol, plan, and Market Data account entitlements.")
    st.stop()

df = detect_trend(df, fast_len=int(fast_ema), slow_len=int(slow_ema))
df["ATR"] = atr(df, 14)

zones = build_zones(
    df=df,
    pivot_left=int(pivot_left),
    pivot_right=int(pivot_right),
    atr_width_mult=float(atr_zone_mult),
)

signal = pullback_signal(df, zones)
last = df.iloc[-1]
spot = float(last["Close"])
trend = str(last["Trend"])
atr_val = float(last["ATR"]) if pd.notna(last["ATR"]) else np.nan

nearest_demand = nearest_zones(spot, zones, "demand", n=1)
nearest_supply = nearest_zones(spot, zones, "supply", n=1)
demand_zone = nearest_demand[0] if nearest_demand else None
supply_zone = nearest_supply[0] if nearest_supply else None

quote = get_md_stock_quote(symbol)

# =========================================================
# SUMMARY
# =========================================================
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spot", f"{spot:.2f}")
m2.metric("Trend", trend)
m3.metric("ATR(14)", f"{atr_val:.2f}" if pd.notna(atr_val) else "n/a")
m4.metric("Setup", signal.get("state", "n/a"))

if quote and quote.get("updated") is not None:
    try:
        updated_ts = int(quote["updated"])
        updated_dt = datetime.utcfromtimestamp(updated_ts)
        st.caption(f"Quote updated (UTC): {updated_dt}")
    except Exception:
        pass

# =========================================================
# CHART
# =========================================================
st.plotly_chart(make_chart(df, zones), use_container_width=True)

# =========================================================
# CURRENT READ
# =========================================================
st.subheader("Current Read")

if trend == "Bullish" and demand_zone is not None:
    st.write(
        f"""
**State:** {signal.get('state', 'n/a')}  
**Trend:** {trend}  
**Nearest Demand Zone:** {demand_zone.low:.2f} to {demand_zone.high:.2f}  
**Touches:** {demand_zone.touches}
"""
    )
elif trend == "Bearish" and supply_zone is not None:
    st.write(
        f"""
**State:** {signal.get('state', 'n/a')}  
**Trend:** {trend}  
**Nearest Supply Zone:** {supply_zone.low:.2f} to {supply_zone.high:.2f}  
**Touches:** {supply_zone.touches}
"""
    )
else:
    st.write(f"**State:** {signal.get('state', 'n/a')}")

# =========================================================
# ZONES
# =========================================================
nz1, nz2 = st.columns(2)

with nz1:
    st.markdown("### Demand Zones")
    st.dataframe(zone_table(nearest_zones(spot, zones, "demand", n=5), spot), use_container_width=True)

with nz2:
    st.markdown("### Supply Zones")
    st.dataframe(zone_table(nearest_zones(spot, zones, "supply", n=5), spot), use_container_width=True)

# =========================================================
# SPREAD IDEAS
# =========================================================
st.subheader("Spread Ideas Based on Trend")

try:
    expirations = get_expirations(symbol)
except Exception as e:
    st.error(f"Failed to load expirations: {e}")
    expirations = []

chosen_exp = choose_expiration(expirations, min_dte=int(min_dte), max_dte=int(max_dte))

if not chosen_exp:
    st.warning("No expiration found in the selected DTE window.")
else:
    st.write(f"**Chosen expiration:** {chosen_exp}")

    try:
        chain = get_option_chain(symbol, chosen_exp)
    except Exception as e:
        st.error(f"Failed to load option chain: {e}")
        chain = pd.DataFrame()

    if chain.empty:
        st.info("No option chain returned.")
    else:
        bull_df = pd.DataFrame()
        bear_df = pd.DataFrame()

        if trend == "Bullish":
            bull_df = recommend_bull_put_spreads(
                chain=chain,
                spot=spot,
                relevant_demand=demand_zone,
                max_short_delta=float(max_short_delta),
                min_credit=float(min_credit),
                max_width=10.0,
            )
        elif trend == "Bearish":
            bear_df = recommend_bear_call_spreads(
                chain=chain,
                spot=spot,
                relevant_supply=supply_zone,
                max_short_delta=float(max_short_delta),
                min_credit=float(min_credit),
                max_width=10.0,
            )
        else:
            bull_df = recommend_bull_put_spreads(
                chain=chain,
                spot=spot,
                relevant_demand=demand_zone,
                max_short_delta=float(max_short_delta),
                min_credit=float(min_credit),
                max_width=10.0,
            )
            bear_df = recommend_bear_call_spreads(
                chain=chain,
                spot=spot,
                relevant_supply=supply_zone,
                max_short_delta=float(max_short_delta),
                min_credit=float(min_credit),
                max_width=10.0,
            )

        bull_df = label_spreads(bull_df)
        bear_df = label_spreads(bear_df)

        s1, s2 = st.columns(2)

        with s1:
            st.markdown("### Bull Put Spreads")
            if bull_df.empty:
                st.info("No bull put spreads passed filters.")
            else:
                st.dataframe(
                    bull_df[
                        [
                            "style", "short_strike", "long_strike", "width", "credit",
                            "max_loss", "breakeven", "roc", "short_delta",
                            "dist_from_spot_pct", "zone_buffer"
                        ]
                    ],
                    use_container_width=True
                )

        with s2:
            st.markdown("### Bear Call Spreads")
            if bear_df.empty:
                st.info("No bear call spreads passed filters.")
            else:
                st.dataframe(
                    bear_df[
                        [
                            "style", "short_strike", "long_strike", "width", "credit",
                            "max_loss", "breakeven", "roc", "short_delta",
                            "dist_from_spot_pct", "zone_buffer"
                        ]
                    ],
                    use_container_width=True
                )

# =========================================================
# WATCHLIST SCANNER
# =========================================================
st.subheader("Watchlist Scanner")

watchlist_text = st.text_area("Symbols", value=DEFAULT_WATCHLIST, height=100)

if st.button("Scan Watchlist"):
    symbols = [x.strip().upper() for x in watchlist_text.split(",") if x.strip()]
    rows = []

    for sym in symbols:
        try:
            h = get_md_stock_candles(sym, resolution="D", bars=220)
            if h.empty or len(h) < 80:
                continue

            h = detect_trend(h, fast_len=int(fast_ema), slow_len=int(slow_ema))
            h["ATR"] = atr(h, 14)
            z = build_zones(
                h,
                pivot_left=int(pivot_left),
                pivot_right=int(pivot_right),
                atr_width_mult=float(atr_zone_mult),
            )
            sig = pullback_signal(h, z)
            last_row = h.iloc[-1]

            rows.append(
                {
                    "Symbol": sym,
                    "Close": round(float(last_row["Close"]), 2),
                    "Trend": str(last_row["Trend"]),
                    "Setup": sig.get("state", "n/a"),
                }
            )
        except Exception:
            continue

    scan_df = pd.DataFrame(rows)

    if scan_df.empty:
        st.info("No symbols scanned successfully.")
    else:
        priority = {
            "Bullish Pullback In Zone": 1,
            "Bearish Pullback In Zone": 2,
            "Bullish Pullback Watch": 3,
            "Bearish Pullback Watch": 4,
            "Bullish Trend - No Pullback": 5,
            "Bearish Trend - No Pullback": 6,
            "Neutral / No Setup": 7,
        }
        scan_df["Sort"] = scan_df["Setup"].map(priority).fillna(99)
        scan_df = scan_df.sort_values(["Sort", "Symbol"]).drop(columns=["Sort"])
        st.dataframe(scan_df, use_container_width=True)