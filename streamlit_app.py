# streamlit_app.py

import math
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Trade Setup Scanner", layout="wide")

# =========================
# CONFIG
# =========================
DEFAULT_TICKERS = [
    "SPY","QQQ","IWM","DIA",
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA",
    "AMD","AVGO","NFLX","JPM","XOM","LLY","UNH","COST","CRM",
    "PLTR","MU","SMCI","BA","DIS","SHOP","COIN"
]

SECTOR_ETFS = [
    "XLK","XLF","XLE","XLV","XLY","XLI","XLP","XLB","XLU","XLC","VNQ","SMH"
]

# =========================
# HELPERS
# =========================
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def round_to_step(value: float, step: float, direction: str = "nearest") -> float:
    if step <= 0:
        return value
    if direction == "up":
        return math.ceil(value / step) * step
    elif direction == "down":
        return math.floor(value / step) * step
    return round(value / step) * step


def infer_strike_step(price: float) -> float:
    if price < 25:
        return 0.5
    elif price < 100:
        return 1.0
    elif price < 250:
        return 2.5
    elif price < 500:
        return 5.0
    return 10.0


def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1.0) * 100.0


# =========================
# DATA
# =========================
@st.cache_data(ttl=1800, show_spinner=False)
def load_price_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(columns=str.title)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan
    df = df[needed].dropna(subset=["Close"])
    return df.copy()


@st.cache_data(ttl=1800, show_spinner=False)
def load_option_expirations(ticker: str) -> List[str]:
    try:
        tk = yf.Ticker(ticker)
        return list(tk.options)
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def load_option_chain(ticker: str, expiration: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
        calls = chain.calls.copy() if chain.calls is not None else pd.DataFrame()
        puts = chain.puts.copy() if chain.puts is not None else pd.DataFrame()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


# =========================
# INDICATORS
# =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["EMA20"] = out["Close"].ewm(span=20, adjust=False).mean()
    out["EMA50"] = out["Close"].ewm(span=50, adjust=False).mean()
    out["EMA200"] = out["Close"].ewm(span=200, adjust=False).mean()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    prev_close = out["Close"].shift(1)
    tr1 = out["High"] - out["Low"]
    tr2 = (out["High"] - prev_close).abs()
    tr3 = (out["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()

    out["VOL20"] = out["Volume"].rolling(20).mean()
    out["RET20"] = out["Close"].pct_change(20)
    out["RET60"] = out["Close"].pct_change(60)

    return out


# =========================
# ZONES
# =========================
@dataclass
class Zone:
    kind: str          # "supply" or "demand"
    low: float
    high: float
    pivot_idx: int
    touches: int
    freshness: int     # bars since pivot


def detect_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
    highs = df["High"].values
    lows = df["Low"].values

    pivot_highs = []
    pivot_lows = []

    for i in range(left, len(df) - right):
        window_high = highs[i-left:i+right+1]
        window_low = lows[i-left:i+right+1]

        if highs[i] == np.max(window_high):
            if list(window_high).count(highs[i]) == 1:
                pivot_highs.append(i)

        if lows[i] == np.min(window_low):
            if list(window_low).count(lows[i]) == 1:
                pivot_lows.append(i)

    return pivot_highs, pivot_lows


def build_zones(df: pd.DataFrame, max_zones: int = 6) -> List[Zone]:
    if len(df) < 60:
        return []

    ph, pl = detect_pivots(df, left=3, right=3)
    atr = safe_float(df["ATR14"].iloc[-1], 0)
    if pd.isna(atr) or atr <= 0:
        atr = safe_float((df["High"] - df["Low"]).rolling(14).mean().iloc[-1], 1.0)
    zone_half = max(atr * 0.5, df["Close"].iloc[-1] * 0.005)

    zones: List[Zone] = []

    for idx in ph:
        p = safe_float(df["High"].iloc[idx])
        z = Zone(
            kind="supply",
            low=p - zone_half,
            high=p + zone_half,
            pivot_idx=idx,
            touches=0,
            freshness=len(df) - idx
        )
        zones.append(z)

    for idx in pl:
        p = safe_float(df["Low"].iloc[idx])
        z = Zone(
            kind="demand",
            low=p - zone_half,
            high=p + zone_half,
            pivot_idx=idx,
            touches=0,
            freshness=len(df) - idx
        )
        zones.append(z)

    # Count touches after pivot
    for z in zones:
        future = df.iloc[z.pivot_idx + 1:]
        if len(future) == 0:
            continue
        hits = ((future["High"] >= z.low) & (future["Low"] <= z.high)).sum()
        z.touches = int(hits)

    # Score zones: recent + moderate touches
    def zone_score(z: Zone):
        freshness_score = max(0, 120 - z.freshness)
        touch_score = 15 if 1 <= z.touches <= 4 else 5 if z.touches == 0 else 0
        return freshness_score + touch_score

    zones = sorted(zones, key=zone_score, reverse=True)

    # Merge overlapping similar zones
    merged = []
    for z in zones:
        overlap = False
        for m in merged:
            if z.kind == m.kind and not (z.high < m.low or z.low > m.high):
                m.low = min(m.low, z.low)
                m.high = max(m.high, z.high)
                m.touches = max(m.touches, z.touches)
                m.freshness = min(m.freshness, z.freshness)
                overlap = True
                break
        if not overlap:
            merged.append(z)

    return merged[:max_zones]


def nearest_zones(price: float, zones: List[Zone]) -> Dict[str, Optional[Zone]]:
    supply_above = [z for z in zones if z.kind == "supply" and z.low > price]
    demand_below = [z for z in zones if z.kind == "demand" and z.high < price]

    nearest_supply = min(supply_above, key=lambda z: z.low - price) if supply_above else None
    nearest_demand = min(demand_below, key=lambda z: price - z.high) if demand_below else None

    return {
        "supply": nearest_supply,
        "demand": nearest_demand
    }


# =========================
# SCORING
# =========================
def score_setup(df: pd.DataFrame, zones: List[Zone]) -> Dict[str, float]:
    row = df.iloc[-1]
    close = safe_float(row["Close"])
    ema20 = safe_float(row["EMA20"])
    ema50 = safe_float(row["EMA50"])
    ema200 = safe_float(row["EMA200"])
    rsi = safe_float(row["RSI14"])
    ret20 = safe_float(row["RET20"])
    ret60 = safe_float(row["RET60"])
    atr = safe_float(row["ATR14"])

    trend_score = 0
    if close > ema20:
        trend_score += 10
    if close > ema50:
        trend_score += 12
    if close > ema200:
        trend_score += 15
    if ema20 > ema50 > ema200:
        trend_score += 18
    elif ema20 < ema50 < ema200:
        trend_score -= 18

    momentum_score = 0
    if pd.notna(rsi):
        if 52 <= rsi <= 68:
            momentum_score += 12
        elif 45 <= rsi < 52:
            momentum_score += 6
        elif rsi > 75:
            momentum_score -= 5
        elif rsi < 35:
            momentum_score -= 5

    if pd.notna(ret20):
        momentum_score += max(-8, min(8, ret20 * 100))
    if pd.notna(ret60):
        momentum_score += max(-10, min(10, ret60 * 60))

    structure_score = 0
    nz = nearest_zones(close, zones)
    supply = nz["supply"]
    demand = nz["demand"]

    if demand is not None:
        dist_to_demand = close - demand.high
        if atr > 0:
            atrs = dist_to_demand / atr
            if 0.2 <= atrs <= 1.5:
                structure_score += 10
            elif atrs > 4:
                structure_score -= 4

    if supply is not None:
        dist_to_supply = supply.low - close
        if atr > 0:
            atrs = dist_to_supply / atr
            if 0.5 <= atrs <= 3.0:
                structure_score += 8
            elif atrs < 0.25:
                structure_score -= 10

    total = trend_score + momentum_score + structure_score

    direction = "Neutral"
    if close > ema50 and ema20 > ema50 and rsi >= 50:
        direction = "Bullish"
    elif close < ema50 and ema20 < ema50 and rsi <= 50:
        direction = "Bearish"

    return {
        "trend_score": round(trend_score, 2),
        "momentum_score": round(momentum_score, 2),
        "structure_score": round(structure_score, 2),
        "setup_score": round(total, 2),
        "direction_num": 1 if direction == "Bullish" else -1 if direction == "Bearish" else 0,
        "direction": direction
    }


# =========================
# OPTIONS / STRATEGY LOGIC
# =========================
def get_iv_proxy(ticker: str) -> float:
    expirations = load_option_expirations(ticker)
    if not expirations:
        return np.nan

    for exp in expirations[:2]:
        calls, puts = load_option_chain(ticker, exp)
        ivs = []
        if not calls.empty and "impliedVolatility" in calls.columns:
            ivs.extend(calls["impliedVolatility"].dropna().tolist())
        if not puts.empty and "impliedVolatility" in puts.columns:
            ivs.extend(puts["impliedVolatility"].dropna().tolist())
        if ivs:
            iv = float(np.nanmedian(ivs))
            if iv > 0:
                return iv
    return np.nan


def suggest_strategy(
    ticker: str,
    df: pd.DataFrame,
    zones: List[Zone],
    score: Dict[str, float],
    use_options_data: bool = True
) -> Dict[str, str]:
    row = df.iloc[-1]
    price = safe_float(row["Close"])
    atr = safe_float(row["ATR14"])
    rsi = safe_float(row["RSI14"])
    direction = score["direction"]
    score_val = score["setup_score"]
    iv_proxy = get_iv_proxy(ticker) if use_options_data else np.nan

    nz = nearest_zones(price, zones)
    nearest_supply = nz["supply"]
    nearest_demand = nz["demand"]

    dist_supply = (nearest_supply.low - price) if nearest_supply else np.nan
    dist_demand = (price - nearest_demand.high) if nearest_demand else np.nan

    strategy = "Watchlist"
    reason = "No clean edge."
    setup = ""

    # Bullish
    if direction == "Bullish":
        if pd.notna(iv_proxy) and iv_proxy >= 0.40:
            strategy = "Bull Put Spread"
            reason = "Bullish trend with elevated IV favors selling premium."
        elif score_val >= 40 and rsi < 70:
            strategy = "Buy Stock / Call"
            reason = "Strong directional trend and momentum."
        else:
            strategy = "Bull Call LEAPS"
            reason = "Trend is constructive but a longer time horizon may fit better."

        short_put, long_put = bullish_spread_strikes_outside_demand(price, atr, nearest_demand)
        if short_put is not None:
            setup = f"Sell {short_put:.2f}P / Buy {long_put:.2f}P (below demand zone)"
        else:
            setup = "No clean put spread strike outside demand zone"

    # Bearish
    elif direction == "Bearish":
        if pd.notna(iv_proxy) and iv_proxy >= 0.40:
            strategy = "Bear Call Spread"
            reason = "Bearish trend with elevated IV favors selling premium."
        elif score_val <= -20:
            strategy = "Buy Put"
            reason = "Weak trend and momentum support outright bearish exposure."
        else:
            strategy = "Bear Put LEAPS"
            reason = "Bearish structure may work better with more time."

        short_call, long_call = bearish_spread_strikes_outside_supply(price, atr, nearest_supply)
        if short_call is not None:
            setup = f"Sell {short_call:.2f}C / Buy {long_call:.2f}C (above supply zone)"
        else:
            setup = "No clean call spread strike outside supply zone"

    else:
        strategy = "Watchlist / Defined Risk Spread"
        reason = "Mixed trend. Wait for cleaner alignment."
        setup = "Wait for break above supply or reclaim from demand"

    if strategy == "Bull Call LEAPS":
        leap_strike = round_to_step(price * 0.95, infer_strike_step(price), "nearest")
        setup = f"Buy 6-12 month {leap_strike:.2f}C"
    elif strategy == "Bear Put LEAPS":
        leap_strike = round_to_step(price * 1.05, infer_strike_step(price), "nearest")
        setup = f"Buy 6-12 month {leap_strike:.2f}P"
    elif strategy == "Buy Stock / Call":
        call_strike = round_to_step(price, infer_strike_step(price), "nearest")
        setup = f"Buy shares or near-ATM {call_strike:.2f}C"
    elif strategy == "Buy Put":
        put_strike = round_to_step(price, infer_strike_step(price), "nearest")
        setup = f"Buy near-ATM {put_strike:.2f}P"

    return {
        "strategy": strategy,
        "reason": reason,
        "trade_setup": setup,
        "iv_proxy": round(iv_proxy, 3) if pd.notna(iv_proxy) else np.nan,
        "nearest_supply": f"{nearest_supply.low:.2f}-{nearest_supply.high:.2f}" if nearest_supply else "",
        "nearest_demand": f"{nearest_demand.low:.2f}-{nearest_demand.high:.2f}" if nearest_demand else ""
    }


def bullish_spread_strikes_outside_demand(price: float, atr: float, demand_zone: Optional[Zone]) -> Tuple[Optional[float], Optional[float]]:
    step = infer_strike_step(price)
    width = step * 2 if price < 150 else step * 4

    if demand_zone is not None:
        # short put below demand low with small buffer
        short_put = round_to_step(demand_zone.low - max(0.25 * atr, step), step, "down")
    else:
        short_put = round_to_step(price - max(1.5 * atr, 3 * step), step, "down")

    long_put = short_put - width
    if long_put <= 0:
        return None, None
    return short_put, long_put


def bearish_spread_strikes_outside_supply(price: float, atr: float, supply_zone: Optional[Zone]) -> Tuple[Optional[float], Optional[float]]:
    step = infer_strike_step(price)
    width = step * 2 if price < 150 else step * 4

    if supply_zone is not None:
        # short call above supply high with small buffer
        short_call = round_to_step(supply_zone.high + max(0.25 * atr, step), step, "up")
    else:
        short_call = round_to_step(price + max(1.5 * atr, 3 * step), step, "up")

    long_call = short_call + width
    return short_call, long_call


# =========================
# SCAN
# =========================
def analyze_ticker(ticker: str, use_options_data: bool = True) -> Optional[Dict]:
    try:
        df = load_price_history(ticker)
        if df.empty or len(df) < 220:
            return None

        df = add_indicators(df)
        zones = build_zones(df)
        score = score_setup(df, zones)
        strat = suggest_strategy(ticker, df, zones, score, use_options_data=use_options_data)

        last = df.iloc[-1]
        close = safe_float(last["Close"])
        atr = safe_float(last["ATR14"])
        rsi = safe_float(last["RSI14"])
        ema20 = safe_float(last["EMA20"])
        ema50 = safe_float(last["EMA50"])
        ema200 = safe_float(last["EMA200"])

        return {
            "Ticker": ticker,
            "Price": round(close, 2),
            "ATR14": round(atr, 2) if pd.notna(atr) else np.nan,
            "RSI14": round(rsi, 1) if pd.notna(rsi) else np.nan,
            "EMA20": round(ema20, 2),
            "EMA50": round(ema50, 2),
            "EMA200": round(ema200, 2),
            "Direction": score["direction"],
            "TrendScore": score["trend_score"],
            "MomentumScore": score["momentum_score"],
            "StructureScore": score["structure_score"],
            "SetupScore": score["setup_score"],
            "Strategy": strat["strategy"],
            "TradeSetup": strat["trade_setup"],
            "Reason": strat["reason"],
            "IVProxy": strat["iv_proxy"],
            "NearestSupply": strat["nearest_supply"],
            "NearestDemand": strat["nearest_demand"],
            "Zones": zones,
            "Data": df
        }
    except Exception:
        return None


# =========================
# PLOTTING
# =========================
def plot_chart(df: pd.DataFrame, zones: List[Zone], ticker: str):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], mode="lines", name="EMA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], mode="lines", name="EMA200"))

    x0 = df.index[0]
    x1 = df.index[-1]

    for z in zones:
        fill = "rgba(255,0,0,0.12)" if z.kind == "supply" else "rgba(0,180,0,0.12)"
        line = "rgba(255,0,0,0.35)" if z.kind == "supply" else "rgba(0,180,0,0.35)"
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1,
            y0=z.low, y1=z.high,
            fillcolor=fill,
            line=dict(color=line, width=1),
            layer="below"
        )

    fig.update_layout(
        title=f"{ticker} Price + Supply/Demand Zones",
        xaxis_title="Date",
        yaxis_title="Price",
        height=650,
        xaxis_rangeslider_visible=False
    )
    return fig


# =========================
# UI
# =========================
st.title("Stock Setup Scanner")
st.caption("Scans stocks, scores setup quality, maps supply/demand zones, and suggests outright, LEAPS, or spreads outside zones.")

with st.sidebar:
    st.header("Scan Settings")

    preset = st.selectbox(
        "Universe",
        ["Default List", "Sector ETFs", "Custom"]
    )

    if preset == "Default List":
        tickers = DEFAULT_TICKERS
    elif preset == "Sector ETFs":
        tickers = SECTOR_ETFS
    else:
        custom = st.text_area("Custom tickers (comma separated)", value="SPY,QQQ,AAPL,MSFT,NVDA,AMD,AMZN,META")
        tickers = [x.strip().upper() for x in custom.split(",") if x.strip()]

    use_options_data = st.checkbox("Use options data for IV-based strategy tilt", value=True)
    min_score = st.slider("Minimum setup score", -20, 80, 15)
    top_n = st.slider("Top results", 5, 50, 15)

    scan = st.button("Run Scan", type="primary")

if "scan_results" not in st.session_state:
    st.session_state.scan_results = pd.DataFrame()
if "scan_raw" not in st.session_state:
    st.session_state.scan_raw = []

if scan:
    results = []
    raw = []

    progress = st.progress(0)
    status = st.empty()

    total = max(1, len(tickers))
    for i, t in enumerate(tickers):
        status.write(f"Scanning {t}...")
        res = analyze_ticker(t, use_options_data=use_options_data)
        if res is not None:
            raw.append(res)
            results.append({
                k: v for k, v in res.items()
                if k not in ["Zones", "Data"]
            })
        progress.progress((i + 1) / total)

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values(["SetupScore", "TrendScore"], ascending=False)
        df_res = df_res[df_res["SetupScore"] >= min_score].head(top_n)

    st.session_state.scan_results = df_res
    st.session_state.scan_raw = raw
    status.write("Done.")

results_df = st.session_state.scan_results
raw_results = st.session_state.scan_raw

tab1, tab2, tab3 = st.tabs(["Scanner", "Chart Detail", "How It Thinks"])

with tab1:
    st.subheader("Scanner Results")

    if results_df.empty:
        st.info("Run the scan to see setups.")
    else:
        st.dataframe(results_df, use_container_width=True)

        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results CSV",
            data=csv,
            file_name="scanner_results.csv",
            mime="text/csv"
        )

        st.markdown("### Best-looking names")
        for _, row in results_df.head(5).iterrows():
            st.markdown(
                f"""
**{row['Ticker']}** — {row['Direction']}  
Score: **{row['SetupScore']}**  
Strategy: **{row['Strategy']}**  
Setup: **{row['TradeSetup']}**  
Reason: {row['Reason']}  
Supply: {row['NearestSupply'] or 'N/A'} | Demand: {row['NearestDemand'] or 'N/A'}
"""
            )

with tab2:
    st.subheader("Chart Detail")

    if not raw_results:
        st.info("Run a scan first.")
    else:
        ticker_choices = [r["Ticker"] for r in raw_results if r["Ticker"] in set(results_df["Ticker"]) or results_df.empty]
        if not ticker_choices:
            ticker_choices = [r["Ticker"] for r in raw_results]

        selected = st.selectbox("Select ticker", ticker_choices)
        selected_result = next((r for r in raw_results if r["Ticker"] == selected), None)

        if selected_result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price", selected_result["Price"])
            c2.metric("Direction", selected_result["Direction"])
            c3.metric("Setup Score", selected_result["SetupScore"])
            c4.metric("Strategy", selected_result["Strategy"])

            st.plotly_chart(
                plot_chart(selected_result["Data"], selected_result["Zones"], selected),
                use_container_width=True
            )

            zrows = []
            for z in selected_result["Zones"]:
                zrows.append({
                    "Type": z.kind.title(),
                    "Low": round(z.low, 2),
                    "High": round(z.high, 2),
                    "Touches": z.touches,
                    "FreshnessBars": z.freshness
                })
            if zrows:
                st.markdown("### Detected Zones")
                st.dataframe(pd.DataFrame(zrows), use_container_width=True)

            st.markdown("### Suggested Trade")
            st.write(f"**Strategy:** {selected_result['Strategy']}")
            st.write(f"**Setup:** {selected_result['TradeSetup']}")
            st.write(f"**Reason:** {selected_result['Reason']}")
            st.write(f"**Nearest Supply:** {selected_result['NearestSupply'] or 'N/A'}")
            st.write(f"**Nearest Demand:** {selected_result['NearestDemand'] or 'N/A'}")
            st.write(f"**IV Proxy:** {selected_result['IVProxy']}")

with tab3:
    st.subheader("How the scanner decides")
    st.markdown(
        """
- **Bullish trend** generally means price is above EMA50, EMA20 > EMA50, and RSI is not weak.
- **Bearish trend** is the opposite.
- **Supply zones** come from pivot highs.
- **Demand zones** come from pivot lows.
- **Bull put spreads** are suggested with the short put **below the nearest demand zone**.
- **Bear call spreads** are suggested with the short call **above the nearest supply zone**.
- **LEAPS** are favored when the chart has a directional bias but not enough immediate edge for shorter-term entries.
- **Buy outright** is favored when trend and momentum are both strong and clean.
"""
    )

st.markdown("---")
st.caption("For education only. Not financial advice.")