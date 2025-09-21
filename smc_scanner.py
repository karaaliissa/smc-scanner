# --- Config you can tweak ---
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"]
MIN_24H_QUOTE_VOL = 50_000_000   # 24h quote volume filter
BOS_LEN = 14                     # structure break lookback (try 10–20)
IDM_LOOKBACK = 8                 # search window for last opposite candle (5–8 typical)
# ----------------------------

import os, json, math
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import ccxt
import pandas as pd
from dotenv import load_dotenv

# Telegram (v13.x API)
from telegram import Bot

# ------------- GLOBAL CONFIG -------------
EXCHANGE_ID = "binance"
TIMEFRAME = "1d"
CANDLES_LIMIT = 400
ALERT_STATE_FILE = "smc_state.json"  # prevents duplicate alerts
ALERT_PREFIX = "📣 SMC"
# -----------------------------------------

load_dotenv()
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
bot: Optional[Bot] = Bot(TG_TOKEN) if TG_TOKEN and TG_CHAT else None

def send(msg: str):
    print(msg)
    if bot:
        try:
            bot.send_message(chat_id=TG_CHAT, text=msg[:4000])
        except Exception as e:
            print("Telegram error:", e)

def now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def load_state() -> Dict:
    if os.path.exists(ALERT_STATE_FILE):
        try:
            with open(ALERT_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(st: Dict):
    tmp = ALERT_STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    os.replace(tmp, ALERT_STATE_FILE)

def init_exchange():
    # try preferred exchange first (env or global), then fall back to binanceus, then okx
    preferred = os.getenv("EXCHANGE_ID") or EXCHANGE_ID
    for exch_id in [preferred, "binanceus", "okx"]:
        try:
            ex = getattr(ccxt, exch_id)({
                "enableRateLimit": True,
                "options": {"defaultType": "spot", "fetchCurrencies": False},
                # IMPORTANT: do NOT pass apiKey/secret – we only use public endpoints
            })
            params = {"fetchCurrencies": False} if exch_id.startswith("binance") else {}
            ex.load_markets(params=params)
            print(f"Using exchange: {exch_id}")
            return ex
        except Exception as e:
            print(f"Init failed for {exch_id}: {e}")
            continue
    raise RuntimeError("All exchanges failed to initialize")



def filter_by_volume(ex, symbols: List[str]) -> List[str]:
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        return symbols
    res = []
    for s in symbols:
        t = tickers.get(s)
        qv = (t or {}).get("quoteVolume") or 0
        if qv >= MIN_24H_QUOTE_VOL:
            res.append(s)
    return res or symbols

def fetch_df(ex, symbol: str, limit: int) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

# ---------- SMC helpers ----------
def is_inside(df: pd.DataFrame, i: int) -> bool:
    # candle i is inside candle i+1 (older)
    return df.loc[i, "high"] <= df.loc[i+1, "high"] and df.loc[i, "low"] >= df.loc[i+1, "low"]

def bos_up(df: pd.DataFrame, idx_close: int, lookback: int) -> bool:
    # current close > highest high of prior N bars (exclude current)
    start = max(0, idx_close - lookback)
    hh = df.loc[start:idx_close-1, "high"].max()
    return df.loc[idx_close, "close"] > hh

def bos_down(df: pd.DataFrame, idx_close: int, lookback: int) -> bool:
    start = max(0, idx_close - lookback)
    ll = df.loc[start:idx_close-1, "low"].min()
    return df.loc[idx_close, "close"] < ll

def find_idm_bull(df: pd.DataFrame, idx_close: int, lookback: int) -> Optional[int]:
    # last bearish candle before BOS up that is NOT inside previous candle
    start = max(0, idx_close - lookback)
    for i in range(idx_close-1, start-1, -1):
        if df.loc[i, "close"] < df.loc[i, "open"] and not is_inside(df, i):
            return i
    return None

def find_idm_bear(df: pd.DataFrame, idx_close: int, lookback: int) -> Optional[int]:
    # last bullish candle before BOS down that is NOT inside previous candle
    start = max(0, idx_close - lookback)
    for i in range(idx_close-1, start-1, -1):
        if df.loc[i, "close"] > df.loc[i, "open"] and not is_inside(df, i):
            return i
    return None

def latest_liquidity_sweeps(df: pd.DataFrame) -> Tuple[Optional[int], Optional[int]]:
    """
    Returns indices of the latest (most extreme) HIGH sweep and LOW sweep.
    Sweep = wick pierces previous bar's high/low but body closes back inside.
    """
    hi_idx = None
    hi_level = -float("inf")
    lo_idx = None
    lo_level = float("inf")

    for i in range(1, len(df)):
        prev_h = df.loc[i-1, "high"]
        prev_l = df.loc[i-1, "low"]
        # high sweep
        if df.loc[i, "high"] > prev_h and df.loc[i, "close"] < prev_h:
            if df.loc[i, "high"] > hi_level:
                hi_level = df.loc[i, "high"]; hi_idx = i
        # low sweep
        if df.loc[i, "low"] < prev_l and df.loc[i, "close"] > prev_l:
            if df.loc[i, "low"] < lo_level:
                lo_level = df.loc[i, "low"]; lo_idx = i
    return hi_idx, lo_idx

def first_retest(df: pd.DataFrame, zone_high: float, zone_low: float, from_idx: int) -> Optional[int]:
    """Find first future bar index that touches the zone after from_idx"""
    for i in range(from_idx+1, len(df)):
        lo = df.loc[i, "low"]; hi = df.loc[i, "high"]
        if (lo <= zone_high) and (hi >= zone_low):
            return i
    return None

def fmt_price(p: float) -> str:
    if p >= 100:   return f"{p:.2f}"
    if p >= 1:     return f"{p:.3f}"
    return f"{p:.6f}"

# ---------- Scanner ----------
def scan_symbol(ex, symbol: str, state: Dict):
    df = fetch_df(ex, symbol, CANDLES_LIMIT)

    # only evaluate on the **last closed** candle
    if len(df) < 50:
        return
    idx_close = len(df) - 2  # second to last = closed
    row = df.loc[idx_close]

    # Prepare per-symbol state
    st = state.setdefault(symbol, {
        "last_bull_idm_idx": None,
        "last_bear_idm_idx": None,
        "last_flip_alerts": [],   # track flip messages we already sent
        "last_hi_sweep_idx": None,
        "last_lo_sweep_idx": None
    })

    # ------ IDM + BOS ------
    messages: List[str] = []

    # BOS up
    if bos_up(df, idx_close, BOS_LEN):
        idm_idx = find_idm_bull(df, idx_close, IDM_LOOKBACK)
        if idm_idx is not None and st.get("last_bull_idm_idx") != idm_idx:
            zone_h = df.loc[idm_idx, "high"]; zone_l = df.loc[idm_idx, "low"]
            messages.append(
                f"{ALERT_PREFIX} • {symbol}\n🟢 IDM + BOS (bullish)\nIDM bar: {df.loc[idm_idx,'dt'].date()}  "
                f"Zone {fmt_price(zone_l)}–{fmt_price(zone_h)}\nPlan: watch D1/LTF retrace into zone."
            )
            st["last_bull_idm_idx"] = idm_idx

            # flip-through (close beyond zone)
            if row["close"] > zone_h:
                flip_msg = f"{symbol} • Flip confirmed: supply→demand (close above {fmt_price(zone_h)})."
                if flip_msg not in st["last_flip_alerts"]:
                    messages.append(f"{ALERT_PREFIX} • {flip_msg}")
                    st["last_flip_alerts"].append(flip_msg)

            # first retest after IDM
            touch_idx = first_retest(df, zone_h, zone_l, idx_close)
            if touch_idx is not None:
                messages.append(
                    f"{ALERT_PREFIX} • {symbol}\n🟢 First retest of Bullish IDM zone at "
                    f"{fmt_price(zone_l)}–{fmt_price(zone_h)} on {df.loc[touch_idx,'dt'].date()}."
                )

    # BOS down
    if bos_down(df, idx_close, BOS_LEN):
        idm_idx = find_idm_bear(df, idx_close, IDM_LOOKBACK)
        if idm_idx is not None and st.get("last_bear_idm_idx") != idm_idx:
            zone_h = df.loc[idm_idx, "high"]; zone_l = df.loc[idm_idx, "low"]
            messages.append(
                f"{ALERT_PREFIX} • {symbol}\n🔴 IDM + BOS (bearish)\nIDM bar: {df.loc[idm_idx,'dt'].date()}  "
                f"Zone {fmt_price(zone_l)}–{fmt_price(zone_h)}\nPlan: watch retrace into zone."
            )
            st["last_bear_idm_idx"] = idm_idx

            if row["close"] < zone_l:
                flip_msg = f"{symbol} • Flip confirmed: demand→supply (close below {fmt_price(zone_l)})."
                if flip_msg not in st["last_flip_alerts"]:
                    messages.append(f"{ALERT_PREFIX} • {flip_msg}")
                    st["last_flip_alerts"].append(flip_msg)

            touch_idx = first_retest(df, zone_h, zone_l, idx_close)
            if touch_idx is not None:
                messages.append(
                    f"{ALERT_PREFIX} • {symbol}\n🔴 First retest of Bearish IDM zone at "
                    f"{fmt_price(zone_l)}–{fmt_price(zone_h)} on {df.loc[touch_idx,'dt'].date()}."
                )

    # ------ Liquidity: latest extreme high/low sweeps ------
    hi_idx, lo_idx = latest_liquidity_sweeps(df)
    if hi_idx is not None and st.get("last_hi_sweep_idx") != hi_idx:
        st["last_hi_sweep_idx"] = hi_idx
        messages.append(
            f"{ALERT_PREFIX} • {symbol}\n🔻 Latest HIGH liquidity sweep on {df.loc[hi_idx,'dt'].date()} "
            f"(wick {fmt_price(df.loc[hi_idx,'high'])})."
        )
    if lo_idx is not None and st.get("last_lo_sweep_idx") != lo_idx:
        st["last_lo_sweep_idx"] = lo_idx
        messages.append(
            f"{ALERT_PREFIX} • {symbol}\n🔺 Latest LOW liquidity sweep on {df.loc[lo_idx,'dt'].date()} "
            f"(wick {fmt_price(df.loc[lo_idx,'low'])})."
        )

    # Send
    for m in messages:
        send(m)

def main():
    print(f"[{now_utc_str()}] Starting SMC daily scanner…")
    ex = init_exchange()

    # Volume filter
    symbols = [s for s in WATCHLIST if s in ex.symbols]
    symbols = filter_by_volume(ex, symbols)

    state = load_state()
    for s in symbols:
        try:
            scan_symbol(ex, s, state)
        except Exception as e:
            print(f"Error scanning {s}:", e)
            continue
    save_state(state)
    print(f"[{now_utc_str()}] Done.")

if __name__ == "__main__":
    main()
