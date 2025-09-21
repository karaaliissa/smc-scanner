# ===================== SMC 4H SCANNER =====================
# Timeframe: 4h, evaluates only the last CLOSED bar
# Signals:
# - IDM + BOS (bull/bear) with "not-inside" filter
# - OB levels in alerts (wick & body)
# - Flip-through of OB zone (close beyond)
# - First retest of OB/IDM zone
# - Latest extreme liquidity sweeps (high & low)
# - Optional: volume expansion filter on BOS
# - Range patterns: Type-1 (sell→range→buy), Type-2 (buy→range→buy)

import os, json
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import ccxt
import pandas as pd
from dotenv import load_dotenv
from telegram import Bot

# ---------------- CONFIG YOU MAY TWEAK ----------------
WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "SUI/USDT"]

EXCHANGE_ID = "okx"           # local default; workflow may override via EXCHANGE_ID
TIMEFRAME = "4h"
CANDLES_LIMIT = 600           # more history for ranges on 4h

# Filters / heuristics
MIN_24H_QUOTE_VOL = 50_000_000    # ignore pairs with low 24h quote volume
BOS_LEN = 18                      # lookback bars for structure break on 4h (try 16–24)
IDM_LOOKBACK = 8                  # search window for last opposite candle (5–10 typical)

# Volume expansion filter (on the BOS bar)
REQUIRE_VOL_EXPANSION = True      # if True, only alert BOS when vol expands vs MA
VOL_MA = 20                       # bars for average volume
VOL_FACTOR = 1.5                  # current vol must exceed VOL_FACTOR * MA

# Range pattern heuristics (very lightweight)
RANGE_BARS = 6                    # how many bars define the recent range
BEFORE_RANGE_BARS = 6             # bars before the range to detect "sell" or "buy" leg
RANGE_TIGHTNESS = 0.75            # range width < (RANGE_TIGHTNESS * avg(hl-range, 20))
# ------------------------------------------------------

ALERT_STATE_FILE = "smc_state.json"   # prevents duplicate alerts
ALERT_PREFIX = "📣 SMC"

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

# ---------- Exchange init (no API keys; use public OHLCV) ----------
def init_exchange():
    # prefer env override first, then a robust list
    preferred = os.getenv("EXCHANGE_ID") or EXCHANGE_ID
    candidates = [preferred, "okx", "bybit", "binanceus", "kraken"]
    tried = []
    for exch_id in candidates:
        if exch_id in tried: 
            continue
        tried.append(exch_id)
        try:
            ex = getattr(ccxt, exch_id)({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            })
            ex.load_markets()
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
        t = tickers.get(s) or {}
        qv = t.get("quoteVolume")
        if qv is None:
            # try to approximate if baseVolume and last are present
            try:
                qv = (t.get("baseVolume") or 0) * (t.get("last") or 0)
            except Exception:
                qv = 0
        if (qv or 0) >= MIN_24H_QUOTE_VOL:
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
    Returns indices of the latest extreme HIGH sweep and LOW sweep.
    Sweep = wick pierces previous bar's high/low but body closes back inside.
    """
    hi_idx = None; hi_level = -float("inf")
    lo_idx = None; lo_level = float("inf")
    for i in range(1, len(df)):
        prev_h = df.loc[i-1, "high"]; prev_l = df.loc[i-1, "low"]
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
    for i in range(from_idx+1, len(df)):
        lo = df.loc[i, "low"]; hi = df.loc[i, "high"]
        if (lo <= zone_high) and (hi >= zone_low):
            return i
    return None

def fmt_price(p: float) -> str:
    if p >= 100:   return f"{p:.2f}"
    if p >= 1:     return f"{p:.3f}"
    return f"{p:.6f}"

# ---- Range patterns (very compact heuristics) ----
def _avg_hl(df, window=20, end=None):
    end = len(df) if end is None else end
    start = max(0, end - window)
    return (df.loc[start:end-1, "high"] - df.loc[start:end-1, "low"]).mean()

def detect_range_type(df: pd.DataFrame, idx_close: int) -> Optional[str]:
    """
    Return:
      "Type-1 (sell→range→buy)"  or
      "Type-2 (buy→range→buy)"   or None
    Very lightweight:
      - Look at the last RANGE_BARS before idx_close as a 'range'
      - Range must be tighter than recent avg range
      - Before the range, see a net down (type-1) or net up (type-2) over BEFORE_RANGE_BARS
      - Only evaluated when BOS up is true (continuation to buy)
    """
    r_start = max(0, idx_close - RANGE_BARS)
    r_h = df.loc[r_start:idx_close-1, "high"].max()
    r_l = df.loc[r_start:idx_close-1, "low"].min()
    range_width = r_h - r_l
    avg_recent = _avg_hl(df, window=20, end=idx_close)
    if range_width >= RANGE_TIGHTNESS * (avg_recent or 1e-9):
        return None  # not tight enough

    b_end = r_start
    b_start = max(0, b_end - BEFORE_RANGE_BARS)
    if b_end - b_start < 2:
        return None
    net = df.loc[b_end-1, "close"] - df.loc[b_start, "close"]
    # if the pre-range was net down -> Type-1, if up -> Type-2
    if net < 0:
        return "Type-1 (sell→range→buy)"
    if net > 0:
        return "Type-2 (buy→range→buy)"
    return None

def vol_expanded(df: pd.DataFrame, idx_close: int, ma_len: int, factor: float) -> bool:
    start = max(0, idx_close - ma_len)
    base = df.loc[start:idx_close-1, "vol"].mean()
    return df.loc[idx_close, "vol"] > (factor * (base or 1e-9))

# ---------- Scanner ----------
def scan_symbol(ex, symbol: str, state: Dict):
    df = fetch_df(ex, symbol, CANDLES_LIMIT)

    # only evaluate on the last CLOSED bar (second to last)
    if len(df) < 80:
        return
    idx_close = len(df) - 2
    row = df.loc[idx_close]

    # per-symbol state
    st = state.setdefault(symbol, {
        "last_bull_idm_idx": None,
        "last_bear_idm_idx": None,
        "last_flip_alerts": [],
        "last_hi_sweep_idx": None,
        "last_lo_sweep_idx": None
    })

    messages: List[str] = []

    # ------ IDM + BOS (BULLISH) ------
    if bos_up(df, idx_close, BOS_LEN):
        if (not REQUIRE_VOL_EXPANSION) or vol_expanded(df, idx_close, VOL_MA, VOL_FACTOR):
            idm_idx = find_idm_bull(df, idx_close, IDM_LOOKBACK)
            if idm_idx is not None and st.get("last_bull_idm_idx") != idm_idx:
                zone_h = df.loc[idm_idx, "high"]; zone_l = df.loc[idm_idx, "low"]
                ob_body_low  = min(df.loc[idm_idx, "open"], df.loc[idm_idx, "close"])
                ob_body_high = max(df.loc[idm_idx, "open"], df.loc[idm_idx, "close"])

                lines = [
                    f"{ALERT_PREFIX} • {symbol}",
                    "🟢 IDM + BOS (bullish)",
                    f"IDM/OB bar: {df.loc[idm_idx,'dt']:%Y-%m-%d %H:%M} UTC",
                    f"OB wick:  {fmt_price(zone_l)}–{fmt_price(zone_h)}",
                    f"OB body:  {fmt_price(ob_body_low)}–{fmt_price(ob_body_high)}",
                    "Plan: watch 4H/LTF retrace into OB."
                ]
                # range type hint (optional)
                rtype = detect_range_type(df, idx_close)
                if rtype:
                    lines.append(f"Pattern: {rtype}")

                messages.append("\n".join(lines))
                st["last_bull_idm_idx"] = idm_idx

                # flip-through
                if row["close"] > zone_h:
                    flip_msg = f"{symbol} • Flip confirmed: supply→demand (close above {fmt_price(zone_h)})."
                    if flip_msg not in st["last_flip_alerts"]:
                        messages.append(f"{ALERT_PREFIX} • {flip_msg}")
                        st["last_flip_alerts"].append(flip_msg)

                # first retest
                touch_idx = first_retest(df, zone_h, zone_l, idx_close)
                if touch_idx is not None:
                    messages.append(
                        f"{ALERT_PREFIX} • {symbol}\n🟢 First retest of Bullish OB at "
                        f"{fmt_price(zone_l)}–{fmt_price(zone_h)} on {df.loc[touch_idx,'dt'].date()}."
                    )
        else:
            # optional informational line (not sent as alert): no volume expansion
            pass

    # ------ IDM + BOS (BEARISH) ------
    if bos_down(df, idx_close, BOS_LEN):
        if (not REQUIRE_VOL_EXPANSION) or vol_expanded(df, idx_close, VOL_MA, VOL_FACTOR):
            idm_idx = find_idm_bear(df, idx_close, IDM_LOOKBACK)
            if idm_idx is not None and st.get("last_bear_idm_idx") != idm_idx:
                zone_h = df.loc[idm_idx, "high"]; zone_l = df.loc[idm_idx, "low"]
                ob_body_low  = min(df.loc[idm_idx, "open"], df.loc[idm_idx, "close"])
                ob_body_high = max(df.loc[idm_idx, "open"], df.loc[idm_idx, "close"])

                messages.append(
                    f"{ALERT_PREFIX} • {symbol}\n🔴 IDM + BOS (bearish)\n"
                    f"IDM/OB bar: {df.loc[idm_idx,'dt']:%Y-%m-%d %H:%M} UTC\n"
                    f"OB wick:  {fmt_price(zone_l)}–{fmt_price(zone_h)}\n"
                    f"OB body:  {fmt_price(ob_body_low)}–{fmt_price(ob_body_high)}\n"
                    f"Plan: watch retrace into OB."
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
                        f"{ALERT_PREFIX} • {symbol}\n🔴 First retest of Bearish OB at "
                        f"{fmt_price(zone_l)}–{fmt_price(zone_h)} on {df.loc[touch_idx,'dt'].date()}."
                    )
        else:
            pass

    # ------ Liquidity sweeps (latest extremes) ------
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
    print(f"[{now_utc_str()}] Starting SMC 4H scanner…")
    ex = init_exchange()

    # Symbols supported on this exchange + volume filter
    symbols = [s for s in WATCHLIST if s in ex.symbols]
    if not symbols:
        print("No WATCHLIST symbols available on selected exchange; scanning all majors fallback.")
        majors = [s for s in ex.symbols if s.endswith("/USDT")]
        symbols = filter_by_volume(ex, sorted(majors)[:20])
    else:
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
# =================== END SMC 4H SCANNER ===================
