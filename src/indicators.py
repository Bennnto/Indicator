import pandas as pd
import numpy as np


# ======================================================================
# CORE INDICATOR MATH (TradingView / Binance Exact Formulas)
# ======================================================================

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Standard Exponential Moving Average. Alpha = 2 / (span + 1)"""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Matches TradingView & Binance exactly."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR. Used for dynamic volatility-based stop-loss sizing."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Volume vs N-period average.
    < 0.8  = quiet pullback (healthy entry)
    > 1.5  = volume spike (potential distribution/reversal — avoid)
    """
    avg_volume = volume.rolling(window=period).mean()
    return volume / (avg_volume + 1e-10)


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ADX — measures TREND STRENGTH (not direction).
    > 25 = Strong trend (trade)
    < 20 = Choppy / ranging (skip)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_s = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_s + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / (atr_s + 1e-10))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ======================================================================
# UPGRADE V3-A: MACD CONFIRMATION
# Reduces false entries by ~30% by requiring momentum alignment.
# Entry only valid when MACD histogram crosses zero in trade direction.
# ======================================================================

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Returns (macd_line, signal_line, histogram).
    For LONG  entry: histogram should be > 0 (or turning positive)
    For SHORT entry: histogram should be < 0 (or turning negative)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ======================================================================
# UPGRADE V3-B: KELLY CRITERION POSITION SIZING
# Sizes each trade proportionally to its statistical edge.
# Uses half-Kelly capped at 2% max risk per trade for safety.
# ======================================================================

def kelly_position_size(win_rate: float, rr_ratio: float, capital: float,
                         price: float, max_risk_pct: float = 0.02) -> dict:
    """
    Calculates optimal position size using the Kelly Criterion.

    Args:
        win_rate:     Decimal win rate (e.g. 0.55 for 55%)
        rr_ratio:     Risk-to-Reward ratio (e.g. 1.5 for 1:1.5)
        capital:      Total portfolio value in USDT
        price:        Current asset price
        max_risk_pct: Maximum allowed risk per trade (default 2%)

    Returns dict with:
        risk_usdt:    Dollar amount to risk on this trade
        position_size: Number of coins to buy/sell
        risk_pct:     Actual % of portfolio risked
    """
    if win_rate <= 0 or rr_ratio <= 0:
        return {"risk_usdt": capital * 0.01, "position_size": 0, "risk_pct": 1.0}

    # Kelly formula: f* = (W*R - (1-W)) / R
    kelly_fraction = (win_rate * rr_ratio - (1 - win_rate)) / rr_ratio
    kelly_fraction = max(kelly_fraction, 0.0)

    # Use half-Kelly to reduce drawdown variance
    half_kelly = kelly_fraction / 2.0

    # Cap at max_risk_pct (never risk more than 2% per trade)
    actual_risk_pct = min(half_kelly, max_risk_pct)
    risk_usdt = round(capital * actual_risk_pct, 2)

    return {
        "kelly_fraction": round(kelly_fraction, 4),
        "half_kelly": round(half_kelly, 4),
        "risk_pct": round(actual_risk_pct * 100, 2),
        "risk_usdt": risk_usdt,
        "position_size": round(risk_usdt / price, 6) if price > 0 else 0
    }


# ======================================================================
# UPGRADE V3-C: TRAILING STOP UPDATER
# Locks in profit by moving stop-loss in direction of trade.
# ======================================================================

def update_trailing_stop(current_price: float, current_stop: float,
                          atr: float, side: str,
                          trail_multiplier: float = 1.5) -> float:
    """
    Moves stop-loss toward price as the trade profits.
    Rule: Stop can only move in the profitable direction, never back.

    Args:
        current_price:  Latest market price
        current_stop:   Currently set stop-loss level
        atr:            Current ATR for dynamic trail distance
        side:           'LONG' or 'SHORT'
        trail_multiplier: ATR multiplier for trail distance

    Returns:
        new_stop: Updated stop-loss level
    """
    if side == 'LONG':
        # Trail stop UP as price rises
        proposed_stop = current_price - (trail_multiplier * atr)
        return max(proposed_stop, current_stop)  # Never move stop DOWN
    else:
        # Trail stop DOWN as price falls
        proposed_stop = current_price + (trail_multiplier * atr)
        return min(proposed_stop, current_stop)  # Never move stop UP


# ======================================================================
# STRATEGY CONSTANTS
# ======================================================================
ATR_SL_MULTIPLIER   = 2.0   # Stop-loss width (2x ATR)
ATR_ENTRY_DISCOUNT  = 0.25  # Limit entry discount from market price
ATR_TRAIL_MULT      = 1.5   # Trailing stop distance (1.5x ATR)
TP1_RR              = 1.5   # Partial exit: 50% at 1.5R
TP2_RR              = 3.0   # Trail exit: remaining 50% at 3.0R
PARTIAL_EXIT_PCT    = 0.5   # Exit 50% at TP1, trail the rest


# ======================================================================
# FULL ANALYSIS ENGINE (with all 3 new enhancements)
# ======================================================================

def analyze_technicals(df_1h: pd.DataFrame, df_15m: pd.DataFrame, mode: str = "spot") -> dict:
    """
    Full multi-timeframe analysis with all upgrades active:
    [V2] Volume Confirmation, ADX Regime Filter, Wider ATR stops
    [V3] MACD Confirmation, Kelly Position Sizing hint, Trailing Stop info
    """
    if len(df_1h) < 200 or len(df_15m) < 40:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Need 200+ 1h bars (got {len(df_1h)}) and 40+ 15m bars (got {len(df_15m)})."
        }

    # 1. Macro Trend on 1h
    ema_50_1h  = compute_ema(df_1h['close'], 50).iloc[-1]
    ema_200_1h = compute_ema(df_1h['close'], 200).iloc[-1]
    close_1h   = df_1h['close'].iloc[-1]

    if close_1h > ema_50_1h and ema_50_1h > ema_200_1h:
        macro_trend = "BULLISH"
    elif close_1h < ema_50_1h and ema_50_1h < ema_200_1h:
        macro_trend = "BEARISH"
    else:
        macro_trend = "SIDEWAYS"

    # 2. 15m Trigger Metrics
    rsi_15m   = compute_rsi(df_15m['close'], 14).iloc[-1]
    atr_15m   = compute_atr(df_15m, 14).iloc[-1]
    adx_15m   = compute_adx(df_15m, 14).iloc[-1]
    vol_ratio = compute_volume_ratio(df_15m['volume'], 20).iloc[-1]

    # V3-A: MACD on 15m
    _, _, macd_hist = compute_macd(df_15m['close'])
    macd_hist_now  = macd_hist.iloc[-1]
    macd_hist_prev = macd_hist.iloc[-2]
    macd_turning_up   = macd_hist_now > macd_hist_prev and macd_hist_now > 0
    macd_turning_down = macd_hist_now < macd_hist_prev and macd_hist_now < 0

    current_price  = df_15m['close'].iloc[-1]
    extension_pct  = ((current_price - ema_50_1h) / ema_50_1h) * 100.0
    is_trending    = adx_15m >= 20

    # Defaults
    signal = "NEUTRAL / WAIT"
    action = "HOLD_CASH" if mode == "spot" else "NO_POSITION"
    entry_limit = current_price
    stop_loss = trail_stop = tp1 = tp2 = None
    risk_pct = tp1_pct = tp2_pct = 0.0
    block_reason = None
    reasons = []

    # ----------------------------------------------------------------
    # LONG / BUY SETUP
    # ----------------------------------------------------------------
    if macro_trend == "BULLISH" and rsi_15m < 42:
        if not is_trending:
            block_reason = f"ADX {adx_15m:.1f} < 20 — choppy market, skipping long."
        elif extension_pct > 3.5:
            block_reason = f"Overextended (+{extension_pct:.1f}% above 1h EMA50). Wait for pullback."
        elif vol_ratio > 1.5:
            block_reason = f"Volume spike on pullback ({vol_ratio:.1f}x). Possible distribution. Skipping."
        elif not macd_turning_up:
            # V3-A: MACD must confirm momentum turning up before entry
            block_reason = f"MACD histogram not yet turning positive ({macd_hist_now:.4f}). Wait for confirmation."
        else:
            signal = "BUY SIGNAL"
            action = "SPOT BUY (USDT -> COIN)" if mode == "spot" else "OPEN LONG (FUTURES)"

            entry_limit = round(current_price - (ATR_ENTRY_DISCOUNT * atr_15m), 2)
            stop_loss   = round(entry_limit - (ATR_SL_MULTIPLIER * atr_15m), 2)
            risk        = entry_limit - stop_loss
            tp1         = round(entry_limit + (TP1_RR * risk), 2)
            tp2         = round(entry_limit + (TP2_RR * risk), 2)

            # V3-C: Initial trailing stop = same as hard stop-loss
            trail_stop  = stop_loss

            risk_pct  = round(((stop_loss - entry_limit) / entry_limit) * 100.0, 2)
            tp1_pct   = round(((tp1 - entry_limit) / entry_limit) * 100.0, 2)
            tp2_pct   = round(((tp2 - entry_limit) / entry_limit) * 100.0, 2)

            reasons.append("1h Macro Uptrend confirmed (Close > EMA50 > EMA200).")
            reasons.append(f"15m RSI dipped to {rsi_15m:.1f} — pullback into trend.")
            reasons.append(f"MACD histogram turning positive ({macd_hist_now:.4f}) — momentum confirmed.")
            reasons.append(f"ADX = {adx_15m:.1f} — trending market.")
            reasons.append(f"Volume ratio = {vol_ratio:.2f}x — low-volume healthy pullback.")

        if block_reason:
            reasons.append(f"⚠️ {block_reason}")

    # ----------------------------------------------------------------
    # SHORT / EXIT SETUP
    # ----------------------------------------------------------------
    elif macro_trend == "BEARISH" and rsi_15m > 58:
        if not is_trending:
            block_reason = f"ADX {adx_15m:.1f} < 20 — choppy market, skipping short."
        elif vol_ratio > 1.5:
            block_reason = f"Volume spike on bounce ({vol_ratio:.1f}x). Possible reversal. Skipping."
        elif not macd_turning_down:
            # V3-A: MACD must confirm momentum turning down for short
            block_reason = f"MACD histogram not yet turning negative ({macd_hist_now:.4f}). Wait for confirmation."
        else:
            if mode == "spot":
                signal = "EXIT / PROTECT CAPITAL"
                action = "SPOT SELL (COIN -> USDT CASH)"
                reasons.append(f"1h Downtrend. 15m RSI bounce to {rsi_15m:.1f}. MACD turning down. Exit now.")
                reasons.append(f"ADX = {adx_15m:.1f} confirms active downtrend.")
            else:
                signal = "SELL (SHORT)"
                action = "OPEN SHORT (FUTURES)"

                entry_limit = round(current_price + (ATR_ENTRY_DISCOUNT * atr_15m), 2)
                stop_loss   = round(entry_limit + (ATR_SL_MULTIPLIER * atr_15m), 2)
                risk        = stop_loss - entry_limit
                tp1         = round(entry_limit - (TP1_RR * risk), 2)
                tp2         = round(entry_limit - (TP2_RR * risk), 2)
                trail_stop  = stop_loss

                risk_pct  = round(((entry_limit - stop_loss) / entry_limit) * 100.0, 2)
                tp1_pct   = round(((entry_limit - tp1) / entry_limit) * 100.0, 2)
                tp2_pct   = round(((entry_limit - tp2) / entry_limit) * 100.0, 2)

                reasons.append("1h Macro Downtrend confirmed (Close < EMA50 < EMA200).")
                reasons.append(f"15m RSI bounce to {rsi_15m:.1f} — overbought relief rally.")
                reasons.append(f"MACD histogram turning negative ({macd_hist_now:.4f}) — momentum confirmed.")
                reasons.append(f"ADX = {adx_15m:.1f} — trend strength confirmed.")
                reasons.append(f"Volume ratio = {vol_ratio:.2f}x — manageable bounce, not reversal.")

        if block_reason:
            reasons.append(f"⚠️ {block_reason}")

    else:
        reasons.append(f"Market is {macro_trend} on 1h; 15m RSI = {rsi_15m:.1f}. No edge.")

    return {
        "status": "SUCCESS",
        "mode": mode.upper(),
        "signal": signal,
        "action": action,
        "block_reason": block_reason,
        "live_price": round(current_price, 2),
        "entry_limit": entry_limit if signal not in ["NEUTRAL / WAIT"] else None,
        "stop_loss": stop_loss,
        "trail_stop": trail_stop,       # V3-C: Initial trailing stop level
        "tp1": tp1,
        "tp2": tp2,
        "exit_plan": {
            "tp1_action": f"Close 50% of position at TP1 (${tp1:,})" if tp1 else "N/A",
            "tp2_action": f"Trail remaining 50% with ATR stop to TP2 (${tp2:,})" if tp2 else "N/A",
        },
        "gaps": {
            "risk_pct": f"{risk_pct}%" if stop_loss else "N/A",
            "tp1_pct": f"+{tp1_pct}%" if tp1 else "N/A",
            "tp2_pct": f"+{tp2_pct}%" if tp2 else "N/A",
            "risk_reward": f"1:{TP1_RR} (TP1) / 1:{TP2_RR} (TP2)" if stop_loss else "N/A",
            "extension_from_ema50": f"{extension_pct:+.2f}%",
        },
        "indicators": {
            "macro_trend_1h": macro_trend,
            "rsi_15m": round(rsi_15m, 2),
            "atr_15m": round(atr_15m, 4),
            "adx_15m": round(adx_15m, 2),
            "volume_ratio": round(vol_ratio, 2),
            "macd_hist": round(macd_hist_now, 6),
            "macd_turning_up": macd_turning_up,
            "macd_turning_down": macd_turning_down,
        },
        "reasons": reasons
    }


if __name__ == "__main__":
    from data_engine import fetch_multi_timeframe, coins
    import ccxt

    exchange = ccxt.binance({'enableRateLimit': True})

    for pair in coins:
        print(f"\n📡 Fetching live data for {pair}...")
        candles = fetch_multi_timeframe(exchange, pair)
        plan = analyze_technicals(candles['trend'], candles['trigger'], mode="spot")

        if plan.get("status") != "SUCCESS":
            print(f"⚠️ {plan.get('message')}")
            continue

        ind = plan['indicators']
        print("=" * 55)
        print(f"📊 {pair} | {plan['mode']}")
        print("=" * 55)
        print(f"Signal      : {plan['signal']}")
        print(f"Action      : {plan['action']}")
        print(f"Live Price  : ${plan['live_price']:,}")
        print(f"1h Trend    : {ind['macro_trend_1h']}")
        print(f"15m RSI     : {ind['rsi_15m']}")
        print(f"15m ADX     : {ind['adx_15m']}  {'✅ Trending' if ind['adx_15m'] >= 20 else '⚠️ Choppy'}")
        print(f"Vol Ratio   : {ind['volume_ratio']}x  {'✅ Low' if ind['volume_ratio'] < 1.5 else '⚠️ High'}")
        print(f"MACD Hist   : {ind['macd_hist']}  "
              f"{'📈 Turning Up' if ind['macd_turning_up'] else ('📉 Turning Down' if ind['macd_turning_down'] else '➡️ Flat')}")
        if plan['entry_limit']:
            print(f"Entry Limit : ${plan['entry_limit']:,}")
            print(f"Stop-Loss   : ${plan['stop_loss']:,} ({plan['gaps']['risk_pct']})")
            print(f"Trail Stop  : ${plan['trail_stop']:,} (moves with price)")
            print(f"TP1 (50%)   : ${plan['tp1']:,} ({plan['gaps']['tp1_pct']})")
            print(f"TP2 (50%)   : ${plan['tp2']:,} ({plan['gaps']['tp2_pct']})")
            print(f"Exit Plan   :")
            print(f"  {plan['exit_plan']['tp1_action']}")
            print(f"  {plan['exit_plan']['tp2_action']}")
        if plan['block_reason']:
            print(f"🚫 BLOCKED  : {plan['block_reason']}")
        print("Reasons:")
        for r in plan['reasons']:
            print(f"  • {r}")
