import ccxt
import time
import argparse
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

from data_engine import fetch_multi_timeframe, coins, get_binance_exchange, fetch_order_book_imbalance
from indicators import analyze_technicals, kelly_position_size
from news_engine import analyze_news
from ml_engine import predict_probability
from notifier import send_alert, DEFAULT_NTFY_TOPIC

# ======================================================================
# CONFIGURATION
# ======================================================================
# Your ntfy.sh topic for instant iPhone / Android push notifications:
NTFY_TOPIC = DEFAULT_NTFY_TOPIC  # 'ben-secret107468'

# Optional iMessage phone / Apple ID (used as secondary alert if on Mac)
ALERT_RECIPIENT = "+1234567890"

# Set to "spot" (Buy dips + Hold cash) or "futures" (Longs + Shorts)
TRADING_MODE = "spot"

# Default automatic scan interval in minutes (30 minutes aligned with :00 and :30)
SCAN_INTERVAL_MINUTES = 30

# Cooldown between repeated alerts for the same coin (in hours)
ALERT_COOLDOWN_HOURS = 2.0

# Cache to track when each coin last dispatched an alert
last_alert_times = {}


def synthesize_trade_decision(symbol: str, exchange, btc_state: dict = None) -> dict:
    """
    Binds Technical Indicators, Candlestick Reversals, Order Book Imbalance,
    LightGBM Statistical Edge, and News Sentiment into one unified verdict.
    """
    # 1. RUN TECHNICAL ANALYSIS & CANDLESTICK CHECKS
    candles = fetch_multi_timeframe(exchange, symbol)
    tech = analyze_technicals(candles['trend'], candles['trigger'], mode=TRADING_MODE)

    if tech.get("status") != "SUCCESS":
        return {"status": "ERROR", "message": tech.get("message")}

    # 2. RUN ORDER BOOK DEPTH ANALYSIS
    ob_info = fetch_order_book_imbalance(exchange, symbol, depth=15)

    # 3. RUN BREAKING NEWS SENTIMENT
    news = analyze_news(symbol)

    # 4. EXTRACT CORE METRICS
    raw_signal     = tech["signal"]
    action         = tech["action"]
    live_price     = tech["live_price"]
    entry_limit    = tech.get("entry_limit")
    stop_loss      = tech.get("stop_loss")
    tp1            = tech.get("tp1")
    tp2            = tech.get("tp2")
    macro_trend    = tech["indicators"]["macro_trend_1h"]
    rsi            = tech["indicators"]["rsi_15m"]
    adx            = tech["indicators"]["adx_15m"]
    vol_ratio      = tech["indicators"]["volume_ratio"]

    news_sentiment = news["sentiment"]
    is_dangerous   = news["is_dangerous"]
    top_headline   = news["top_headline"]

    # Candlestick characteristics
    df_15m = candles['trigger']
    if len(df_15m) >= 2:
        last_c = df_15m.iloc[-1]
        c_range = max(float(last_c['high']) - float(last_c['low']), 1e-8)
        lower_wick_ratio = max(0.0, min(float(last_c['open']), float(last_c['close'])) - float(last_c['low'])) / c_range
        candle_is_green = 1.0 if float(last_c['close']) >= float(last_c['open']) else 0.0
    else:
        lower_wick_ratio = 0.25
        candle_is_green = 0.0

    # 5. RUN LIGHTGBM STATISTICAL WIN PREDICTOR
    ml_features = {
        'rsi_15m': rsi,
        'macd_hist_15m': tech['indicators']['macd_hist_15m'],
        'adx_15m': adx,
        'volume_ratio': vol_ratio,
        'extension_from_ema50': float(tech['gaps']['extension_from_ema50'].replace('%', '')),
        'ema50_over_ema200': 1.0 if tech['indicators']['ema50_1h'] > tech['indicators']['ema200_1h'] else 0.0,
        'lower_wick_ratio': lower_wick_ratio,
        'candle_is_green': candle_is_green,
        'ob_ratio': ob_info.get('ratio', 1.0),
        'btc_trend_bullish': 1.0 if (btc_state and btc_state.get('macro_trend') == 'BULLISH') else 0.0
    }
    ml_eval = predict_probability(ml_features)

    # Calculate Kelly recommended position sizing
    win_prob = ml_eval['prob']
    kelly = kelly_position_size(win_rate=win_prob, rr_ratio=1.5, capital=1000.0, price=live_price)

    # 6. BINDING & CONFLICT FILTERING
    final_decision = raw_signal
    is_approved    = False
    veto_reason    = None
    combined_notes = list(tech.get("reasons", []))

    # Confluence notes
    combined_notes.append(f"📊 Order Book: {ob_info['pressure']} (B/A Ratio: {ob_info['ratio']:.2f})")
    combined_notes.append(f"🌲 Statistical Win Probability: {ml_eval['prob_pct']} ({ml_eval['verdict']})")

    if tech.get("block_reason"):
        veto_reason = tech["block_reason"]
        final_decision = "⚠️ BLOCKED BY TECHNICAL FILTER"

    # BTC Market Leader Veto: Altcoins must not fight Bitcoin downtrend
    elif symbol != "BTC/USDT" and btc_state and "BUY" in raw_signal:
        btc_trend = btc_state.get("macro_trend", "SIDEWAYS")
        btc_price = btc_state.get("live_price", 0)
        btc_ema50 = btc_state.get("ema50_1h", 0)
        if btc_trend == "BEARISH" or (btc_ema50 > 0 and btc_price < btc_ema50):
            final_decision = "⚠️ BLOCKED (BTC MARKET LEADER VETO)"
            veto_reason = f"BTC 1h trend is BEARISH (${btc_price:,.2f} < EMA50 ${btc_ema50:,.2f}). Altcoin longs carry dump risk."
            combined_notes.append(f"⛔ VETO: {veto_reason}")

    # Order Book Sell Wall Veto
    elif "BUY" in raw_signal and ob_info.get("ratio", 1.0) < 0.45:
        final_decision = "⚠️ BLOCKED (HEAVY SELL WALL)"
        veto_reason = f"Order book asks heavily outweigh bids ({ob_info['ratio']:.2f} B/A ratio). Resistance overhead."
        combined_notes.append(f"⛔ VETO: {veto_reason}")

    # Dangerous News Veto
    elif is_dangerous and "BUY" in raw_signal:
        final_decision = "🚨 TRADE VETOED BY DANGEROUS NEWS"
        veto_reason = f"Red-flag headline: \"{top_headline}\""
        combined_notes.append(f"⛔ VETO: {veto_reason}")

    elif "BUY" in raw_signal and news_sentiment == "BEARISH":
        final_decision = "⚠️ SKIPPED (NEWS CONFLICT)"
        veto_reason = f"Technicals show pullback, but news sentiment is BEARISH."
        combined_notes.append(f"⛔ CONFLICT: {veto_reason}")

    elif "SHORT" in raw_signal and news_sentiment == "BULLISH":
        final_decision = "⚠️ SKIPPED (NEWS CONFLICT)"
        veto_reason = f"Technicals show short setup, but breaking news is BULLISH."
        combined_notes.append(f"⛔ CONFLICT: {veto_reason}")

    # Statistical Probability Threshold: Must have >= 55% statistical edge
    elif raw_signal in ["BUY SIGNAL", "SELL (SHORT)"] and win_prob < 0.55:
        final_decision = "⚠️ BLOCKED (LOW STATISTICAL EDGE)"
        veto_reason = f"LightGBM statistical win probability is {ml_eval['prob_pct']} (< 55.0% threshold)."
        combined_notes.append(f"⛔ VETO: {veto_reason}")

    elif raw_signal in ["BUY SIGNAL", "SELL (SHORT)"]:
        is_approved = True
        combined_notes.append(f"✅ Fundamental alignment: News ({news_sentiment}) does not contradict trade.")
        combined_notes.append(f"✅ Quantitative edge confirmed by LightGBM ({ml_eval['prob_pct']}).")

    return {
        "status": "SUCCESS",
        "symbol": symbol,
        "mode": TRADING_MODE.upper(),
        "is_approved": is_approved,
        "final_decision": final_decision,
        "action": action,
        "live_price": live_price,
        "entry_limit": entry_limit,
        "stop_loss": stop_loss,
        "tp1": tp1,
        "tp2": tp2,
        "gaps": tech.get("gaps", {}),
        "macro_trend": macro_trend,
        "rsi": rsi,
        "adx": adx,
        "vol_ratio": vol_ratio,
        "order_book": ob_info,
        "ml_eval": ml_eval,
        "kelly": kelly,
        "news_sentiment": news_sentiment,
        "top_headline": top_headline,
        "veto_reason": veto_reason,
        "reasons": combined_notes,
        "indicators": tech.get("indicators", {})
    }


def format_alert_title(d: dict) -> str:
    """
    Creates a prominent lock-screen notification title that displays
    the coin, entry price, and stop-loss directly in the notification banner.
    No 'AI' word used.
    Example: 🟢 BUY BTC/USDT @ $81,350.00 | SL: $79,800.00
    """
    direction_emoji = "🟢" if "BUY" in d['action'] else "🔴"
    action_short = "BUY" if "BUY" in d['action'] else "SHORT"
    entry_str = f"${d['entry_limit']:,.2f}" if d.get('entry_limit') else f"${d['live_price']:,.2f}"
    sl_str = f"${d['stop_loss']:,.2f}" if d.get('stop_loss') else "N/A"
    return f"{direction_emoji} {action_short} {d['symbol']} @ {entry_str} | SL: {sl_str}"


def format_alert_body(d: dict) -> str:
    """
    Formats an approved trade setup into a structured trade decision card.
    No 'AI' word used.
    """
    risk_pct = d['gaps'].get('risk_pct', 'N/A')
    tp1_pct = d['gaps'].get('tp1_pct', 'N/A')
    tp2_pct = d['gaps'].get('tp2_pct', 'N/A')
    ob = d.get('order_book', {})
    ml = d.get('ml_eval', {})
    kelly = d.get('kelly', {})

    return (
        f"🚨 TRADE SIGNAL: {d['symbol']} ({d['mode']})\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Action         : {d['action']}\n"
        f"📍 Entry Price : ${d['entry_limit']:,.2f}\n"
        f"🛑 Stop Loss at: ${d['stop_loss']:,.2f} ({risk_pct} Risk)\n"
        f"🎯 Target 1 at : ${d['tp1']:,.2f} ({tp1_pct} Gain | 1.5R)\n"
        f"🎯 Target 2 at : ${d['tp2']:,.2f} ({tp2_pct} Gain | 2.5R)\n"
        f"💵 Current Live: ${d['live_price']:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 1h Trend    : {d['macro_trend']}\n"
        f"📊 15m RSI     : {d['rsi']}\n"
        f"🧱 Order Book  : {ob.get('pressure', 'BALANCED')} (B/A Ratio: {ob.get('ratio', 1.0)})\n"
        f"🎯 Win Probability: {ml.get('prob_pct', 'N/A')} ({ml.get('verdict', 'CALIBRATED')})\n"
        f"⚖️ Position Size  : {kelly.get('risk_pct', '1.5')}% Risk (Half-Kelly Criterion)\n"
        f"📰 News        : {d['news_sentiment']}\n"
        f"Top Story      : \"{d['top_headline']}\"\n\n"
        f"💡 Thesis      : {d['reasons'][0]}"
    )


def run_scanner_cycle(exchange, is_scheduled_summary: bool = False, is_manual_run: bool = False):
    """
    Executes a single scan cycle across all monitored coins.
    - Evaluates Bitcoin market health first for cross-market alignment.
    - If actionable trade signal: immediately dispatches high-priority alert.
    - If scheduled 9:00 AM / 7:00 PM briefing or manual run: sends briefing even if 0 setups.
    """
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"📡 SCAN CYCLE STARTED: {now_str}")
    print(f"{'='*60}")

    approved_count = 0
    scanned_results = []

    # 1. Evaluate Bitcoin first to extract benchmark state for altcoins
    btc_decision = synthesize_trade_decision("BTC/USDT", exchange, btc_state=None)
    btc_state = {
        "macro_trend": btc_decision.get("macro_trend", "SIDEWAYS"),
        "live_price": btc_decision.get("live_price", 0),
        "ema50_1h": btc_decision.get("indicators", {}).get("ema50_1h", 0)
    } if btc_decision.get("status") == "SUCCESS" else None

    # 2. Iterate through all monitored coins
    for pair in coins:
        try:
            if pair == "BTC/USDT":
                d = btc_decision
            else:
                d = synthesize_trade_decision(pair, exchange, btc_state=btc_state)

            if d.get("status") != "SUCCESS":
                print(f"⚠️ {pair}: {d.get('message')}")
                continue

            scanned_results.append(d)
            status_icon = "🟢" if d['is_approved'] else "⚪"
            ob_ratio = d.get('order_book', {}).get('ratio', 1.0)
            ml_prob = d.get('ml_eval', {}).get('prob_pct', 'N/A')
            print(f"{status_icon} {pair:<10} | {d['final_decision']:<26} | Trend: {d['macro_trend']:<7} | RSI: {d['rsi']:<4} | OB: {ob_ratio:.2f} | WinProb: {ml_prob}")

            # Check if setup is approved
            if d['is_approved']:
                approved_count += 1
                now = datetime.now()
                last_time = last_alert_times.get(pair)

                # Check cooldown to prevent duplicate alerts
                if last_time and (now - last_time).total_seconds() < (ALERT_COOLDOWN_HOURS * 3600):
                    print(f"   ⏳ Cooldown active for {pair} (Alerted within last {ALERT_COOLDOWN_HOURS}h). Skipping dispatch.")
                else:
                    print(f"   🔥 Approved setup found! Dispatching push alert for {pair}...")
                    alert_title = format_alert_title(d)
                    alert_body = format_alert_body(d)
                    sent = send_alert(alert_body, title=alert_title, imessage_recipient=ALERT_RECIPIENT)
                    if sent:
                        last_alert_times[pair] = now

        except Exception as e:
            print(f"❌ Error during analysis of {pair}: {e}")

    print(f"{'='*60}")
    print(f"🏁 Cycle complete. {approved_count} actionable setup(s) found.")

    # If this was a scheduled 9:00 AM / 7:00 PM briefing OR a manual workflow run, send summary notification
    if (is_scheduled_summary or is_manual_run) and approved_count == 0:
        dispatch_briefing_notification(scanned_results, is_manual=is_manual_run)


def dispatch_briefing_notification(results: list, is_manual: bool = False):
    """Sends a market briefing at 9:00 AM / 7:00 PM or upon manual workflow dispatch."""
    ny_tz = ZoneInfo("America/New_York")
    now_ny = datetime.now(ny_tz)
    time_str = now_ny.strftime("%I:%M %p %Z")

    title = "Market Check" if is_manual else f"Market Briefing ({now_ny.strftime('%I:%M %p')})"

    body_lines = [
        f"📊 {title} - {time_str}",
        "━━━━━━━━━━━━━━━━━━━━━━",
        f"Monitored: {len(results)} pairs | Setups: 0 (Consolidation/Wait)",
        ""
    ]
    for r in results:
        trend_arrow = "↗️" if r['macro_trend'] == "BULLISH" else ("↘️" if r['macro_trend'] == "BEARISH" else "➡️")
        ob_ratio = r.get('order_book', {}).get('ratio', 1.0)
        prob = r.get('ml_eval', {}).get('prob_pct', 'N/A')
        body_lines.append(f"• {r['symbol']}: ${r['live_price']:,.2f} | 1h: {trend_arrow} {r['macro_trend']} | RSI: {r['rsi']} | Depth: {ob_ratio:.2f}")

    body_lines.append("\n💡 Monitoring 24/7 with LightGBM. Will alert immediately when a high-probability trade confirms.")
    summary_message = "\n".join(body_lines)

    send_alert(
        summary_message,
        title=f"📊 {title}: {len(results)} pairs scanned",
        priority="default",
        tags="bar_chart,satellite"
    )


def seconds_until_next_candle_close(interval_minutes: int = 30) -> int:
    """
    Calculates exact seconds until the next 30-minute boundary
    (:00 or :30) plus 5 seconds buffer for exchange candle finalization.
    """
    now = datetime.now()
    minute = now.minute
    second = now.second

    minutes_left = interval_minutes - (minute % interval_minutes)
    target_seconds = (minutes_left * 60) - second + 5

    if target_seconds <= 0:
        target_seconds += interval_minutes * 60

    return target_seconds


def run_continuous_daemon(interval_minutes: int = 30):
    """
    Runs the scanner continuously on an automatic schedule, synchronized
    to every 30-minute boundary (:00 and :30).
    """
    exchange = get_binance_exchange()
    print(f"🚀 Crypto Scanner Daemon Online")
    print(f"Data Source       : Binance Official Data API (Vision Mirror)")
    print(f"Scheduled Interval: Every {interval_minutes} minutes (Aligned with :00 and :30 boundaries)")
    print(f"Monitoring Pairs  : {', '.join(coins)}")
    print(f"Alert Channel     : https://ntfy.sh/{NTFY_TOPIC}")
    print("Press Ctrl + C to stop at any time.\n")

    # Run immediate scan on launch
    run_scanner_cycle(exchange)

    # Loop forever
    while True:
        try:
            wait_sec = seconds_until_next_candle_close(interval_minutes)
            next_time = (datetime.now() + timedelta(seconds=wait_sec)).strftime('%H:%M:%S')
            print(f"\n💤 Sleeping {wait_sec // 60}m {wait_sec % 60}s until next boundary ({next_time})...")

            time.sleep(wait_sec)

            # Check if this cycle is 9:00 AM or 7:00 PM (19:00) EDT/EST
            now_ny = datetime.now(ZoneInfo("America/New_York"))
            is_briefing = (now_ny.hour in [9, 19] and now_ny.minute < 30)
            run_scanner_cycle(exchange, is_scheduled_summary=is_briefing)

        except KeyboardInterrupt:
            print("\n👋 Daemon stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error in daemon loop: {e}. Retrying in 60s...")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Market Scanner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan cycle and exit immediately"
    )
    parser.add_argument(
        "--briefing",
        action="store_true",
        help="Force send a market briefing summary (used for 9:00 AM / 7:00 PM scheduled runs)"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Flag indicating a manual trigger from GitHub Actions"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=SCAN_INTERVAL_MINUTES,
        help=f"Interval in minutes for continuous scanning (default: {SCAN_INTERVAL_MINUTES})"
    )

    args = parser.parse_args()

    if args.once:
        exchange = get_binance_exchange()
        now_ny = datetime.now(ZoneInfo("America/New_York"))
        auto_briefing = args.briefing or (now_ny.hour in [9, 19] and now_ny.minute < 30)
        run_scanner_cycle(exchange, is_scheduled_summary=auto_briefing, is_manual_run=args.manual)
    else:
        run_continuous_daemon(interval_minutes=args.interval)
