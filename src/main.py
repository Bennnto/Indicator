import ccxt
import time
import argparse
from datetime import datetime, timedelta

from data_engine import fetch_multi_timeframe, coins, get_binance_exchange
from indicators import analyze_technicals
from news_engine import analyze_news
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


def synthesize_trade_decision(symbol: str, exchange) -> dict:
    """
    Binds Technical Analysis and News Sentiment together into one final verdict.
    """
    # 1. RUN TECHNICAL ANALYSIS
    candles = fetch_multi_timeframe(exchange, symbol)
    tech = analyze_technicals(candles['trend'], candles['trigger'], mode=TRADING_MODE)

    if tech.get("status") != "SUCCESS":
        return {"status": "ERROR", "message": tech.get("message")}

    # 2. RUN BREAKING NEWS ANALYSIS
    news = analyze_news(symbol)

    # 3. EXTRACT CORE METRICS
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

    # 4. BINDING & CONFLICT FILTERING
    final_decision = raw_signal
    is_approved    = False
    veto_reason    = None
    combined_notes = list(tech.get("reasons", []))

    if tech.get("block_reason"):
        veto_reason = tech["block_reason"]
        final_decision = "⚠️ BLOCKED BY TECHNICAL FILTER"

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

    elif raw_signal in ["BUY SIGNAL", "SELL (SHORT)"]:
        is_approved = True
        combined_notes.append(f"✅ Fundamental alignment: News ({news_sentiment}) does not contradict trade.")

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
        "news_sentiment": news_sentiment,
        "top_headline": top_headline,
        "veto_reason": veto_reason,
        "reasons": combined_notes
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
        f"📰 News        : {d['news_sentiment']}\n"
        f"Top Story      : \"{d['top_headline']}\"\n\n"
        f"💡 Thesis      : {d['reasons'][0]}"
    )


def run_scanner_cycle(exchange):
    """Executes a single scan cycle across all monitored coins."""
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*60}")
    print(f"📡 SCAN CYCLE STARTED: {now_str}")
    print(f"{'='*60}")

    approved_count = 0

    for pair in coins:
        try:
            d = synthesize_trade_decision(pair, exchange)
            if d.get("status") != "SUCCESS":
                print(f"⚠️ {pair}: {d.get('message')}")
                continue

            status_icon = "🟢" if d['is_approved'] else "⚪"
            print(f"{status_icon} {pair:<10} | {d['final_decision']:<26} | Trend: {d['macro_trend']:<7} | RSI: {d['rsi']:<4} | News: {d['news_sentiment']}")

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
    print(f"🚀 Crypto Analyst Scanner Daemon Online")
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
            run_scanner_cycle(exchange)

        except KeyboardInterrupt:
            print("\n👋 Daemon stopped by user. Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Unexpected error in daemon loop: {e}. Retrying in 60s...")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Analyst Scanner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan cycle and exit immediately"
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
        run_scanner_cycle(exchange)
    else:
        run_continuous_daemon(interval_minutes=args.interval)
