# INDICATOR AND NOTIFICATION

An algorithmic trading copilot that combines **multi-timeframe technical analysis** (1h macro trend + 15m RSI/ATR triggers) with **breaking news sentiment & red-flag veto filters**, delivering instant push notifications to your phone via [ntfy.sh](https://ntfy.sh).

## Features
- **Multi-Timeframe Architecture**: 1-hour 50/200 EMA trend detection paired with 15-minute execution triggers.
- **Dynamic Volatility Stops**: Stop-loss and targets dynamically calibrated using Wilder's Average True Range (ATR).
- **Regime & Volume Filters**: ADX trend strength filtering and volume spike protection.
- **Breaking News Circuit Breaker**: Scrapes real-time RSS feeds (CoinDesk, CoinTelegraph) to veto entries during exploits, lawsuits, or high-impact negative catalysts.
- **24/7 Cloud Automation**: Runs automatically on GitHub Actions every 30 minutes without needing your local machine to stay awake.
- **Zero-Cost Push Alerts**: Dispatches actionable trade cards to your phone lock screen via `ntfy.sh`.

## Architecture
```
[ Binance Candlestick Feeds ] ──┐
  (1h Trend + 15m Execution)    ├──► [ Multi-Timeframe Engine ] ──┐
                                                                   ├──► [ Synthesizer & Conflict Filter ] ──► [ ntfy.sh Push Alert ]
[ CoinDesk / CoinTelegraph ]   ──► [ News Sentiment & Veto ]   ──┘
  (Real-Time Breaking RSS)
```

## Quick Start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/main.py --once
```
