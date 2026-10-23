import ccxt
import pandas as pd
import time
import os

# Monitored pairs
coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']


def get_binance_exchange():
    """
    Creates a Binance exchange client using the official Binance Data API mirror
    (data-api.binance.vision) for SPOT data.
    This public mirror allows unrestricted reading of live spot candles and order books
    from any global cloud provider (including US GitHub Actions runners) without HTTP 451.
    """
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'fetchMarkets': ['spot'],
        },
        'urls': {
            'api': {
                'public': 'https://data-api.binance.vision/api/v3',
                'fapiPublic': 'https://data-api.binance.vision/api/v3',
            }
        }
    })
    return exchange


def fetch_candles(exchange, symbol: str, timeframe: str = '15m', limit: int = 300) -> pd.DataFrame:
    """
    Fetches raw OHLCV candles from Binance Spot with automatic fallback to OKX
    if Binance is ever unreachable.
    """
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"⚠️ Binance data error for {symbol} ({timeframe}): {e}. Attempting fallback...")
        try:
            fallback_ex = ccxt.okx({'enableRateLimit': True})
            bars = fallback_ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if bars:
                df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
        except Exception as fb_err:
            print(f"❌ Fallback failed for {symbol}: {fb_err}")
        return pd.DataFrame()


def fetch_paginated_ohlcv(exchange, symbol: str, timeframe: str = '1h', days: int = 90) -> pd.DataFrame:
    """
    Fetches multiple months of historical data using pagination.
    Used by backtest.py to download sufficient history for multi-month backtesting.
    """
    all_bars = []
    limit = 500
    since_ms = int((pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).timestamp() * 1000)

    print(f"  📅 Fetching {days} days of {timeframe} data for {symbol}...")

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
        except Exception as e:
            print(f"  ❌ Pagination error: {e}")
            break

        if not bars:
            break

        all_bars += bars
        last_ts = bars[-1][0]
        since_ms = last_ts + 1

        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        if last_ts >= now_ms or len(bars) < limit:
            break

        time.sleep(0.3)

    if not all_bars:
        print(f"  ⚠️ No data returned for {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').reset_index(drop=True)
    return df


def fetch_multi_timeframe(exchange, symbol: str,
                           trend_tf: str = '1h',
                           trigger_tf: str = '15m') -> dict:
    """
    Fetches both macro trend (300 bars, 1h) and trigger (100 bars, 15m) candles
    from Binance for live signal analysis.
    """
    df_trend   = fetch_candles(exchange, symbol, timeframe=trend_tf, limit=300)
    df_trigger = fetch_candles(exchange, symbol, timeframe=trigger_tf, limit=100)
    return {'trend': df_trend, 'trigger': df_trigger}


def fetch_order_book_imbalance(exchange, symbol: str, depth: int = 15) -> dict:
    """
    Fetches live Binance order book depth to evaluate Bid/Ask volume imbalance.
    - Ratio > 1.2: Buyers dominate (strong support wall beneath price).
    - Ratio < 0.6: Heavy sell wall overhead (distribution / resistance).
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=depth)
        bids = ob.get('bids', [])[:depth]
        asks = ob.get('asks', [])[:depth]

        bids_vol = sum(b[1] for b in bids) if bids else 0.0
        asks_vol = sum(a[1] for a in asks) if asks else 0.0

        ratio = bids_vol / (asks_vol + 1e-8)
        
        if ratio > 1.2:
            pressure = "BULLISH_SUPPORT"
        elif ratio < 0.6:
            pressure = "HEAVY_SELL_WALL"
        else:
            pressure = "BALANCED"

        return {
            "status": "SUCCESS",
            "bid_vol": round(bids_vol, 2),
            "ask_vol": round(asks_vol, 2),
            "ratio": round(ratio, 2),
            "pressure": pressure
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e),
            "ratio": 1.0,
            "pressure": "BALANCED"
        }


if __name__ == "__main__":
    exchange = get_binance_exchange()
    print("Testing Binance Spot Vision:")
    for coin in coins:
        bars = fetch_candles(exchange, coin, timeframe='15m', limit=3)
        ob = fetch_order_book_imbalance(exchange, coin, depth=10)
        if not bars.empty:
            print(f"✅ {coin}: Live Price ${bars['close'].iloc[-1]:,.2f} | Order Book: {ob['pressure']} (B/A Ratio: {ob['ratio']})")
        else:
            print(f"❌ {coin}: Failed")
