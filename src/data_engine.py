import ccxt
import pandas as pd
import time

coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']


def get_binance_exchange():
    """
    Returns a Binance CCXT client strictly locked to SPOT market endpoints.
    Points to Binance's official data mirror (data-api.binance.vision) and
    explicitly disables all futures (fapi/dapi) requests to prevent HTTP 451
    geo-restriction errors on GitHub Actions and US cloud servers.
    """
    return ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'fetchMarkets': ['spot'],  # Strictly block any fapi / futures calls
            'adjustForTimeDifference': True,
        },
        'urls': {
            'api': {
                'public': 'https://data-api.binance.vision/api/v3',
                'fapiPublic': 'https://data-api.binance.vision/api/v3',
                'fapiPrivate': 'https://data-api.binance.vision/api/v3',
            }
        }
    })


def get_fallback_exchange():
    """Global fallback exchange if Binance is unreachable from a cloud IP."""
    return ccxt.okx({'enableRateLimit': True})


def fetch_candles(exchange, symbol: str, timeframe: str = '1h', limit: int = 300) -> pd.DataFrame:
    """
    Fetches OHLCV candles from Binance, with automatic fallback if geo-blocked.
    """
    try:
        bars = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"⚠️ Binance {timeframe} error for {symbol}: {e}. Trying fallback exchange...")
        try:
            fb = get_fallback_exchange()
            bars = fb.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"✅ Fallback successful for {symbol} ({len(df)} candles)")
            return df
        except Exception as fb_err:
            print(f"❌ Fallback failed for {symbol}: {fb_err}")
            return pd.DataFrame()


def fetch_paginated_ohlcv(exchange, symbol: str, timeframe: str = '15m', days: int = 90) -> pd.DataFrame:
    """
    Fetches up to `days` of historical OHLCV data from Binance by paginating
    backward through 1000-bar limits.
    """
    all_bars = []
    limit = 1000

    since_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)

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


if __name__ == "__main__":
    exchange = get_binance_exchange()
    print("Testing Binance Spot Vision:")
    for coin in coins:
        bars = fetch_candles(exchange, coin, timeframe='15m', limit=3)
        if not bars.empty:
            print(f"  ✅ {coin}: ${bars['close'].iloc[-1]:,.2f}")
