import ccxt
import pandas as pd
import time

coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT']


def fetch_candles(exchange, symbol: str, timeframe: str = '1h', limit: int = 300) -> pd.DataFrame:
    """Fetches a single batch of OHLCV candles."""
    try:
        bars = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"❌ Error fetching {timeframe} candles for {symbol}: {e}")
        return pd.DataFrame()


def fetch_paginated_ohlcv(exchange, symbol: str, timeframe: str = '15m', days: int = 90) -> pd.DataFrame:
    """
    Fetches up to `days` of historical OHLCV data by paginating
    backward through Binance's 1000-bar limit per request.
    Respects rate limits with a 0.5s sleep between requests.

    Example:
        df = fetch_paginated_ohlcv(exchange, 'BTC/USDT', timeframe='15m', days=90)
        # Returns ~8640 candles = 90 days of 15m data
    """
    all_bars = []
    limit = 1000

    # Calculate start timestamp
    since_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)
    since_ms = int(since_dt.timestamp() * 1000)

    print(f"  📅 Fetching {days} days of {timeframe} data for {symbol} "
          f"(from {since_dt.date()})...")

    while True:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe,
                                        since=since_ms, limit=limit)
        except Exception as e:
            print(f"  ❌ Pagination error: {e}")
            break

        if not bars:
            break

        all_bars += bars
        last_ts = bars[-1][0]
        since_ms = last_ts + 1  # Start next batch from next bar

        # Stop when we have reached now
        now_ms = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000)
        if last_ts >= now_ms or len(bars) < limit:
            break

        time.sleep(0.3)  # Respect rate limits

    if not all_bars:
        print(f"  ⚠️ No data returned for {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('timestamp').reset_index(drop=True)

    print(f"  ✅ {len(df)} bars fetched: "
          f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    return df


def fetch_multi_timeframe(exchange, symbol: str,
                           trend_tf: str = '1h',
                           trigger_tf: str = '15m') -> dict:
    """
    Fetches both macro trend (300 bars, 1h) and trigger (100 bars, 15m) candles
    for live signal analysis.
    """
    df_trend   = fetch_candles(exchange, symbol, timeframe=trend_tf, limit=300)
    df_trigger = fetch_candles(exchange, symbol, timeframe=trigger_tf, limit=100)
    return {'trend': df_trend, 'trigger': df_trigger}


if __name__ == "__main__":
    exchange = ccxt.binance({'enableRateLimit': True})

    # Demo: fetch paginated 15m data for BTC
    df = fetch_paginated_ohlcv(exchange, 'BTC/USDT', timeframe='15m', days=90)
    print(f"\nTotal candles fetched: {len(df)}")
    print(df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].tail(3))
