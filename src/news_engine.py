import feedparser
import re

FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/"
]

COIN_KEYWORDS = {
    'BTC/USDT': ['bitcoin', 'btc'],
    'ETH/USDT': ['ethereum', 'eth'],
    'SOL/USDT': ['solana', 'sol'],
    'BNB/USDT': ['binance', 'bnb'],
    'XRP/USDT': ['ripple', 'xrp']
}

# UPGRADE 4: Trade-killing red flags that VETO any signal
RED_FLAGS = [
    'hack', 'exploit', 'drain', 'sec', 'lawsuit', 'halt',
    'fraud', 'bankrupt', 'insolvent', 'freeze', 'indictment', 'scam'
]
BULLISH_WORDS = ['etf', 'surge', 'rally', 'record', 'inflow', 'approval', 'soar', 'breakout', 'partnership', 'bull']
BEARISH_WORDS = ['plunge', 'crash', 'dump', 'drop', 'outflow', 'liquidation', 'ban', 'bear', 'sell-off']


def fetch_headlines() -> list:
    """Aggregates latest headlines across all RSS feeds."""
    headlines = []
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
                headlines.append(entry.title)
        except Exception as e:
            print(f"⚠️ Feed error ({url}): {e}")
    return headlines


def analyze_news(symbol: str) -> dict:
    """
    UPGRADE 4: Breaking News Veto Filter.
    Scans headlines for red-flag events that should cancel all signals,
    and scores general sentiment for the coin.

    Returns:
        sentiment:    'BULLISH', 'BEARISH', or 'NEUTRAL'
        is_dangerous: True if a trade-killing red flag was found
        veto_signal:  Which trade type to veto ('LONG', 'SHORT', or None)
        top_headline: Most relevant recent headline
    """
    headlines = fetch_headlines()
    keywords = COIN_KEYWORDS.get(symbol, [symbol.split('/')[0].lower()])

    # 1. Filter articles mentioning this coin
    relevant = []
    for title in headlines:
        t_lower = title.lower()
        if any(re.search(r'\b' + kw + r'\b', t_lower) for kw in keywords):
            relevant.append(title)

    if not relevant:
        return {
            "sentiment": "NEUTRAL",
            "is_dangerous": False,
            "veto_signal": None,
            "relevant_count": 0,
            "top_headline": "No recent breaking news.",
            "red_flag_details": []
        }

    # 2. Check for Critical Red Flags (vetoes ALL long trades)
    is_dangerous = False
    flagged_reasons = []
    for title in relevant:
        lower = title.lower()
        for flag in RED_FLAGS:
            if re.search(r'\b' + flag + r'\b', lower):
                is_dangerous = True
                flagged_reasons.append(f"🚨 '{flag}' detected: \"{title}\"")

    # 3. Score general sentiment
    net_score = 0
    for title in relevant:
        lower = title.lower()
        net_score += sum(1 for w in BULLISH_WORDS if re.search(r'\b' + w, lower))
        net_score -= sum(1 for w in BEARISH_WORDS if re.search(r'\b' + w, lower))

    if net_score > 0:
        sentiment = "BULLISH"
    elif net_score < 0:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"

    # 4. Determine veto type
    # Dangerous news = veto ALL longs (never buy into hacks/lawsuits)
    # Bullish news = veto SHORTS (do not short into positive catalysts)
    veto_signal = None
    if is_dangerous:
        veto_signal = "LONG"      # Block buying during dangerous events
    elif sentiment == "BULLISH":
        veto_signal = "SHORT"     # Block shorting into strong positive news

    return {
        "sentiment": sentiment,
        "is_dangerous": is_dangerous,
        "veto_signal": veto_signal,
        "relevant_count": len(relevant),
        "top_headline": relevant[0],
        "red_flag_details": flagged_reasons
    }


if __name__ == "__main__":
    for pair in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        res = analyze_news(pair)
        print(f"\n📰 {pair} News Analysis:")
        print(f"  Sentiment    : {res['sentiment']}")
        print(f"  Dangerous?   : {'🚨 YES — Trade VETOED' if res['is_dangerous'] else '✅ NO'}")
        print(f"  Veto Signal  : {res['veto_signal'] or 'None'}")
        print(f"  Top Headline : {res['top_headline']}")
        if res['red_flag_details']:
            for f in res['red_flag_details']:
                print(f"  {f}")
