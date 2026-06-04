"""Standalone check that the market-retrieval data layer finds real markets.

Hits the LIVE Polymarket public-search and Manifold search endpoints (no API
keys required) and prints the matched markets with implied probabilities. Use it
to confirm the market-retrieval fixes without running the full bot/pipeline.

Usage:
    python scripts/check_market_retrieval.py "Makerfield by-election"
"""
import io
import json
import sys
import urllib.parse
import urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

_HEADERS = {"User-Agent": "Mozilla/5.0 (market-retrieval-check)"}


def _get_json(url, params):
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{url}?{query}", headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def _first_price(market):
    prices = market.get("outcomePrices")
    try:
        if isinstance(prices, str):
            return json.loads(prices)[0]
        if isinstance(prices, (list, tuple)) and prices:
            return prices[0]
    except Exception:
        return prices
    return None


def check_polymarket(query):
    print(f"\n=== Polymarket public-search: {query!r} ===")
    try:
        data = _get_json(
            "https://gamma-api.polymarket.com/public-search",
            {"q": query, "limit_per_type": 10, "events_status": "active"},
        )
    except Exception as error:
        print(f"  request failed: {error!r}")
        return
    events = data.get("events", []) if isinstance(data, dict) else []
    if not events:
        print("  no events found")
        return
    for event in events[:5]:
        print(f"  EVENT: {event.get('title')}  [{event.get('slug')}]")
        for market in event.get("markets", []) or []:
            name = market.get("groupItemTitle") or market.get("question") or "?"
            price = _first_price(market)
            if price not in (None, ""):
                print(f"      {name}: {price}")


def check_manifold(query):
    print(f"\n=== Manifold search: {query!r} ===")
    try:
        data = _get_json(
            "https://api.manifold.markets/v0/search-markets",
            {"term": query, "limit": 10},
        )
    except Exception as error:
        print(f"  request failed: {error!r}")
        return
    if not isinstance(data, list) or not data:
        print("  no markets found")
        return
    for market in data:
        prob = market.get("probability")
        prob_text = f"{prob:.2f}" if isinstance(prob, (int, float)) else "(multi)"
        print(f"  - {market.get('question')}  [p={prob_text}]")


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "Makerfield by-election"
    check_polymarket(query)
    check_manifold(query)


if __name__ == "__main__":
    main()
