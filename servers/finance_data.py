"""MCP server for finance data tools (yfinance + BSE/NSE reports). Port 8011."""

import logging
import time

import requests
import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# ── Macro data cache (6-hour TTL) to avoid hammering public APIs ──
_macro_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 6 * 3600

logger = logging.getLogger("mcp_tool_servers.finance_data")

mcp = FastMCP("finance-data", instructions="Financial market data and reports tools.")


@mcp.tool()
def get_ticker_data(ticker: str) -> str:
    """Get basic market data and company information for a given ticker symbol.
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'.
    Returns current price, market cap, P/E ratios, and a business summary."""
    logger.info("Fetching market data for ticker='%s'", ticker)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        relevant_keys = [
            "shortName", "symbol", "currentPrice", "marketCap", "sector", "industry",
            "trailingPE", "forwardPE", "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
            "longBusinessSummary",
        ]
        data = {k: info.get(k) for k in relevant_keys}
        lines = [f"**{k}:** {v}" for k, v in data.items() if v is not None]
        return "\n".join(lines) if lines else f"No data found for {ticker}"
    except Exception as e:
        logger.error("Error fetching data for ticker='%s': %s", ticker, e)
        return f"Failed to fetch data for {ticker}: {e}"


@mcp.tool()
def get_bse_nse_reports(ticker: str) -> str:
    """Fetch raw quarterly and yearly financial reports (Income Statement, Balance Sheet, Cash Flow)
    for a given ticker (use .NS or .BO suffix for Indian stocks).
    Returns markdown-formatted tabular data."""
    logger.info("Fetching financial reports for ticker='%s'", ticker)
    results = []
    try:
        t = yf.Ticker(ticker)

        report_sources = [
            ("Yearly Income Statement", lambda: t.income_stmt),
            ("Quarterly Income Statement", lambda: t.quarterly_income_stmt),
            ("Yearly Balance Sheet", lambda: t.balance_sheet),
            ("Quarterly Balance Sheet", lambda: t.quarterly_balance_sheet),
            ("Yearly Cash Flow", lambda: t.cashflow),
            ("Quarterly Cash Flow", lambda: t.quarterly_cashflow),
        ]

        for title, fetch_fn in report_sources:
            try:
                df = fetch_fn()
                if not df.empty:
                    results.append(f"## {ticker} {title}\n\n{df.to_markdown()}")
            except Exception as e:
                logger.warning("Failed to fetch %s for %s: %s", title, ticker, e)

        return "\n\n---\n\n".join(results) if results else f"No financial reports found for {ticker}"
    except Exception as e:
        logger.error("Error fetching financial reports for ticker='%s': %s", ticker, e)
        return f"Failed to fetch reports for {ticker}: {e}"


@mcp.tool()
def get_historical_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> str:
    """Get price history summary and trend analysis for a ticker symbol.
    Returns multi-timeframe returns, monthly price series, moving averages, and volume.
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    interval options: 1d, 1wk, 1mo
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'."""
    logger.info("Fetching OHLCV history for ticker='%s', period='%s'", ticker, period)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            return f"No price history found for {ticker}."

        latest_close = float(hist["Close"].iloc[-1])
        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())

        # Multi-timeframe returns
        lookbacks = {"5d": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}
        return_parts = []
        for label, n in lookbacks.items():
            if len(hist) >= n:
                past = float(hist["Close"].iloc[-n])
                pct = ((latest_close - past) / past) * 100
                return_parts.append(f"{label}: {pct:+.1f}%")

        # Monthly summary — last 12 months
        monthly = hist["Close"].resample("ME").last().tail(12)
        monthly_str = "  ".join(
            f"{d.strftime('%b %y')}: {v:.1f}" for d, v in monthly.items()
        )

        # Moving averages
        sma_50 = float(hist["Close"].tail(50).mean())
        sma_200_str = ""
        if len(hist) >= 200:
            sma_200 = float(hist["Close"].tail(200).mean())
            sma_200_str = f" | SMA 200: {sma_200:.2f}"

        avg_vol_10d = int(hist["Volume"].tail(10).mean())
        avg_vol_30d = int(hist["Volume"].tail(30).mean())

        lines = [
            f"## {ticker} Price Analysis ({period})",
            f"Latest Close: {latest_close:.2f}",
            f"52W High: {high_52w:.2f} | 52W Low: {low_52w:.2f}",
            f"Returns: {' | '.join(return_parts)}",
            f"SMA 50: {sma_50:.2f}{sma_200_str}",
            f"Avg Volume: 10d = {avg_vol_10d:,} | 30d = {avg_vol_30d:,}",
            f"Monthly Closes: {monthly_str}",
        ]
        return "\n".join(lines)
    except Exception as e:
        logger.error("Error fetching OHLCV for ticker='%s': %s", ticker, e)
        return f"Failed to fetch price history for {ticker}: {e}"


@mcp.tool()
def get_macro_indicators() -> str:
    """Fetch key Indian and global macro market indicators.
    Returns USD/INR, Brent Crude, Gold, US 10Y Treasury yield, Nifty 50, Sensex,
    India VIX, and US Dollar Index — all with day-over-day change.
    Results are cached for 6 hours.
    Note: For RBI repo rate and CPI/IIP data use tavily_quick_search('RBI repo rate India 2026')."""
    cache_key = "macro_indicators"
    now = time.time()
    if cache_key in _macro_cache:
        ts, cached = _macro_cache[cache_key]
        if now - ts < _CACHE_TTL:
            logger.info("Returning cached macro indicators")
            return cached

    logger.info("Fetching macro indicators from yfinance")
    indicators = [
        ("USDINR=X", "USD/INR"),
        ("BZ=F", "Brent Crude (USD/bbl)"),
        ("GC=F", "Gold (USD/oz)"),
        ("^TNX", "US 10Y Yield (%)"),
        ("^NSEI", "Nifty 50"),
        ("^BSESN", "BSE Sensex"),
        ("^INDIAVIX", "India VIX"),
        ("DX-Y.NYB", "US Dollar Index"),
    ]

    lines = ["## Indian & Global Macro Indicators",
             "*(For RBI repo rate / CPI / IIP use tavily_quick_search)*", ""]
    fetched = 0
    for symbol, label in indicators:
        try:
            info = yf.Ticker(symbol).fast_info
            price = info.last_price
            prev = info.previous_close
            if price is None:
                continue
            if prev and prev != 0:
                chg = ((price - prev) / prev) * 100
                sign = "+" if chg >= 0 else ""
                lines.append(f"**{label}:** {price:.2f} ({sign}{chg:.2f}%)")
            else:
                lines.append(f"**{label}:** {price:.2f}")
            fetched += 1
        except Exception as e:
            logger.warning("Skipping %s (%s): %s", symbol, label, e)

    if fetched == 0:
        return "Failed to fetch macro indicators. Use tavily_quick_search for current data."

    result = "\n".join(lines)
    _macro_cache[cache_key] = (now, result)
    return result


@mcp.tool()
def get_fii_dii_flows(days: int = 30) -> str:
    """Fetch FII/DII equity trading activity from NSE India for the last N trading days.
    Returns gross buy, gross sell, and net investment values in crores INR.
    FII (Foreign Institutional Investors) and DII (Domestic Institutional Investors) flows
    are a key indicator of institutional sentiment in Indian equity markets."""
    logger.info("Fetching FII/DII flows for last %d days", days)
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        })
        # Initialize NSE session to get cookies
        session.get("https://www.nseindia.com", timeout=15)

        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            return "No FII/DII data returned from NSE."

        recent = data[:days]
        lines = [f"## FII/DII Equity Flows — Last {len(recent)} Trading Days (NSE)", ""]
        for entry in recent:
            category = entry.get("category", "?")
            date = entry.get("date", "")
            buy_val = entry.get("buyValue", "0")
            sell_val = entry.get("sellValue", "0")
            net_val = entry.get("netValue", "0")
            net_float = float(str(net_val).replace(",", ""))
            sentiment = "NET BUYER" if net_float >= 0 else "NET SELLER"
            lines.append(
                f"**{date}** | {category} | Buy: ₹{buy_val}Cr | Sell: ₹{sell_val}Cr "
                f"| Net: ₹{net_val}Cr ({sentiment})"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error("NSE FII/DII fetch failed: %s", e)
        return (
            f"NSE FII/DII data unavailable: {e}. "
            "Fallback: use tavily_quick_search('FII DII net investment India equity today')."
        )


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["finance-data"])
