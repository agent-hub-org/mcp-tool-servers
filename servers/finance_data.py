"""MCP server for finance data tools (yfinance + BSE/NSE reports). Port 8011."""

import logging
import time

import requests
import yfinance as yf
from cachetools import cached, TTLCache
from dotenv import load_dotenv
from fastmcp import FastMCP
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# ── Macro data cache (6-hour TTL) to avoid hammering public APIs ──
_macro_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 6 * 3600

logger = logging.getLogger("mcp_tool_servers.finance_data")

mcp = FastMCP("finance-data", instructions="Financial market data and reports tools.")

# ── Indian ticker alias table ── keys are lowercase normalized inputs, values are bare NSE symbols ──
_INDIAN_TICKER_ALIASES: dict[str, str] = {
    # PSU Banks
    "sbi": "SBIN", "state bank": "SBIN", "state bank of india": "SBIN", "sbin": "SBIN",
    "pnb": "PNB", "punjab national bank": "PNB",
    "bob": "BANKBARODA", "bank of baroda": "BANKBARODA", "bankbaroda": "BANKBARODA",
    "boi": "BANKINDIA", "bank of india": "BANKINDIA", "bankindia": "BANKINDIA",
    "ubi": "UNIONBANK", "union bank": "UNIONBANK", "union bank of india": "UNIONBANK", "unionbank": "UNIONBANK",
    "canara": "CANBK", "canara bank": "CANBK", "canbk": "CANBK",
    "uco": "UCOBANK", "uco bank": "UCOBANK", "ucobank": "UCOBANK",
    "iob": "IOB", "indian overseas bank": "IOB",
    "central bank": "CENTRALBK", "centralbk": "CENTRALBK",
    "indian bank": "INDIANB", "indianb": "INDIANB",
    "mahabank": "MAHABANK", "maharashtra bank": "MAHABANK",
    # Private Banks
    "hdfc bank": "HDFCBANK", "hdfcbank": "HDFCBANK", "hdfc": "HDFCBANK",
    "icici bank": "ICICIBANK", "icicibank": "ICICIBANK", "icici": "ICICIBANK",
    "axis bank": "AXISBANK", "axisbank": "AXISBANK", "axis": "AXISBANK",
    "kotak": "KOTAKBANK", "kotak bank": "KOTAKBANK", "kotakbank": "KOTAKBANK",
    "kotak mahindra bank": "KOTAKBANK",
    "indusind": "INDUSINDBK", "indusind bank": "INDUSINDBK", "indusindbk": "INDUSINDBK",
    "yes bank": "YESBANK", "yesbank": "YESBANK",
    "idfc first": "IDFCFIRSTB", "idfc first bank": "IDFCFIRSTB", "idfcfirstb": "IDFCFIRSTB",
    "federal bank": "FEDERALBNK", "federalbnk": "FEDERALBNK",
    "bandhan bank": "BANDHANBNK", "bandhanbnk": "BANDHANBNK",
    "rbl bank": "RBLBANK", "rblbank": "RBLBANK",
    "karnataka bank": "KTKBANK", "ktkbank": "KTKBANK",
    "south indian bank": "SOUTHBANK", "southbank": "SOUTHBANK",
    # IT / Technology
    "tcs": "TCS", "tata consultancy": "TCS", "tata consultancy services": "TCS",
    "infy": "INFY", "infosys": "INFY",
    "wipro": "WIPRO",
    "hcl": "HCLTECH", "hcl tech": "HCLTECH", "hcltech": "HCLTECH", "hcl technologies": "HCLTECH",
    "tech mahindra": "TECHM", "techm": "TECHM",
    "ltimindtree": "LTIM", "lti mindtree": "LTIM", "ltim": "LTIM",
    "mphasis": "MPHASIS",
    "persistent": "PERSISTENT", "persistent systems": "PERSISTENT",
    "coforge": "COFORGE",
    "hexaware": "HEXAWARE",
    # Large Caps / Nifty 50
    "reliance": "RELIANCE", "ril": "RELIANCE", "reliance industries": "RELIANCE",
    "itc": "ITC",
    "hul": "HINDUNILVR", "hindustan unilever": "HINDUNILVR", "hindunilvr": "HINDUNILVR",
    "bajaj finance": "BAJFINANCE", "bajfinance": "BAJFINANCE",
    "bajaj finserv": "BAJAJFINSV", "bajajfinsv": "BAJAJFINSV",
    "l&t": "LT", "lt": "LT", "larsen": "LT", "larsen and toubro": "LT",
    "maruti": "MARUTI", "maruti suzuki": "MARUTI",
    "tata motors": "TATAMOTORS", "tatamotors": "TATAMOTORS",
    "m&m": "M&M", "mahindra": "M&M", "mahindra and mahindra": "M&M",
    "hero motocorp": "HEROMOTOCO", "heromotoco": "HEROMOTOCO",
    "bajaj auto": "BAJAJ-AUTO", "bajaj-auto": "BAJAJ-AUTO",
    "tata steel": "TATASTEEL", "tatasteel": "TATASTEEL",
    "jsw steel": "JSWSTEEL", "jswsteel": "JSWSTEEL",
    "hindalco": "HINDALCO",
    "vedanta": "VEDL", "vedl": "VEDL",
    "ntpc": "NTPC",
    "powergrid": "POWERGRID", "power grid": "POWERGRID",
    "ongc": "ONGC",
    "coal india": "COALINDIA", "coalindia": "COALINDIA",
    "bpcl": "BPCL", "bharat petroleum": "BPCL",
    "hpcl": "HPCL", "hindustan petroleum": "HPCL",
    "ioc": "IOC", "iocl": "IOC", "indian oil": "IOC",
    "gail": "GAIL",
    "sun pharma": "SUNPHARMA", "sunpharma": "SUNPHARMA",
    "dr reddy": "DRREDDY", "drreddy": "DRREDDY", "dr reddys": "DRREDDY",
    "cipla": "CIPLA",
    "divis": "DIVISLAB", "divis lab": "DIVISLAB", "divislab": "DIVISLAB",
    "apollo hospitals": "APOLLOHOSP", "apollohosp": "APOLLOHOSP",
    "asian paints": "ASIANPAINT", "asianpaint": "ASIANPAINT",
    "nestle india": "NESTLEIND", "nestleind": "NESTLEIND",
    "britannia": "BRITANNIA",
    "adani ports": "ADANIPORTS", "adaniports": "ADANIPORTS",
    "adani green": "ADANIGREEN", "adanigreen": "ADANIGREEN",
    "adani ent": "ADANIENT", "adanient": "ADANIENT", "adani enterprises": "ADANIENT",
    "adani total gas": "ATGL", "atgl": "ATGL",
    "tata consumer": "TATACONSUM", "tataconsum": "TATACONSUM",
    "ultracemco": "ULTRACEMCO", "ultratech cement": "ULTRACEMCO",
    "shree cement": "SHREECEM", "shreecem": "SHREECEM",
    "acc": "ACC", "acc cement": "ACC",
    "ambuja cement": "AMBUJACEM", "ambujacem": "AMBUJACEM",
    "siemens": "SIEMENS",
    "abb india": "ABB", "abb": "ABB",
    "bhel": "BHEL",
    "irfc": "IRFC",
    "lic": "LICI", "lici": "LICI", "life insurance corporation": "LICI",
    "sbi life": "SBILIFE", "sbilife": "SBILIFE",
    "hdfc life": "HDFCLIFE", "hdfclife": "HDFCLIFE",
    "icici prudential": "ICICIPRULI", "icicipruli": "ICICIPRULI",
    "zomato": "ZOMATO",
    "paytm": "PAYTM",
    "nykaa": "FSN", "fsn": "FSN",
    "dmart": "DMART", "avenue supermarts": "DMART",
    "titan": "TITAN",
    "havells": "HAVELLS",
    "voltas": "VOLTAS",
    "pidilite": "PIDILITIND", "pidilitind": "PIDILITIND",
    "berger paints": "BERGEPAINT", "bergepaint": "BERGEPAINT",
    "godrej consumer": "GODREJCP", "godrejcp": "GODREJCP",
    "dabur": "DABUR",
    "colgate": "COLPAL", "colgate palmolive india": "COLPAL", "colpal": "COLPAL",
    "marico": "MARICO",
    "emami": "EMAMILTD", "emamiltd": "EMAMILTD",
    "page industries": "PAGEIND", "pageind": "PAGEIND",
    "info edge": "NAUKRI", "naukri": "NAUKRI",
    "indigo": "INDIGO", "interglobe aviation": "INDIGO",
    "irctc": "IRCTC",
    "mrf": "MRF",
    "tvs motor": "TVSMOTOR", "tvsmotor": "TVSMOTOR",
    "eicher motors": "EICHERMOT", "eichermot": "EICHERMOT",
    "srf": "SRF",
    "pi industries": "PIIND", "piind": "PIIND",
    "upl": "UPL",
    "tata power": "TATAPOWER", "tatapower": "TATAPOWER",
    "nhpc": "NHPC",
    "torrent power": "TORNTPOWER", "torntpower": "TORNTPOWER",
    "tata chemicals": "TATACHEM", "tatachem": "TATACHEM",
    "tata elxsi": "TATAELXSI", "tataelxsi": "TATAELXSI",
    "muthoot finance": "MUTHOOTFIN", "muthootfin": "MUTHOOTFIN",
    "cholamandalam": "CHOLAFIN", "cholafin": "CHOLAFIN",
    "l&t finance": "LTF", "ltf": "LTF",
    "shriram finance": "SHRIRAMFIN", "shriramfin": "SHRIRAMFIN",
    "hdfc amc": "HDFCAMC", "hdfcamc": "HDFCAMC",
    "nippon amc": "NAM-INDIA", "nam-india": "NAM-INDIA",
}


@mcp.tool()
def resolve_indian_ticker(company_name_or_symbol: str) -> str:
    """Resolve an Indian company name, abbreviation, or partial symbol to the correct
    NSE ticker (with .NS suffix) for use with all other finance tools.

    Call this BEFORE get_ticker_data, get_bse_nse_reports, or get_historical_ohlcv
    whenever the user provides a company name or abbreviation rather than an explicit
    NSE/BSE ticker like 'RELIANCE.NS'.

    Examples:
      "SBI"       → "SBIN.NS  (State Bank of India)"
      "UBI"       → "UNIONBANK.NS  (Union Bank of India)"
      "HDFC Bank" → "HDFCBANK.NS  (HDFC Bank Limited)"

    Returns a single line: "<NSE_TICKER>  (<confirmed company name>)"
    On failure returns an error string explaining what was tried."""
    logger.info("resolve_indian_ticker called with input='%s'", company_name_or_symbol)

    normalized = company_name_or_symbol.lower().strip()
    # Strip .ns/.bo so "SBIN.NS" normalizes to "sbin" — makes tool idempotent
    if normalized.endswith(".ns"):
        normalized = normalized[:-3]
    elif normalized.endswith(".bo"):
        normalized = normalized[:-3]

    if normalized in _INDIAN_TICKER_ALIASES:
        nse_symbol = _INDIAN_TICKER_ALIASES[normalized]
        ticker_with_suffix = f"{nse_symbol}.NS"
        logger.info("Alias table hit: '%s' → '%s'", company_name_or_symbol, ticker_with_suffix)
        try:
            name = yf.Ticker(ticker_with_suffix).fast_info.display_name or ticker_with_suffix
        except Exception:
            name = ticker_with_suffix
        return f"{ticker_with_suffix}  ({name})"

    # Fallback: yf.Search hits Yahoo Finance search API
    logger.info("Alias miss for '%s', falling back to yf.Search", company_name_or_symbol)
    try:
        results = yf.Search(
            company_name_or_symbol,
            max_results=10,
            news_count=0,
            lists_count=0,
        )
        indian_quotes = [
            q for q in results.quotes
            if q.get("symbol", "").endswith((".NS", ".BO"))
        ]
        if indian_quotes:
            best = indian_quotes[0]
            symbol = best["symbol"]
            name = best.get("shortname") or best.get("longname") or symbol
            logger.info("yf.Search resolved '%s' → '%s' (%s)", company_name_or_symbol, symbol, name)
            return f"{symbol}  ({name})"

        return (
            f"Could not resolve '{company_name_or_symbol}' to an Indian ticker. "
            "Try the explicit NSE symbol directly (e.g. 'SBIN.NS')."
        )
    except Exception as e:
        logger.error("yf.Search failed for '%s': %s", company_name_or_symbol, e)
        return (
            f"Ticker resolution failed for '{company_name_or_symbol}': {e}. "
            "Try the explicit NSE symbol directly."
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _yf_get_info(ticker: str) -> dict:
    """Inner fetch for yfinance info — retried on transient failures."""
    return yf.Ticker(ticker).info


@mcp.tool()
@cached(cache=TTLCache(maxsize=100, ttl=1800))
def get_ticker_data(ticker: str) -> str:
    """Get basic market data and company information for a given ticker symbol.
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'.
    Returns current price, market cap, P/E ratios, and a business summary."""
    logger.info("Fetching market data for ticker='%s'", ticker)
    try:
        info = _yf_get_info(ticker)
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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _yf_fetch_report(fetch_fn) -> object:
    """Inner fetch for a single yfinance report — retried on transient failures."""
    return fetch_fn()


@mcp.tool()
@cached(cache=TTLCache(maxsize=100, ttl=21600))
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

        missing: list[str] = []
        for title, fetch_fn in report_sources:
            try:
                df = _yf_fetch_report(fetch_fn)
                if not df.empty:
                    results.append(f"## {ticker} {title}\n\n{df.to_markdown()}")
                else:
                    missing.append(title)
            except Exception as e:
                logger.warning("Failed to fetch %s for %s: %s", title, ticker, e)
                missing.append(title)

        if not results:
            return f"No financial reports found for {ticker}"
        output = "\n\n---\n\n".join(results)
        if missing:
            output += f"\n\n**Note:** The following reports were unavailable: {', '.join(missing)}."
        return output
    except Exception as e:
        logger.error("Error fetching financial reports for ticker='%s': %s", ticker, e)
        return f"Failed to fetch reports for {ticker}: {e}"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _yf_get_history(ticker: str, period: str, interval: str, end_date: str | None = None):
    """Inner fetch for yfinance price history — retried on transient failures."""
    return yf.Ticker(ticker).history(period=period, interval=interval, end=end_date)


@mcp.tool()
@cached(cache=TTLCache(maxsize=100, ttl=3600))
def get_historical_ohlcv(ticker: str, period: str = "1y", interval: str = "1d", end_date: str | None = None) -> str:
    """Get price history summary and trend analysis for a ticker symbol.
    Returns multi-timeframe returns, monthly price series, moving averages, and volume.
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    interval options: 1d, 1wk, 1mo
    end_date: optional ISO date string (e.g. '2026-02-02') to cap history at a specific date
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'."""
    logger.info("Fetching OHLCV history for ticker='%s', period='%s', end_date=%s", ticker, period, end_date)
    try:
        hist = _yf_get_history(ticker, period, interval, end_date)
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
    failed: list[str] = []
    for symbol, label in indicators:
        try:
            info = yf.Ticker(symbol).fast_info
            price = info.last_price
            prev = info.previous_close
            if price is None:
                failed.append(label)
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
            failed.append(label)

    if fetched == 0:
        return "Failed to fetch macro indicators. Use tavily_quick_search for current data."

    if failed:
        lines.append(f"\n**Unavailable:** {', '.join(failed)} — use tavily_quick_search for these.")

    result = "\n".join(lines)
    _macro_cache[cache_key] = (now, result)
    return result


@mcp.tool()
def get_fii_dii_flows(days: int = 30) -> str:
    """Fetch FII/DII equity trading activity from NSE India for the last N trading days.
    Returns gross buy, gross sell, and net investment values in crores INR.
    FII (Foreign Institutional Investors) and DII (Domestic Institutional Investors) flows
    are a key indicator of institutional sentiment in Indian equity markets.

    Args:
        days: Number of trading days to fetch (default 30). Must be a positive integer.
    """
    if not isinstance(days, int) or days <= 0:
        return "Error: 'days' must be a positive integer."

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
