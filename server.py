"""Combined MCP server — all tools on a single port."""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import arxiv
import base64
import re
import requests
import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP
from firecrawl import FirecrawlApp
from tavily import TavilyClient

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.vector_db import VectorDB

logger = logging.getLogger("mcp_tool_servers")

mcp = FastMCP("mcp-tool-servers", instructions="Web search, finance data, and vector DB tools.")


def _clamp(value: int, min_val: int, max_val: int, name: str) -> int:
    """Clamp *value* to [min_val, max_val]. Logs a warning when clamping occurs."""
    clamped = max(min_val, min(value, max_val))
    if clamped != value:
        logger.warning("Parameter '%s' clamped from %d to %d", name, value, clamped)
    return clamped


_VALID_YFINANCE_PERIODS = frozenset({
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
})
_VALID_YFINANCE_INTERVALS = frozenset({
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo",
})

# ── Macro data cache (6-hour TTL) ──
_macro_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 6 * 3600


# ── Lazy clients ──────────────────────────────────────────────

_tavily_client = None
_firecrawl_app = None
_arxiv_client = arxiv.Client()
_db_instances = {}


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily_client


def _get_firecrawl() -> FirecrawlApp:
    global _firecrawl_app
    if _firecrawl_app is None:
        _firecrawl_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    return _firecrawl_app


def _get_db(index_name: str) -> VectorDB:
    if index_name not in _db_instances:
        _db_instances[index_name] = VectorDB(index_name=index_name)
    return _db_instances[index_name]


# ── Web Search Tools ──────────────────────────────────────────

@mcp.tool()
def tavily_quick_search(query: str, max_results: int = 3) -> str:
    """Perform a quick web search across the internet. Returns synthesized answers and snippets.
    Ideal for news, quick fact-checking, and broad questions."""
    max_results = _clamp(max_results, 1, 10, "max_results")
    logger.info("Tavily search — query='%s', max_results=%d", query, max_results)
    try:
        client = _get_tavily()
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )
        results = []
        if response.get("answer"):
            results.append(f"**AI Answer:** {response['answer']}")
        for r in response.get("results", []):
            results.append(f"**{r['title']}** ({r['url']})\n{r['content']}")
        return "\n\n---\n\n".join(results) if results else "No results found."
    except Exception as e:
        logger.error("Tavily search error: %s", e)
        return f"Search failed: {e}"


@mcp.tool()
def firecrawl_deep_scrape(url: str) -> str:
    """Deep scrape a specific URL to extract its full markdown content.
    Use when you need to read a long-form article, report, or earnings transcript.

    Args:
        url: The full HTTP(S) URL to scrape. Must be a valid web address.
    """
    if not url.startswith(("http://", "https://")):
        return "Error: Invalid URL. Please provide a URL starting with http:// or https://"
    logger.info("Firecrawl scraping — url='%s'", url)
    try:
        app = _get_firecrawl()
        scrape_result = app.scrape(url, formats=["markdown"])
        markdown = scrape_result.get("markdown", "")
        metadata = scrape_result.get("metadata", {})
        title = metadata.get("title", "")
        return f"# {title}\n\n{markdown}" if title else markdown
    except Exception as e:
        logger.error("Firecrawl scrape error: %s", e)
        return f"Scrape failed for {url}: {e}"


# ── Finance Data Tools ────────────────────────────────────────

# ── Indian ticker alias table ──────────────────────────────────
_INDIAN_TICKER_ALIASES: dict[str, str] = {
    "sbi": "SBIN", "state bank": "SBIN", "state bank of india": "SBIN", "sbin": "SBIN",
    "pnb": "PNB", "punjab national bank": "PNB",
    "bob": "BANKBARODA", "bank of baroda": "BANKBARODA", "bankbaroda": "BANKBARODA",
    "boi": "BANKINDIA", "bank of india": "BANKINDIA", "bankindia": "BANKINDIA",
    "ubi": "UNIONBANK", "union bank": "UNIONBANK", "union bank of india": "UNIONBANK", "unionbank": "UNIONBANK",
    "canara": "CANBK", "canara bank": "CANBK", "canbk": "CANBK",
    "hdfc bank": "HDFCBANK", "hdfcbank": "HDFCBANK", "hdfc": "HDFCBANK",
    "icici bank": "ICICIBANK", "icicibank": "ICICIBANK", "icici": "ICICIBANK",
    "axis bank": "AXISBANK", "axisbank": "AXISBANK", "axis": "AXISBANK",
    "kotak": "KOTAKBANK", "kotak bank": "KOTAKBANK", "kotakbank": "KOTAKBANK",
    "reliance": "RELIANCE", "ril": "RELIANCE", "reliance industries": "RELIANCE",
    "itc": "ITC",
    "tcs": "TCS", "tata consultancy": "TCS", "tata consultancy services": "TCS",
    "infy": "INFY", "infosys": "INFY",
    "wipro": "WIPRO",
    "hcl": "HCLTECH", "hcl tech": "HCLTECH", "hcltech": "HCLTECH", "hcl technologies": "HCLTECH",
    "bajaj finance": "BAJFINANCE", "bajfinance": "BAJFINANCE",
    "l&t": "LT", "lt": "LT", "larsen": "LT", "larsen and toubro": "LT",
    "maruti": "MARUTI", "maruti suzuki": "MARUTI",
    "tata motors": "TATAMOTORS", "tatamotors": "TATAMOTORS",
    "sun pharma": "SUNPHARMA", "sunpharma": "SUNPHARMA",
    "asian paints": "ASIANPAINT", "asianpaint": "ASIANPAINT",
}


@mcp.tool()
def resolve_indian_ticker(company_name_or_symbol: str) -> str:
    """Resolve an Indian company name, abbreviation, or partial symbol to the correct
    NSE ticker (with .NS suffix) for use with all other finance tools.

    Call this BEFORE get_ticker_data, get_bse_nse_reports, or get_historical_ohlcv
    whenever the user provides a company name or abbreviation rather than an explicit
    NSE/BSE ticker like 'RELIANCE.NS'.

    Returns a single line: "<NSE_TICKER>  (<confirmed company name>)"
    On failure returns an error string explaining what was tried."""
    logger.info("resolve_indian_ticker called with input='%s'", company_name_or_symbol)

    normalized = company_name_or_symbol.lower().strip()
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
def get_historical_ohlcv(ticker: str, period: str = "1y", interval: str = "1d", end_date: str | None = None) -> str:
    """Get price history summary and trend analysis for a ticker symbol.
    Returns multi-timeframe returns, monthly price series, moving averages, and volume.
    period options: 1mo, 3mo, 6mo, 1y, 2y, 5y
    interval options: 1d, 1wk, 1mo
    end_date: optional ISO date string (e.g. '2026-02-02') to cap history at a specific date
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g., 'RELIANCE.NS'."""
    if period not in _VALID_YFINANCE_PERIODS:
        return f"Invalid period '{period}'. Valid options: {sorted(_VALID_YFINANCE_PERIODS)}"
    if interval not in _VALID_YFINANCE_INTERVALS:
        return f"Invalid interval '{interval}'. Valid options: {sorted(_VALID_YFINANCE_INTERVALS)}"
    logger.info("Fetching OHLCV history for ticker='%s', period='%s', end_date=%s", ticker, period, end_date)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, end=end_date)
        if hist.empty:
            return f"No price history found for {ticker}."

        latest_close = float(hist["Close"].iloc[-1])
        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())

        lookbacks = {"5d": 5, "1m": 21, "3m": 63, "6m": 126, "1y": 252}
        return_parts = []
        for label, n in lookbacks.items():
            if len(hist) >= n:
                past = float(hist["Close"].iloc[-n])
                pct = ((latest_close - past) / past) * 100
                return_parts.append(f"{label}: {pct:+.1f}%")

        monthly = hist["Close"].resample("ME").last().tail(12)
        monthly_str = "  ".join(
            f"{d.strftime('%b %y')}: {v:.1f}" for d, v in monthly.items()
        )

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


# ── Financial Education Calculators ──────────────────────────

@mcp.tool()
def calculate_sip_returns(
    monthly_investment: float,
    annual_return_rate: float,
    years: int,
    inflation_rate: float = 6.0,
) -> str:
    """Calculate SIP maturity value and inflation-adjusted real value.
    monthly_investment: INR per month (e.g. 5000)
    annual_return_rate: expected annual return % (e.g. 12)
    years: investment duration (e.g. 10)
    inflation_rate: annual inflation % for real-value calc (default 6)"""
    if monthly_investment <= 0 or annual_return_rate <= 0 or years <= 0:
        return "All inputs must be positive numbers."
    n = years * 12
    r = (annual_return_rate / 100) / 12
    maturity = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)
    invested = monthly_investment * n
    real_value = maturity / ((1 + inflation_rate / 100) ** years)
    return (
        f"## SIP Calculator Result\n\n"
        f"**Monthly SIP:** ₹{monthly_investment:,.0f} | "
        f"**Duration:** {years} years | **Expected return:** {annual_return_rate}%\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Total Invested | ₹{invested:,.0f} |\n"
        f"| Estimated Maturity Value | ₹{maturity:,.0f} |\n"
        f"| Wealth Gained | ₹{maturity - invested:,.0f} |\n"
        f"| Real Value (after {inflation_rate}% inflation) | ₹{real_value:,.0f} |\n\n"
        f"*Estimates assume constant returns. Actual mutual fund returns vary.*"
    )


@mcp.tool()
def calculate_goal_sip(
    target_amount: float,
    annual_return_rate: float,
    years: int,
) -> str:
    """Reverse SIP calculator: monthly investment needed to reach a target amount.
    target_amount: financial goal in INR (e.g. 1000000 for ₹10 lakh)
    annual_return_rate: expected annual return % (e.g. 12)
    years: time horizon in years (e.g. 10)"""
    if target_amount <= 0 or annual_return_rate <= 0 or years <= 0:
        return "All inputs must be positive numbers."
    n = years * 12
    r = (annual_return_rate / 100) / 12
    fv_factor = (((1 + r) ** n - 1) / r) * (1 + r)
    monthly = target_amount / fv_factor
    return (
        f"## Goal-Based SIP Calculator\n\n"
        f"**Target:** ₹{target_amount:,.0f} in {years} years at {annual_return_rate}% returns\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Required Monthly SIP | ₹{monthly:,.0f} |\n"
        f"| Total You'll Invest | ₹{monthly * n:,.0f} |\n"
        f"| Market Will Add | ₹{target_amount - monthly * n:,.0f} |\n\n"
        f"*Start a SIP with any AMFI-registered mutual fund to begin.*"
    )


@mcp.tool()
def calculate_inflation_impact(
    current_value: float,
    inflation_rate: float,
    years: int,
) -> str:
    """Show how inflation erodes purchasing power over time.
    current_value: amount in INR today (e.g. 1000000 for ₹10 lakh)
    inflation_rate: annual inflation % (e.g. 6)
    years: number of years (e.g. 10)"""
    if current_value <= 0 or inflation_rate <= 0 or years <= 0:
        return "All inputs must be positive numbers."
    future_cost = current_value * ((1 + inflation_rate / 100) ** years)
    loss = future_cost - current_value
    return (
        f"## Inflation Impact Calculator\n\n"
        f"**Today's Value:** ₹{current_value:,.0f} | "
        f"**Inflation:** {inflation_rate}%/yr | **Years:** {years}\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Future Cost of Same Thing | ₹{future_cost:,.0f} |\n"
        f"| Purchasing Power Eroded | ₹{loss:,.0f} ({loss / future_cost * 100:.1f}%) |\n\n"
        f"This is why savings accounts (3-4% interest) lose to inflation. "
        f"Investments must beat {inflation_rate}% to grow your real wealth.\n\n"
        f"*India's CPI inflation historically averages 5-7%/year.*"
    )


# ── Vector DB Tools ───────────────────────────────────────────

@mcp.tool()
def check_in_vector_db(identifier: str, index_name: str) -> str:
    """Check if documents with a given identifier exist in the vector DB.
    For financial reports, identifier is a ticker (e.g., 'RELIANCE.NS') and index_name is 'financial-reports'.
    For research papers, use a query string and index_name 'research-papers'."""
    logger.info("Checking vector DB — identifier='%s', index='%s'", identifier, index_name)
    try:
        db = _get_db(index_name)
        if index_name == "research-papers":
            exists = db.papers_exist(identifier)
        else:
            exists = db.reports_exist(identifier)
        status = "found" if exists else "not found"
        return f"Documents for '{identifier}' in index '{index_name}': {status}"
    except Exception as e:
        logger.error("Error checking vector DB: %s", e)
        return f"Error checking vector DB: {e}"


@mcp.tool()
def upsert_to_vector_db(data: str, metadata_json: str, index_name: str) -> str:
    """Upsert a document to the vector DB. data is the text content, metadata_json is a JSON string
    of metadata fields, and index_name is the target Pinecone index."""
    logger.info("Upserting to vector DB — index='%s'", index_name)
    try:
        metadata = json.loads(metadata_json)
        db = _get_db(index_name)
        doc_id = metadata.get("doc_id", "doc")
        num_chunks = db.upsert_chunks(doc_id, data, metadata)
        return f"Successfully upserted {num_chunks} chunks to index '{index_name}'"
    except Exception as e:
        logger.error("Error upserting to vector DB: %s", e)
        return f"Upsert failed: {e}"


@mcp.tool()
def retrieve_from_vector_db(query: str, index_name: str, filter_key: str = "",
                            filter_value: str = "", top_k: int = 5) -> str:
    """Retrieve relevant document chunks from the vector DB using semantic search.
    Args:
        query: Search query
        index_name: Pinecone index name (e.g., 'financial-reports' or 'research-papers')
        filter_key: Optional metadata key to filter by (e.g., 'ticker')
        filter_value: Optional metadata value to filter by (e.g., 'RELIANCE.NS')
        top_k: Number of results to return"""
    top_k = _clamp(top_k, 1, 50, "top_k")
    logger.info("Retrieving from vector DB — query='%s', index='%s'", query[:80], index_name)
    try:
        db = _get_db(index_name)
        results = db.retrieve(query, top_k=top_k, filter_key=filter_key, filter_value=filter_value)
        if not results:
            return "No relevant results found."
        chunks = []
        for r in results:
            score = r.pop("score", 0)
            text = r.pop("text", "")
            title = r.get("title", "Unknown")
            chunks.append(f"**{title}** (score: {score:.3f})\n{text}")
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        logger.error("Error retrieving from vector DB: %s", e)
        return f"Retrieval failed: {e}"


@mcp.tool()
def add_financial_reports_to_db(ticker: str) -> str:
    """Fetch quarterly and yearly financial reports for a ticker and store them in the vector DB.
    Always check if reports exist first with check_in_vector_db."""
    logger.info("Adding financial reports for ticker='%s'", ticker)
    try:
        db = _get_db("financial-reports")
        if db.reports_exist(ticker):
            return f"Reports for {ticker} already exist in the database."

        t = yf.Ticker(ticker)
        reports = []

        report_sources = [
            ("Yearly Income Statement", "Yearly", "Income Statement", lambda: t.income_stmt),
            ("Quarterly Income Statement", "Quarterly", "Income Statement", lambda: t.quarterly_income_stmt),
            ("Yearly Balance Sheet", "Yearly", "Balance Sheet", lambda: t.balance_sheet),
            ("Quarterly Balance Sheet", "Quarterly", "Balance Sheet", lambda: t.quarterly_balance_sheet),
            ("Yearly Cash Flow", "Yearly", "Cash Flow", lambda: t.cashflow),
            ("Quarterly Cash Flow", "Quarterly", "Cash Flow", lambda: t.quarterly_cashflow),
        ]

        for title_suffix, period, report_type, fetch_fn in report_sources:
            try:
                df = fetch_fn()
                if not df.empty:
                    reports.append({
                        "title": f"{ticker} {title_suffix}",
                        "content": f"{title_suffix} for {ticker}:\n\n{df.to_markdown()}",
                        "period": period,
                        "type": report_type,
                    })
            except Exception as e:
                logger.warning("Failed to fetch %s for %s: %s", title_suffix, ticker, e)

        if not reports:
            return f"No financial reports found for {ticker}."

        db.upsert_reports(ticker, reports)
        return f"Successfully loaded {len(reports)} reports into the vector DB for {ticker}."
    except Exception as e:
        logger.error("Error adding reports: %s", e)
        return f"Failed to add reports: {e}"


@mcp.tool()
def check_papers_in_db(query: str) -> str:
    """Check if relevant research papers exist in the vector DB for a given query."""
    logger.info("Checking papers in DB — query='%s'", query[:80])
    try:
        db = _get_db("research-papers")
        exists = db.papers_exist(query)
        if exists:
            return f"Relevant papers found in the database for query: '{query}'"
        return f"No relevant papers found in the database for query: '{query}'"
    except Exception as e:
        logger.error("Error checking papers: %s", e)
        return f"Error checking papers: {e}"


@mcp.tool()
def retrieve_papers(query: str, top_k: int = 5) -> str:
    """Retrieve relevant research paper chunks from the vector DB using semantic search."""
    logger.info("Retrieving papers — query='%s', top_k=%d", query[:80], top_k)
    try:
        db = _get_db("research-papers")
        results = db.retrieve(query, top_k=top_k)
        if not results:
            return "No relevant papers found."
        chunks = []
        for r in results:
            score = r.pop("score", 0)
            text = r.pop("text", "")
            title = r.get("title", "Unknown")
            authors = r.get("authors", "")
            chunks.append(f"**{title}** by {authors} (score: {score:.3f})\n{text}")
        return "\n\n---\n\n".join(chunks)
    except Exception as e:
        logger.error("Error retrieving papers: %s", e)
        return f"Retrieval failed: {e}"


def _rerank_candidates(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Rerank arxiv candidates by embedding similarity to the query."""
    if len(candidates) <= top_n:
        return candidates

    db = _get_db("research-papers")
    candidate_texts = [f"{c['title']}. {c['summary']}" for c in candidates]
    query_vector = db.embeddings.embed_query(query)
    candidate_vectors = db.embeddings.embed_documents(candidate_texts)

    similarities = []
    for i, cvec in enumerate(candidate_vectors):
        dot = sum(q * c for q, c in zip(query_vector, cvec))
        similarities.append((i, dot))

    similarities.sort(key=lambda x: x[1], reverse=True)
    reranked = []
    for idx, score in similarities[:top_n]:
        candidates[idx]["_relevance_score"] = round(score, 4)
        reranked.append(candidates[idx])

    logger.info("Reranked %d candidates → top %d (scores: %.3f – %.3f)",
                len(candidates), len(reranked),
                reranked[0].get("_relevance_score", 0),
                reranked[-1].get("_relevance_score", 0))
    return reranked


@mcp.tool()
def download_and_store_arxiv_papers(query: str, max_results: int = 5,
                                     sort_by: str = "relevance",
                                     categories: str = "") -> str:
    """Search arXiv, download PDFs, convert to markdown, and store in the vector DB.
    Fetches a larger pool of candidates from arXiv, then reranks them by semantic
    similarity to the query and only downloads the most relevant ones.

    Args:
        query: Search query — use precise academic terms for best results.
        max_results: Max papers to download (default 5). Keep small for targeted searches.
        sort_by: Sort order — 'relevance' (default), 'submitted' (newest first), or 'updated'.
        categories: Optional comma-separated arXiv categories to filter by
                    (e.g., 'cs.AI,cs.CL,cs.LG'). Leave empty for all categories.
    """
    logger.info("Downloading arXiv papers — query='%s', max_results=%d, sort='%s', categories='%s'",
                query, max_results, sort_by, categories)
    try:
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }
        sort_criterion = sort_map.get(sort_by, arxiv.SortCriterion.Relevance)

        full_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{c.strip()}" for c in categories.split(",") if c.strip())
            full_query = f"({query}) AND ({cat_filter})"

        fetch_count = max_results * 3
        search = arxiv.Search(
            query=full_query,
            max_results=fetch_count,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        candidates = []
        for paper in _arxiv_client.results(search):
            candidates.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "short_id": paper.get_short_id(),
                "categories": list(paper.categories),
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            })

        if not candidates:
            return "No papers found on arXiv for this query."

        logger.info("Fetched %d candidates from arXiv, reranking to select top %d",
                     len(candidates), max_results)

        top_candidates = _rerank_candidates(query, candidates, max_results)

        download_dir = Path("papers")
        download_dir.mkdir(exist_ok=True)

        papers = []
        for candidate in top_candidates:
            file_name = f"{candidate['short_id']}.pdf"
            file_path = download_dir / file_name

            logger.info("Downloading paper: '%s' (relevance: %.3f) → %s",
                        candidate["title"],
                        candidate.get("_relevance_score", 0),
                        file_path)

            paper_search = arxiv.Search(id_list=[candidate["short_id"]])
            paper_obj = next(_arxiv_client.results(paper_search), None)
            if paper_obj is None:
                logger.warning("Could not re-fetch paper '%s', skipping", candidate["short_id"])
                continue
            paper_obj.download_pdf(dirpath=str(download_dir), filename=file_name)

            papers.append({
                "title": candidate["title"],
                "authors": candidate["authors"],
                "summary": candidate["summary"],
                "pdf_path": str(file_path),
                "pdf_url": candidate["pdf_url"],
                "categories": candidate["categories"],
                "published": candidate["published"],
            })

        if not papers:
            return "No papers could be downloaded."

        db = _get_db("research-papers")
        db.upsert_papers(papers)

        summaries = []
        for i, p in enumerate(papers):
            authors = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors += " et al."
            date_str = f" ({p['published']})" if p.get("published") else ""
            score_str = f" [relevance: {top_candidates[i].get('_relevance_score', 'N/A')}]"
            summaries.append(f"- **{p['title']}** by {authors}{date_str}{score_str}")

        return f"Downloaded and stored {len(papers)} papers (reranked from {len(candidates)} candidates):\n" + "\n".join(summaries)
    except Exception as e:
        logger.error("Error downloading arXiv papers: %s", e)
        return f"Failed to download papers: {e}"


# ── GitHub Tools ──────────────────────────────────────────────

_GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
_GH_HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "mcp-tool-servers/1.0",
}
if _GITHUB_TOKEN:
    _GH_HEADERS["Authorization"] = f"Bearer {_GITHUB_TOKEN}"

_GH_SKIP = re.compile(
    r"(node_modules|\.git|__pycache__|\.venv|venv|dist|build|coverage"
    r"|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.idea|\.vscode"
    r"|migrations|fixtures|vendor|\.next|\.nuxt)/|"
    r"\.(pyc|pyo|so|dylib|dll|exe|bin|lock|svg|png|jpg|jpeg|gif|ico"
    r"|woff2?|ttf|otf|eot|map|min\.js|min\.css)$",
    re.IGNORECASE,
)
_GH_PRIORITY = [
    (10, re.compile(r"^README", re.IGNORECASE)),
    (9,  re.compile(r"^(main|app|index|server|core)\.(py|ts|js|go|java|rs|rb)$")),
    (8,  re.compile(r"^(pyproject\.toml|package\.json|go\.mod|Cargo\.toml|pom\.xml|build\.gradle)$")),
    (7,  re.compile(r"\.(py|ts|go|java|rs|rb|jsx|tsx)$")),
    (6,  re.compile(r"\.(js|c|cpp|h|cs|swift|kt)$")),
    (5,  re.compile(r"\.(ya?ml|toml|json|env\.example)$")),
    (4,  re.compile(r"\.(md|txt)$")),
]
_GH_MAX_BYTES = 60_000
_GH_TIMEOUT = 20


def _gh_file_priority(path: str) -> int:
    name = os.path.basename(path)
    for score, pattern in _GH_PRIORITY:
        if pattern.search(name):
            return score
    return 3


def _gh_validate_url(repo_url: str) -> tuple[str, str]:
    """Validate a GitHub URL and return (owner, repo). Raises ValueError on failure."""
    url = repo_url.strip().rstrip("/")
    normalized = re.sub(r"^https?://(www\.)?", "", url, flags=re.IGNORECASE)
    if not normalized.lower().startswith("github.com/"):
        raise ValueError(f"Only public github.com repositories are supported. Got: {url!r}")
    path_part = re.split(r"[?#]", normalized[len("github.com/"):])[0]
    parts = [p for p in path_part.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse owner/repo from URL {url!r}")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    _valid = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")
    if not _valid.match(owner) or not _valid.match(repo):
        raise ValueError(f"Invalid owner/repo characters in URL: {url!r}")
    return owner, repo


@mcp.tool()
def fetch_github_repo(repo_url: str, max_files: int = 40) -> str:
    """Fetch key source files from a public GitHub repository for code analysis.

    Args:
        repo_url: Public GitHub URL (https://github.com/owner/repo). Only public repos.
        max_files: Files to fetch content for (1–60, default 40).

    Returns JSON with: repo_url, repo_name, owner, language, description,
    file_tree, key_files [{path, content}], summary, total_files.
    Returns "Error: ..." on failure.
    """
    try:
        owner, repo = _gh_validate_url(repo_url)
    except ValueError as exc:
        return f"Error: {exc}"

    repo_full = f"{owner}/{repo}"
    max_files = max(1, min(int(max_files), 60))

    try:
        meta_res = requests.get(
            f"https://api.github.com/repos/{repo_full}",
            headers=_GH_HEADERS, timeout=_GH_TIMEOUT,
        )
        if meta_res.status_code == 404:
            return f"Error: Repository not found or is private: {repo_full}"
        if meta_res.status_code == 403:
            return f"Error: Access denied for {repo_full}. Only public repositories are supported."
        if not meta_res.ok:
            return f"Error: GitHub API returned {meta_res.status_code} for {repo_full}"
        meta = meta_res.json()

        default_branch = meta.get("default_branch", "main")
        tree_res = requests.get(
            f"https://api.github.com/repos/{repo_full}/git/trees/{default_branch}?recursive=1",
            headers=_GH_HEADERS, timeout=_GH_TIMEOUT,
        )
        if not tree_res.ok:
            return f"Error: Failed to fetch file tree (status {tree_res.status_code})"

        all_blobs = [
            item for item in tree_res.json().get("tree", [])
            if item["type"] == "blob" and not _GH_SKIP.search(item["path"])
        ]
        all_blobs.sort(key=lambda f: (-_gh_file_priority(f["path"]), len(f["path"])))

        def _fetch_file(item: dict) -> dict | None:
            """Fetch a single file from GitHub. Returns {path, content} or None on failure."""
            try:
                cr = requests.get(
                    f"https://api.github.com/repos/{repo_full}/contents/{item['path']}",
                    headers=_GH_HEADERS, timeout=_GH_TIMEOUT,
                )
                if not cr.ok:
                    return None
                fj = cr.json()
                if fj.get("encoding") != "base64":
                    return None
                raw = base64.b64decode(fj["content"]).decode("utf-8", errors="replace")
                if len(raw) > _GH_MAX_BYTES:
                    raw = raw[:_GH_MAX_BYTES] + f"\n\n... [truncated at {_GH_MAX_BYTES} bytes]"
                return {"path": item["path"], "content": raw}
            except Exception as exc:
                logger.warning("Skipping %s: %s", item["path"], exc)
                return None

        targets = all_blobs[:max_files]
        key_files: list[dict[str, str]] = []
        # Fetch files concurrently — up to 10 parallel GitHub API calls
        with ThreadPoolExecutor(max_workers=min(10, len(targets))) as executor:
            futures = {executor.submit(_fetch_file, item): item for item in targets}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    key_files.append(result)
        # Restore priority order (parallel execution loses original sort order)
        path_order = {item["path"]: idx for idx, item in enumerate(targets)}
        key_files.sort(key=lambda f: path_order.get(f["path"], 999))

        all_paths = [item["path"] for item in all_blobs[:200]]
        language = meta.get("language") or "unknown"
        description = meta.get("description") or ""

        summary_parts = []
        if description:
            summary_parts.append(f"Description: {description}")
        summary_parts.extend([
            f"Primary language: {language}",
            f"Total visible files: {len(all_blobs)}",
            f"Files fetched: {len(key_files)}",
        ])
        if all_paths:
            summary_parts.append("\nFile tree (up to 80 paths):")
            summary_parts.extend(f"  {p}" for p in all_paths[:80])

        return json.dumps({
            "repo_url": repo_url,
            "repo_name": repo,
            "owner": owner,
            "language": language,
            "description": description,
            "file_tree": all_paths,
            "key_files": key_files,
            "summary": "\n".join(summary_parts),
            "total_files": len(all_blobs),
        })

    except requests.exceptions.Timeout:
        return f"Error: Request timed out fetching {repo_full}."
    except requests.exceptions.ConnectionError as exc:
        return f"Error: Connection failed for {repo_full}: {exc}"
    except Exception as exc:
        logger.error("Unexpected error fetching repo %s: %s", repo_full, exc)
        return f"Error: Unexpected failure fetching {repo_full}: {exc}"


# Expose the ASGI app so uvicorn can import it directly (enables --workers).
# Each worker handles independent tool-call HTTP requests so no shared session
# state is needed — multi-worker is safe for the streamable-http transport.
app = mcp.http_app()

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8010))
    workers = int(os.getenv("WEB_CONCURRENCY", 1))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
    )
