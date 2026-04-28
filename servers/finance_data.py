"""MCP server for finance data tools (yfinance + BSE/NSE reports). Port 8011."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import httpx
import yfinance as yf
from cachetools import cached, TTLCache
from dotenv import load_dotenv
from fastmcp import FastMCP
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.ticker_aliases import INDIAN_TICKER_ALIASES

logger = logging.getLogger("mcp_tool_servers.finance_data")

mcp = FastMCP("finance-data", instructions="Financial market data and reports tools.")

_MACRO_CACHE_TTL = 6 * 3600
_macro_cache: TTLCache = TTLCache(maxsize=16, ttl=_MACRO_CACHE_TTL)

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
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

    if normalized in INDIAN_TICKER_ALIASES:
        nse_symbol = INDIAN_TICKER_ALIASES[normalized]
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
@cached(cache=_macro_cache)
def get_macro_indicators() -> str:
    """Fetch key Indian and global macro market indicators.
    Returns USD/INR, Brent Crude, Gold, US 10Y Treasury yield, Nifty 50, Sensex,
    India VIX, and US Dollar Index — all with day-over-day change.
    Results are cached for 6 hours.
    Note: For RBI repo rate and CPI/IIP data use tavily_quick_search('RBI repo rate India 2026')."""
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

    return "\n".join(lines)


@mcp.tool()
async def get_fii_dii_flows(days: int = 30) -> str:
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
        async with httpx.AsyncClient(headers=_NSE_HEADERS, timeout=15, follow_redirects=True) as client:
            await client.get("https://www.nseindia.com")
            resp = await client.get("https://www.nseindia.com/api/fiidiiTradeReact")
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


@mcp.tool()
@cached(cache=TTLCache(maxsize=100, ttl=3600))
def get_dcf_inputs(ticker: str) -> str:
    """Extract structured DCF inputs directly from financial statements for a ticker.

    Returns a JSON object with all fields needed to call run_dcf immediately:
      current_fcf_cr     — most recent annual Free Cash Flow in INR crores
      growth_rate_pct    — 3-year FCF CAGR (%) capped at ±50%, use as initial growth_rate_pct
      shares_outstanding_cr — total shares in crores
      net_debt_cr        — totalDebt minus totalCash in crores (positive = net debt)
      current_price      — latest market price

    Use this BEFORE calling run_dcf so values come from actual financial data,
    not guessed or hardcoded. Adjust growth_rate_pct based on analyst consensus
    or sector outlook if you have better forward estimates.

    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g. 'JSWSTEEL.NS'.
    """
    logger.info("get_dcf_inputs called for ticker='%s'", ticker)
    try:
        t = yf.Ticker(ticker)
        info = _yf_get_info(ticker)

        # ── shares outstanding → crores ──────────────────────────────────────
        shares_raw = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        shares_outstanding_cr = round(shares_raw / 1e7, 2) if shares_raw else None

        # ── current price ────────────────────────────────────────────────────
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if current_price:
            current_price = round(float(current_price), 2)

        # ── FCF from yearly cash flow statement → crores ────────────────────
        fcf_values_cr: list[float] = []
        try:
            cf = _yf_fetch_report(lambda: t.cashflow)
            if not cf.empty:
                # Try "Free Cash Flow" row first
                fcf_row = None
                for label in cf.index:
                    if "free cash flow" in str(label).lower():
                        fcf_row = cf.loc[label]
                        break
                if fcf_row is None:
                    # Fallback: Operating Cash Flow − |CapEx|
                    ocf_row = next(
                        (cf.loc[l] for l in cf.index
                         if "operating" in str(l).lower() and "cash" in str(l).lower()),
                        None,
                    )
                    capex_row = next(
                        (cf.loc[l] for l in cf.index
                         if "capital" in str(l).lower()
                         and ("expenditure" in str(l).lower() or "capex" in str(l).lower())),
                        None,
                    )
                    if ocf_row is not None and capex_row is not None:
                        for o_val, c_val in zip(ocf_row.dropna().values[:4], capex_row.dropna().values[:4]):
                            try:
                                fcf_values_cr.append(round((float(o_val) - abs(float(c_val))) / 1e7, 2))
                            except (ValueError, TypeError):
                                pass
                else:
                    for v in fcf_row.dropna().values[:4]:
                        try:
                            fcf_values_cr.append(round(float(v) / 1e7, 2))
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logger.warning("Could not extract FCF for %s: %s", ticker, e)

        current_fcf_cr = fcf_values_cr[0] if fcf_values_cr else None

        # ── 3-year FCF CAGR ──────────────────────────────────────────────────
        growth_rate_pct = None
        if (len(fcf_values_cr) >= 3
                and fcf_values_cr[0] is not None
                and fcf_values_cr[2] is not None
                and fcf_values_cr[2] > 0
                and fcf_values_cr[0] > 0):
            try:
                cagr = ((fcf_values_cr[0] / fcf_values_cr[2]) ** (1.0 / 2) - 1.0) * 100
                growth_rate_pct = round(max(-50.0, min(50.0, cagr)), 1)
            except (ValueError, ZeroDivisionError):
                pass

        # ── net debt → crores ────────────────────────────────────────────────
        total_debt = float(info.get("totalDebt") or 0)
        total_cash = float(info.get("totalCash") or 0)
        net_debt_cr = round((total_debt - total_cash) / 1e7, 2)

        fcf_history_note = (
            "FCF last 4yr (cr): " + str(fcf_values_cr[:4])
            if fcf_values_cr else "FCF data unavailable"
        )

        result: dict = {
            "ticker": ticker,
            "current_fcf_cr": current_fcf_cr,
            "growth_rate_pct": growth_rate_pct,
            "shares_outstanding_cr": shares_outstanding_cr,
            "net_debt_cr": net_debt_cr,
            "current_price": current_price,
            "notes": (
                f"{fcf_history_note}. "
                "growth_rate_pct = 3yr FCF CAGR capped ±50% — adjust upward/downward based on "
                "forward guidance or sector outlook. "
                "net_debt_cr = totalDebt - totalCash. "
                "Pass all fields directly to run_dcf."
            ),
        }

        missing = [k for k, v in result.items() if v is None and k not in ("notes",)]
        if missing:
            result["warnings"] = (
                f"Could not extract: {', '.join(missing)}. "
                "You must supply these manually before calling run_dcf."
            )

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("get_dcf_inputs failed for '%s': %s", ticker, e)
        return json.dumps({"error": f"Failed to extract DCF inputs for {ticker}: {e}"})


@mcp.tool()
@cached(cache=TTLCache(maxsize=100, ttl=900))
def get_price_series(ticker: str, period: str = "1y") -> str:
    """Return a JSON list of daily closing prices for a ticker, most recent last.

    Pass the returned `closes` list directly as the `prices` argument to
    calculate_technical_signals and calculate_risk_metrics — do NOT try to
    extract prices from the markdown output of get_historical_ohlcv.

    period options: 3mo, 6mo, 1y, 2y (default: 1y)
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g. 'JSWSTEEL.NS'.
    """
    logger.info("get_price_series called for ticker='%s', period='%s'", ticker, period)
    try:
        hist = _yf_get_history(ticker, period=period, interval="1d")
        if hist.empty:
            return json.dumps({"error": f"No price data found for {ticker}"})
        closes = [round(float(v), 2) for v in hist["Close"].tolist()]
        return json.dumps({
            "ticker": ticker,
            "period": period,
            "count": len(closes),
            "closes": closes,
        })
    except Exception as e:
        logger.error("get_price_series failed for '%s': %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch price series for {ticker}: {e}"})


@mcp.tool()
def get_comparable_metrics(tickers: list[str]) -> str:
    """Fetch structured valuation and quality metrics for a list of tickers.

    Returns data formatted for direct use with run_comparable_valuation:
      target_ticker  — first ticker in the list
      target_metrics — PE, PB, EV/EBITDA, ROE, EBITDA margin, revenue growth
      peers          — same metrics for each remaining ticker

    Pass target_ticker, target_metrics, and peers directly to run_comparable_valuation.
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g. ['JSWSTEEL.NS', 'TATASTEEL.NS'].
    At least 2 tickers are recommended (1 target + 1 peer minimum).
    """
    if not tickers:
        return json.dumps({"error": "At least one ticker required"})

    logger.info("get_comparable_metrics called for tickers=%s", tickers)

    def _fetch_metrics(ticker: str) -> dict:
        try:
            info = _yf_get_info(ticker)
            roe = info.get("returnOnEquity")
            ebitda_m = info.get("ebitdaMargins")
            rev_g = info.get("revenueGrowth")
            rec: dict = {"ticker": ticker}
            if info.get("trailingPE") is not None:
                rec["pe"] = round(float(info["trailingPE"]), 2)
            if info.get("priceToBook") is not None:
                rec["pb"] = round(float(info["priceToBook"]), 2)
            if info.get("enterpriseToEbitda") is not None:
                rec["ev_ebitda"] = round(float(info["enterpriseToEbitda"]), 2)
            if roe is not None:
                rec["roe"] = round(float(roe) * 100, 2)          # fraction → %
            if ebitda_m is not None:
                rec["ebitda_margin"] = round(float(ebitda_m) * 100, 2)  # fraction → %
            if rev_g is not None:
                rec["revenue_growth"] = round(float(rev_g) * 100, 2)    # fraction → %
            return rec
        except Exception as e:
            logger.warning("Could not fetch metrics for %s: %s", ticker, e)
            return {"ticker": ticker, "error": str(e)}

    all_metrics = [_fetch_metrics(t) for t in tickers]
    target = all_metrics[0]
    peers = all_metrics[1:]
    target_metrics = {k: v for k, v in target.items() if k not in ("ticker", "error")}

    missing = [t["ticker"] for t in all_metrics if "error" in t]
    result: dict = {
        "target_ticker": target["ticker"],
        "target_metrics": target_metrics,
        "peers": peers,
    }
    if missing:
        result["warnings"] = f"Could not fetch metrics for: {', '.join(missing)}"

    return json.dumps(result, indent=2)


@mcp.tool()
async def get_regime_inputs() -> str:
    """Fetch and structure current Indian macro indicators as inputs for detect_market_regime.

    Returns a JSON object whose field names match RegimeDetectorInput exactly:
      india_vix, usd_inr, crude_brent, nifty_pe, fii_net_30d
    Fields that require an external search (repo_rate, cpi_yoy, credit_growth, gsec_10y)
    are listed in the `needs_search` array with suggested tavily queries.

    Pass the returned values directly to detect_market_regime — do NOT use
    the markdown output of get_macro_indicators for this purpose.
    """
    logger.info("get_regime_inputs called")
    result: dict = {}
    warnings: list[str] = []

    # ── Live market data from yfinance ──────────────────────────────────────
    _yfin_map = {
        "^INDIAVIX": "india_vix",
        "USDINR=X": "usd_inr",
        "BZ=F": "crude_brent",
    }
    for symbol, field in _yfin_map.items():
        try:
            val = yf.Ticker(symbol).fast_info.last_price
            if val is not None:
                result[field] = round(float(val), 2)
            else:
                warnings.append(f"{field} returned None from yfinance")
        except Exception as e:
            warnings.append(f"{field} unavailable: {e}")

    # ── Nifty 50 PE from yfinance info ──────────────────────────────────────
    try:
        nifty_info = _yf_get_info("^NSEI")
        pe = nifty_info.get("trailingPE")
        if pe:
            result["nifty_pe"] = round(float(pe), 1)
        else:
            warnings.append("nifty_pe not in yfinance ^NSEI info — use tavily_quick_search('Nifty 50 PE ratio today')")
    except Exception as e:
        warnings.append(f"nifty_pe unavailable: {e}")

    # ── FII net 30-day flows from NSE (async httpx) ──────────────────────────
    try:
        async with httpx.AsyncClient(headers=_NSE_HEADERS, timeout=15, follow_redirects=True) as client:
            await client.get("https://www.nseindia.com")
            resp = await client.get("https://www.nseindia.com/api/fiidiiTradeReact")
        resp.raise_for_status()
        data = resp.json()
        fii_net = sum(
            float(str(e.get("netValue", "0")).replace(",", ""))
            for e in data[:30]
            if "FII" in str(e.get("category", "")).upper()
        )
        result["fii_net_30d"] = round(fii_net, 2)
    except Exception as e:
        warnings.append(f"fii_net_30d unavailable from NSE: {e}")

    # ── Fields that need search ──────────────────────────────────────────────
    now = datetime.now()
    month_year = now.strftime("%B %Y")
    prev_month_year = now.replace(day=1).replace(month=now.month - 1 if now.month > 1 else 12).strftime("%B %Y")
    result["needs_search"] = {
        "repo_rate": f"tavily_quick_search('RBI repo rate India {month_year}')",
        "cpi_yoy": f"tavily_quick_search('India CPI inflation YoY {prev_month_year}')",
        "credit_growth": f"tavily_quick_search('India bank credit growth YoY {now.year}')",
        "gsec_10y": "tavily_quick_search('India 10 year G-sec yield today')",
    }
    if warnings:
        result["warnings"] = warnings

    return json.dumps(result, indent=2)


@mcp.tool()
@cached(cache=TTLCache(maxsize=200, ttl=3600))
def get_earnings_calendar(ticker: str) -> str:
    """Return the next earnings date and key financial calendar events for a ticker.

    Returns upcoming earnings date, dividend date, ex-dividend date, and analyst
    estimates (EPS and revenue) where available from yfinance.
    For Indian stocks use .NS (NSE) or .BO (BSE) suffix, e.g. 'HDFCBANK.NS'.
    """
    logger.info("get_earnings_calendar called for ticker='%s'", ticker)
    try:
        t = yf.Ticker(ticker)
        info = _yf_get_info(ticker)

        result: dict = {"ticker": ticker}

        # Earnings dates from calendar
        try:
            cal = t.calendar
            if cal is not None and not cal.empty:
                if "Earnings Date" in cal.index:
                    dates = cal.loc["Earnings Date"]
                    result["earnings_dates"] = [str(d)[:10] for d in (dates if hasattr(dates, "__iter__") else [dates])]
                if "Earnings Average" in cal.index:
                    result["eps_estimate"] = float(cal.loc["Earnings Average"]) if cal.loc["Earnings Average"] is not None else None
                if "Revenue Average" in cal.index:
                    rev = cal.loc["Revenue Average"]
                    result["revenue_estimate_cr"] = round(float(rev) / 1e7, 2) if rev else None
        except Exception:
            pass

        # Dividend info
        ex_div = info.get("exDividendDate")
        if ex_div:
            result["ex_dividend_date"] = datetime.fromtimestamp(ex_div).strftime("%Y-%m-%d")
        div_rate = info.get("dividendRate")
        if div_rate:
            result["annual_dividend"] = round(float(div_rate), 2)
        div_yield = info.get("dividendYield")
        if div_yield:
            result["dividend_yield_pct"] = round(float(div_yield) * 100, 2)

        # Last reported quarter
        fiscal_ye = info.get("fiscalYearEnd")
        if fiscal_ye:
            result["fiscal_year_end_month"] = fiscal_ye

        if len(result) == 1:
            result["note"] = "No earnings calendar data available. Try tavily_quick_search for upcoming results."

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error("get_earnings_calendar failed for '%s': %s", ticker, e)
        return json.dumps({"error": f"Failed to fetch earnings calendar for {ticker}: {e}"})


@mcp.tool()
def run_scenario_simulation(
    ticker: str,
    scenario: str,
    base_price: float,
    shock_pct: float,
    current_sip_monthly: float = 0.0,
    holding_period_years: float = 1.0,
) -> str:
    """Simulate the impact of a market scenario on a stock price and SIP portfolio.

    Args:
        ticker: The ticker symbol (e.g. 'NIFTYBEES.NS', 'RELIANCE.NS').
        scenario: Plain-English description of the scenario (e.g. 'Nifty falls 20% in a bear market').
        base_price: Current price of the ticker (use get_ticker_data to fetch it first).
        shock_pct: Price change in percent, e.g. -20.0 for a 20% fall, +15.0 for a 15% rally.
        current_sip_monthly: Optional monthly SIP amount in INR. 0 if not applicable.
        holding_period_years: How many years to project the scenario over.
    """
    import math

    logger.info("run_scenario_simulation: ticker='%s', shock=%.1f%%, sip=%.0f, years=%.1f",
                ticker, shock_pct, current_sip_monthly, holding_period_years)

    shock = shock_pct / 100.0
    shocked_price = round(base_price * (1 + shock), 2)
    price_delta = round(shocked_price - base_price, 2)

    result: dict = {
        "scenario": scenario,
        "ticker": ticker,
        "base_price": base_price,
        "shocked_price": shocked_price,
        "price_change": price_delta,
        "price_change_pct": shock_pct,
    }

    # Portfolio impact for a single-stock holding
    result["impact_per_unit"] = f"{'Loss' if shock < 0 else 'Gain'} of Rs.{abs(price_delta):.2f} per share"
    result["impact_on_1L_holding"] = f"Rs.{round(100000 * shock, 2):,.2f} ({'loss' if shock < 0 else 'gain'}) on a Rs.1,00,000 position"

    # SIP scenario — rupee cost averaging effect
    if current_sip_monthly > 0 and holding_period_years > 0:
        months = int(holding_period_years * 12)
        total_invested = current_sip_monthly * months

        # Linear price trajectory from base to shocked price over the period
        prices = [base_price + (shocked_price - base_price) * (i / max(months - 1, 1)) for i in range(months)]
        units_accumulated = sum(current_sip_monthly / p for p in prices)
        final_value = units_accumulated * shocked_price
        xirr_approx = (final_value / total_invested) ** (1 / holding_period_years) - 1

        result["sip_analysis"] = {
            "monthly_sip": current_sip_monthly,
            "holding_period_years": holding_period_years,
            "total_invested": round(total_invested, 2),
            "units_accumulated": round(units_accumulated, 4),
            "final_value": round(final_value, 2),
            "absolute_pnl": round(final_value - total_invested, 2),
            "approx_cagr_pct": round(xirr_approx * 100, 2),
            "rupee_cost_avg_price": round(total_invested / units_accumulated, 2),
            "note": (
                "Rupee-cost averaging softens the impact vs lump-sum. "
                "Early months buy more units at lower prices in a bear scenario."
                if shock < 0 else
                "In a bull scenario, early SIP months buy fewer units at lower prices — "
                "later months are more expensive."
            ),
        }

    result["key_takeaway"] = (
        f"In this '{scenario}' scenario, {ticker} drops to Rs.{shocked_price:,.2f} "
        f"({shock_pct:+.1f}%). "
        + (f"A monthly SIP of Rs.{current_sip_monthly:,.0f} over {holding_period_years:.0f}y "
           f"would accumulate {result['sip_analysis']['units_accumulated']:.2f} units "
           f"worth Rs.{result['sip_analysis']['final_value']:,.2f} "
           f"(approx. {result['sip_analysis']['approx_cagr_pct']:+.1f}% CAGR)."
           if current_sip_monthly > 0 else "")
    )

    return json.dumps(result, indent=2)


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["finance-data"])
