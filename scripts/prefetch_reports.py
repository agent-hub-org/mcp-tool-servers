"""
Pre-fetch quarterly and yearly financial reports for Nifty 50 and Sensex stocks.

Intended to run as a scheduled cron job (GitHub Actions or similar).
Fetches reports from yfinance and upserts them into the vector DB,
replacing stale data with fresh quarterly results.
"""
import logging
import os
import sys
import time
from datetime import datetime, timezone

import yfinance as yf

# Add project root to path so shared modules are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.vector_db import VectorDB

# Refresh threshold: only re-fetch if last fetch is older than this many days.
# Quarterly reports update every ~90 days; use 60 to catch end-of-quarter updates.
REFRESH_DAYS = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prefetch_reports")

# Nifty 50 constituents (NSE tickers)
NIFTY_50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ONGC.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", "TECHM.NS", "INDUSINDBK.NS",
    "HINDALCO.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BPCL.NS",
    "EICHERMOT.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "TATACONSUM.NS",
    "LTIM.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS",
]

# Sensex 30 (BSE tickers) — a subset; many overlap with Nifty 50
SENSEX_30_TICKERS = [
    "RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO",
    "HINDUNILVR.BO", "ITC.BO", "SBIN.BO", "BHARTIARTL.BO", "KOTAKBANK.BO",
    "LT.BO", "AXISBANK.BO", "ASIANPAINT.BO", "MARUTI.BO", "HCLTECH.BO",
    "SUNPHARMA.BO", "TITAN.BO", "BAJFINANCE.BO", "WIPRO.BO", "ULTRACEMCO.BO",
    "NESTLEIND.BO", "NTPC.BO", "POWERGRID.BO", "M&M.BO", "TATAMOTORS.BO",
    "INDUSINDBK.BO", "TECHM.BO", "BAJAJ-AUTO.BO", "JSWSTEEL.BO", "TATASTEEL.BO",
]


def _is_stale(last_fetched: str | None) -> bool:
    """Return True if the last fetch was more than REFRESH_DAYS ago, or never happened."""
    if last_fetched is None:
        return True
    try:
        fetched_date = datetime.strptime(last_fetched, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - fetched_date).days
        return age_days >= REFRESH_DAYS
    except ValueError:
        return True


def fetch_and_store_reports(ticker: str, db: VectorDB, force: bool = False) -> int:
    """Fetch financial reports for a single ticker and store in vector DB.

    Skips the fetch if existing data is fresher than REFRESH_DAYS days.
    Returns the number of reports stored (0 if skipped).
    """
    if not force:
        last_fetched = db.get_last_fetched(ticker)
        if not _is_stale(last_fetched):
            logger.info("Skipping %s — data is fresh (fetched_at=%s, threshold=%d days)",
                        ticker, last_fetched, REFRESH_DAYS)
            return 0
        if last_fetched:
            logger.info("Refreshing %s — data is stale (fetched_at=%s)", ticker, last_fetched)
        else:
            logger.info("First fetch for %s", ticker)

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
            if df is not None and not df.empty:
                reports.append({
                    "title": f"{ticker} {title_suffix}",
                    "content": f"{title_suffix} for {ticker}:\n\n{df.to_markdown()}",
                    "period": period,
                    "type": report_type,
                })
        except Exception as e:
            logger.warning("Failed to fetch %s for %s: %s", title_suffix, ticker, e)

    if not reports:
        logger.warning("No reports found for %s", ticker)
        return 0

    db.upsert_reports(ticker, reports)
    logger.info("Stored %d reports for %s", len(reports), ticker)
    return len(reports)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-fetch financial reports into vector DB")
    parser.add_argument("--force", action="store_true", help="Force refresh even if reports exist")
    parser.add_argument("--nifty-only", action="store_true", help="Only fetch Nifty 50 (skip Sensex BSE)")
    args = parser.parse_args()

    # Combine tickers (Nifty 50 NSE + Sensex 30 BSE)
    tickers = list(NIFTY_50_TICKERS)
    if not args.nifty_only:
        tickers.extend(SENSEX_30_TICKERS)

    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    logger.info("Starting pre-fetch for %d tickers (force=%s)", len(unique_tickers), args.force)

    db = VectorDB(index_name="financial-reports")
    total_reports = 0
    success_count = 0
    fail_count = 0

    for i, ticker in enumerate(unique_tickers, 1):
        logger.info("[%d/%d] Processing %s...", i, len(unique_tickers), ticker)
        try:
            count = fetch_and_store_reports(ticker, db, force=args.force)
            total_reports += count
            if count > 0:
                success_count += 1
            # Rate limit: yfinance can throttle if hit too fast
            time.sleep(1)
        except Exception as e:
            logger.error("Failed to process %s: %s", ticker, e)
            fail_count += 1
            time.sleep(2)

    logger.info(
        "Pre-fetch complete — %d tickers processed, %d successful, %d failed, %d total reports stored",
        len(unique_tickers), success_count, fail_count, total_reports,
    )


if __name__ == "__main__":
    main()
