"""MCP server for vector DB tools (Pinecone). Port 8012."""

import json
import logging
from pathlib import Path

import arxiv
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.vector_db import VectorDB

logger = logging.getLogger("mcp_tool_servers.vector_db_server")

mcp = FastMCP("vector-db", instructions="Vector database tools for storing and retrieving documents.")

_arxiv_client = arxiv.Client()


def _get_db(index_name: str) -> VectorDB:
    return VectorDB(index_name=index_name)


# ---- Generic vector DB operations ----

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


# ---- Financial report pipeline ----

@mcp.tool()
def add_financial_reports_to_db(ticker: str) -> str:
    """Fetch quarterly and yearly financial reports for a ticker and store them in the vector DB.
    Always check if reports exist first with check_in_vector_db."""
    logger.info("Adding financial reports for ticker='%s'", ticker)
    try:
        import yfinance as yf

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


# ---- Research paper tools ----

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


@mcp.tool()
def download_and_store_arxiv_papers(query: str, max_results: int = 5,
                                     sort_by: str = "relevance",
                                     categories: str = "") -> str:
    """Search arXiv, download PDFs, convert to markdown, and store in the vector DB.
    Returns a summary of downloaded papers.

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

        # Build the full query with optional category filtering
        full_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{c.strip()}" for c in categories.split(",") if c.strip())
            full_query = f"({query}) AND ({cat_filter})"

        # Fetch more candidates than needed, then take top max_results
        # This helps when some results are low quality
        fetch_count = max_results * 2
        search = arxiv.Search(
            query=full_query,
            max_results=fetch_count,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        download_dir = Path("papers")
        download_dir.mkdir(exist_ok=True)

        papers = []
        for paper in _arxiv_client.results(search):
            if len(papers) >= max_results:
                break

            file_name = f"{paper.get_short_id()}.pdf"
            file_path = download_dir / file_name

            logger.info("Downloading paper: '%s' → %s", paper.title, file_path)
            paper.download_pdf(dirpath=str(download_dir), filename=file_name)

            papers.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "pdf_path": str(file_path),
                "pdf_url": paper.pdf_url,
                "categories": list(paper.categories),
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            })

        if not papers:
            return "No papers found on arXiv for this query."

        db = _get_db("research-papers")
        db.upsert_papers(papers)

        summaries = []
        for p in papers:
            authors = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors += " et al."
            date_str = f" ({p['published']})" if p.get("published") else ""
            summaries.append(f"- **{p['title']}** by {authors}{date_str}")

        return f"Downloaded and stored {len(papers)} papers:\n" + "\n".join(summaries)
    except Exception as e:
        logger.error("Error downloading arXiv papers: %s", e)
        return f"Failed to download papers: {e}"


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["vector-db"])
