"""MCP server for vector DB tools (Pinecone). Port 8012."""

import asyncio
import json
import logging
import os
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
_cohere_client = None


def _get_db(index_name: str) -> VectorDB:
    return VectorDB(index_name=index_name)


def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(
            api_key=os.environ["AZURE_COHERE_API_KEY"],
            base_url=os.environ["AZURE_COHERE_ENDPOINT"],
        )
    return _cohere_client


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
    of metadata fields, and index_name is the target Pinecone index.

    Args:
        data: The text content of the document.
        metadata_json: JSON string of metadata. Must be valid JSON.
        index_name: Target Pinecone index name.
    """
    logger.info("Upserting to vector DB — index='%s'", index_name)
    try:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            return "Error: metadata_json must be a JSON object."
        db = _get_db(index_name)
        doc_id = metadata.get("doc_id", "doc")
        num_chunks = db.upsert_chunks(doc_id, data, metadata)
        return f"Successfully upserted {num_chunks} chunks to index '{index_name}'"
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s", e)
        return f"Invalid JSON in metadata_json: {e}"
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
def hybrid_retrieve_papers(query: str, top_k: int = 10) -> str:
    """Hybrid retrieval: semantic top-k from vector DB → Cohere cross-encoder rerank → top-2.
    Returns chunks with relevance scores and a 'high_confidence' flag.
    If high_confidence is false, supplement with download_and_store_arxiv_papers and/or
    tavily_quick_search before synthesizing your answer."""
    logger.info("hybrid_retrieve_papers — query='%s', top_k=%d", query[:80], top_k)
    try:
        db = _get_db("research-papers")
        results = db.retrieve(query, top_k=top_k)
        if not results:
            return "high_confidence: false\nNo papers found in vector DB for this query."

        # Build document strings for Cohere reranker
        docs = [r.get("text", "") for r in results]
        co = _get_cohere()
        rerank_response = co.rerank(
            model="rerank-v4.0-fast",
            query=query,
            documents=docs,
            top_n=2,
            return_documents=True,
        )

        top_results = rerank_response.results
        if not top_results:
            return "high_confidence: false\nReranker returned no results."

        high_confidence = top_results[0].relevance_score >= 0.80
        confidence_line = f"high_confidence: {str(high_confidence).lower()}"
        logger.info(
            "hybrid_retrieve_papers: top score=%.3f, high_confidence=%s",
            top_results[0].relevance_score, high_confidence,
        )

        chunks = []
        for rr in top_results:
            original = results[rr.index]
            title = original.get("title", "Unknown")
            authors = original.get("authors", "")
            chunks.append(
                f"**{title}** by {authors} (rerank_score: {rr.relevance_score:.3f})\n"
                f"{rr.document.text}"
            )

        return confidence_line + "\n\n" + "\n\n---\n\n".join(chunks)
    except Exception as e:
        logger.error("hybrid_retrieve_papers error: %s", e)
        return f"high_confidence: false\nRetrieval failed: {e}"


def _rerank_candidates(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    """Rerank arxiv candidates by embedding similarity to the query.

    Each candidate dict must have 'title' and 'summary' keys.
    Returns the top_n most relevant candidates sorted by cosine similarity.
    """
    if len(candidates) <= top_n:
        return candidates

    db = _get_db("research-papers")

    # Build a text representation for each candidate
    candidate_texts = [
        f"{c['title']}. {c['summary']}" for c in candidates
    ]

    # Embed query and all candidates in one batch
    query_vector = db.embeddings.embed_query(query)
    candidate_vectors = db.embeddings.embed_documents(candidate_texts)

    # Compute cosine similarity (vectors are already normalized by most embedding models)
    similarities = []
    for i, cvec in enumerate(candidate_vectors):
        dot = sum(q * c for q, c in zip(query_vector, cvec))
        similarities.append((i, dot))

    # Sort by similarity descending, take top_n
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
async def download_and_store_arxiv_papers(query: str, max_results: int = 5,
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

        # Build the full query with optional category filtering
        full_query = query
        if categories:
            cat_filter = " OR ".join(f"cat:{c.strip()}" for c in categories.split(",") if c.strip())
            full_query = f"({query}) AND ({cat_filter})"

        # Fetch 3x candidates so we have a good pool to rerank from
        fetch_count = max_results * 3
        search = arxiv.Search(
            query=full_query,
            max_results=fetch_count,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        # Collect candidate metadata — keep arxiv.Result objects to avoid re-fetching later
        candidates = []
        paper_objects: dict[str, arxiv.Result] = {}
        for paper in _arxiv_client.results(search):
            short_id = paper.get_short_id()
            paper_objects[short_id] = paper
            candidates.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "summary": paper.summary,
                "pdf_url": paper.pdf_url,
                "short_id": short_id,
                "categories": list(paper.categories),
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else "",
            })

        if not candidates:
            return "No papers found on arXiv for this query."

        logger.info("Fetched %d candidates from arXiv, reranking to select top %d",
                     len(candidates), max_results)

        # Rerank by embedding similarity — only keep the most relevant papers
        top_candidates = _rerank_candidates(query, candidates, max_results)

        # Download PDFs in parallel — use cached paper objects, no per-paper re-fetch
        download_dir = Path("papers")
        download_dir.mkdir(exist_ok=True)

        async def _download_one(candidate: dict) -> dict | None:
            short_id = candidate["short_id"]
            file_name = f"{short_id}.pdf"
            file_path = download_dir / file_name
            paper_obj = paper_objects.get(short_id)
            if paper_obj is None:
                logger.warning("No cached object for '%s', skipping", short_id)
                return None
            logger.info("Downloading paper: '%s' (relevance: %.3f) → %s",
                        candidate["title"], candidate.get("_relevance_score", 0), file_path)
            await asyncio.to_thread(paper_obj.download_pdf, dirpath=str(download_dir), filename=file_name)
            return {
                "title": candidate["title"],
                "authors": candidate["authors"],
                "summary": candidate["summary"],
                "pdf_path": str(file_path),
                "pdf_url": candidate["pdf_url"],
                "categories": candidate["categories"],
                "published": candidate["published"],
            }

        download_results = await asyncio.gather(*[_download_one(c) for c in top_candidates])
        papers = [p for p in download_results if p is not None]

        if not papers:
            return "No papers could be downloaded."

        db = _get_db("research-papers")
        await db.upsert_papers(papers)

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


# ---- Semantic (user-knowledge) memory tools ----

_user_knowledge_db = None


def _get_user_knowledge_db():
    global _user_knowledge_db
    if _user_knowledge_db is None:
        from shared.vector_db import UserKnowledgeDB
        _user_knowledge_db = UserKnowledgeDB()
    return _user_knowledge_db


@mcp.tool()
def retrieve_user_knowledge(user_id: str, query: str, top_k: int = 5) -> str:
    """Retrieve semantic memory facts for a user from the user-knowledge index.
    Returns durable facts extracted from past sessions (preferences, risk profile, interests, etc.)."""
    logger.info("retrieve_user_knowledge — user_id='%s', query='%s'", user_id[:20], query[:60])
    try:
        db = _get_user_knowledge_db()
        facts = db.retrieve(user_id, query, top_k)
        if not facts:
            return "No semantic memories found for this user."
        return "\n".join(f"• {f['fact']} (score: {f['score']:.3f})" for f in facts)
    except Exception as e:
        logger.error("retrieve_user_knowledge error: %s", e)
        return f"Semantic memory retrieval failed: {e}"


@mcp.tool()
def upsert_user_knowledge(user_id: str, facts_json: str) -> str:
    """Upsert a list of durable semantic facts for a user.
    facts_json must be a JSON array of strings (e.g. '[\"risk-conservative investor\"]').
    Deduplication is applied — near-duplicate facts are skipped."""
    logger.info("upsert_user_knowledge — user_id='%s'", user_id[:20])
    try:
        facts = json.loads(facts_json)
        if not isinstance(facts, list):
            return "Error: facts_json must be a JSON array of strings."
        db = _get_user_knowledge_db()
        n = db.upsert_facts(user_id, [str(f) for f in facts if f])
        return f"Upserted {n} new facts (duplicates skipped)."
    except Exception as e:
        logger.error("upsert_user_knowledge error: %s", e)
        return f"Semantic memory upsert failed: {e}"


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["vector-db"])
