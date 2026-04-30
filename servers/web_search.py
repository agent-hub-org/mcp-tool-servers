"""MCP server for web search tools (Tavily + Firecrawl). Port 8010."""

import asyncio
import logging
import os

from dotenv import load_dotenv
from fastmcp import FastMCP
from firecrawl import FirecrawlApp
from tavily import TavilyClient

load_dotenv()

logger = logging.getLogger("mcp_tool_servers.web_search")

mcp = FastMCP("web-search", instructions="Web search and scraping tools.")

_tavily_client = None
_firecrawl_app = None
_azure_openai_client = None

# Simple in-process cache so identical queries skip the rewrite LLM call.
_rewrite_cache: dict[str, str] = {}

_VAGUE_WORDS = frozenset({
    "what", "how", "why", "when", "who", "tell me", "explain",
    "describe", "latest", "recent", "current", "new",
})


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


def _get_azure_openai():
    global _azure_openai_client
    if _azure_openai_client is None:
        from openai import AsyncOpenAI
        _azure_openai_client = AsyncOpenAI(
            base_url=os.environ["AZURE_AI_FOUNDRY_ENDPOINT"],
            api_key=os.environ["AZURE_AI_FOUNDRY_API_KEY"],
        )
    return _azure_openai_client


async def _rewrite_query_for_web(query: str) -> str:
    """Rewrite a vague/conversational query into a precise web-search string.
    Uses Groq llama-3.1-8b-instant (~100ms). Skipped for already-specific queries
    and when DISABLE_QUERY_REWRITE=true is set."""
    if os.getenv("DISABLE_QUERY_REWRITE", "").lower() == "true":
        return query

    if query in _rewrite_cache:
        return _rewrite_cache[query]

    # Skip rewrite for queries that are already specific (long, no vague words)
    lower = query.lower()
    has_vague = any(w in lower for w in _VAGUE_WORDS)
    if len(query) > 60 and not has_vague:
        return query

    try:
        client = _get_azure_openai()
        response = await client.chat.completions.create(
            model=os.getenv("AZURE_QUERY_REWRITE_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the search query for a web search engine. "
                        "Be specific. Add the current year (2025 or 2026) if the query is time-sensitive. "
                        "Output ONLY the rewritten query, nothing else."
                    ),
                },
                {"role": "user", "content": f"Query: {query}"},
            ],
            max_tokens=60,
            temperature=0,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        if rewritten:
            _rewrite_cache[query] = rewritten
            logger.info("Query rewritten: '%s' → '%s'", query[:80], rewritten[:80])
            return rewritten
    except Exception as e:
        logger.warning("Query rewrite failed (%s) — using original query", e)

    return query


@mcp.tool()
async def tavily_quick_search(query: str, max_results: int = 3) -> str:
    """Perform a quick web search across the internet. Returns synthesized answers and snippets.
    Ideal for news, quick fact-checking, and broad questions."""
    logger.info("Tavily search — query='%s', max_results=%d", query, max_results)
    try:
        rewritten_query = await _rewrite_query_for_web(query)
        client = _get_tavily()
        response = await asyncio.to_thread(
            client.search,
            rewritten_query,
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
        url: The full HTTPS URL to scrape. Must be a valid web address.
    """
    logger.info("Firecrawl scraping — url='%s'", url)
    if not url.startswith(("http://", "https://")):
        return "Error: Invalid URL. Please provide a URL starting with http:// or https://"

    try:
        app = _get_firecrawl()
        scrape_result = app.scrape(url, formats=["markdown"])
        markdown = scrape_result.markdown or ""
        title = (scrape_result.metadata.title if scrape_result.metadata else "") or ""
        return f"# {title}\n\n{markdown}" if title else markdown
    except Exception as e:
        logger.error("Firecrawl scrape error: %s", e)
        return f"Scrape failed for {url}: {e}"


@mcp.tool()
def search_pubmed(query: str, max_results: int = 5) -> str:
    """Search PubMed for peer-reviewed biomedical and health science literature.

    Returns article titles, authors, publication date, abstract, and PMID.
    Use for evidence-based health, nutrition, exercise science, and clinical questions.
    Prefer this over Tavily when you need peer-reviewed sources with PMIDs.

    Args:
        query: The search query (e.g. "progressive overload muscle hypertrophy",
               "intermittent fasting metabolic effects", "ACL rehabilitation protocol").
        max_results: Number of results to return (1-10, default 5).
    """
    import urllib.request
    import urllib.parse
    import json as _json

    max_results = max(1, min(max_results, 10))
    logger.info("PubMed search — query='%s', max=%d", query, max_results)

    try:
        # Step 1: esearch to get PMIDs
        search_params = urllib.parse.urlencode({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        })
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{search_params}"
        with urllib.request.urlopen(search_url, timeout=10) as resp:
            search_data = _json.loads(resp.read().decode())

        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return f"No PubMed articles found for: {query}"

        # Step 2: efetch to get abstracts
        fetch_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
            "rettype": "abstract",
        })
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{fetch_params}"

        # Use esummary for structured data
        summary_params = urllib.parse.urlencode({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json",
        })
        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{summary_params}"
        with urllib.request.urlopen(summary_url, timeout=10) as resp:
            summary_data = _json.loads(resp.read().decode())

        results = []
        uids = summary_data.get("result", {}).get("uids", [])
        for uid in uids:
            article = summary_data["result"].get(uid, {})
            title = article.get("title", "No title")
            authors = ", ".join(a.get("name", "") for a in article.get("authors", [])[:3])
            if len(article.get("authors", [])) > 3:
                authors += " et al."
            pub_date = article.get("pubdate", "")
            source = article.get("source", "")
            results.append(
                f"**{title}**\n"
                f"Authors: {authors}\n"
                f"Journal: {source} ({pub_date})\n"
                f"PMID: {uid} — https://pubmed.ncbi.nlm.nih.gov/{uid}/"
            )

        return f"## PubMed Results for: {query}\n\n" + "\n\n---\n\n".join(results)

    except Exception as e:
        logger.error("PubMed search error: %s", e)
        return f"PubMed search failed: {e}. Try tavily_quick_search as fallback."


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["web-search"])
