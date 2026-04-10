"""MCP server for web search tools (Tavily + Firecrawl). Port 8010."""

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


@mcp.tool()
def tavily_quick_search(query: str, max_results: int = 3) -> str:
    """Perform a quick web search across the internet. Returns synthesized answers and snippets.
    Ideal for news, quick fact-checking, and broad questions."""
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
        url: The full HTTPS URL to scrape. Must be a valid web address.
    """
    logger.info("Firecrawl scraping — url='%s'", url)
    if not url.startswith(("http://", "https://")):
        return "Error: Invalid URL. Please provide a URL starting with http:// or https://"

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


if __name__ == "__main__":
    from shared.config import PORTS
    mcp.run(transport="streamable-http", host="0.0.0.0", port=PORTS["web-search"])
