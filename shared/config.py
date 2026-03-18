import os

PORTS = {
    "web-search": int(os.getenv("MCP_WEB_SEARCH_PORT", 8010)),
    "finance-data": int(os.getenv("MCP_FINANCE_DATA_PORT", 8011)),
    "vector-db": int(os.getenv("MCP_VECTOR_DB_PORT", 8012)),
}

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")

def server_url(name: str) -> str:
    return f"http://localhost:{PORTS[name]}"
