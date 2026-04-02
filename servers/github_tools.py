"""MCP server for GitHub public repository fetching. Port 8013 (dev only)."""

import base64
import json
import logging
import os
import re

import requests
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

logger = logging.getLogger("mcp_tool_servers.github_tools")

mcp = FastMCP("github-tools", instructions="GitHub public repository fetching tool.")

# Token is loaded once at module level — stored only in MCP server env, never in agent envs.
_GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")
_BASE_HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "mcp-tool-servers/1.0",
}
if _GITHUB_TOKEN:
    _BASE_HEADERS["Authorization"] = f"Bearer {_GITHUB_TOKEN}"

# Files/paths to skip during tree traversal
_SKIP_PATTERN = re.compile(
    r"(node_modules|\.git|__pycache__|\.venv|venv|dist|build|coverage"
    r"|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.idea|\.vscode"
    r"|migrations|fixtures|vendor|\.next|\.nuxt)/|"
    r"\.(pyc|pyo|so|dylib|dll|exe|bin|lock|svg|png|jpg|jpeg|gif|ico"
    r"|woff2?|ttf|otf|eot|map|min\.js|min\.css)$",
    re.IGNORECASE,
)

# File priority: higher score = fetch first
_PRIORITY_PATTERNS: list[tuple[int, re.Pattern[str]]] = [
    (10, re.compile(r"^README", re.IGNORECASE)),
    (9,  re.compile(r"^(main|app|index|server|core)\.(py|ts|js|go|java|rs|rb)$")),
    (8,  re.compile(r"^(pyproject\.toml|package\.json|go\.mod|Cargo\.toml|pom\.xml|build\.gradle)$")),
    (7,  re.compile(r"\.(py|ts|go|java|rs|rb|jsx|tsx)$")),
    (6,  re.compile(r"\.(js|c|cpp|h|cs|swift|kt)$")),
    (5,  re.compile(r"\.(ya?ml|toml|json|env\.example)$")),
    (4,  re.compile(r"\.(md|txt)$")),
]

_MAX_FILE_BYTES = 60_000
_REQUEST_TIMEOUT = 20  # seconds per HTTP call
_MAX_FILES_HARD_CAP = 60


def _file_priority(path: str) -> int:
    name = os.path.basename(path)
    for score, pattern in _PRIORITY_PATTERNS:
        if pattern.search(name):
            return score
    return 3


def _validate_and_parse_url(repo_url: str) -> tuple[str, str]:
    """
    Validate a GitHub URL and extract (owner, repo).

    Security invariants enforced here:
      - Only github.com is allowed — blocks SSRF to internal services or other hosts
      - Owner and repo name must match GitHub's allowed character set
      - No credentials can be embedded in the URL (e.g. user:pass@github.com)
    Raises ValueError with a user-friendly message on any violation.
    """
    url = repo_url.strip().rstrip("/")

    # Block embedded credentials (user:pass@host)
    if "@" in url.split("/")[2] if url.count("/") >= 2 else url:
        raise ValueError("URLs with embedded credentials are not allowed.")

    # Strip scheme
    normalized = re.sub(r"^https?://(www\.)?", "", url, flags=re.IGNORECASE)

    # Enforce github.com only — no IP addresses, no other domains
    if not normalized.lower().startswith("github.com/"):
        raise ValueError(
            f"Only public github.com repositories are supported. Received: {url!r}"
        )

    path_part = normalized[len("github.com/"):]
    # Strip any fragment or query string
    path_part = re.split(r"[?#]", path_part)[0]

    parts = [p for p in path_part.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"Could not parse owner/repo from URL {url!r}. "
            "Expected format: https://github.com/owner/repo"
        )

    owner = parts[0]
    repo = parts[1].removesuffix(".git")

    # Validate characters — GitHub allows letters, digits, hyphens, underscores, dots
    _valid = re.compile(r"^[a-zA-Z0-9._-]{1,100}$")
    if not _valid.match(owner):
        raise ValueError(f"Invalid owner name in URL: {owner!r}")
    if not _valid.match(repo):
        raise ValueError(f"Invalid repository name in URL: {repo!r}")

    return owner, repo


@mcp.tool()
def fetch_github_repo(repo_url: str, max_files: int = 40) -> str:
    """Fetch key source files from a public GitHub repository for code analysis.

    Retrieves repo metadata, a full file tree, and the content of the most
    important files (READMEs, entry points, source code, config files).

    Args:
        repo_url: Public GitHub repository URL (e.g. https://github.com/owner/repo).
                  Only github.com public repositories are supported.
        max_files: Maximum number of files to fetch content for (1–60, default 40).

    Returns:
        JSON string with keys: repo_url, repo_name, owner, language, description,
        file_tree (list[str]), key_files (list[{path, content}]),
        summary (str), total_files (int).
        Returns an "Error: ..." string on failure — never raises.
    """
    try:
        owner, repo = _validate_and_parse_url(repo_url)
    except ValueError as exc:
        return f"Error: {exc}"

    repo_full = f"{owner}/{repo}"
    max_files = max(1, min(int(max_files), _MAX_FILES_HARD_CAP))

    try:
        # ── 1. Repo metadata ──
        meta_res = requests.get(
            f"https://api.github.com/repos/{repo_full}",
            headers=_BASE_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        if meta_res.status_code == 404:
            return f"Error: Repository not found or is private: {repo_full}"
        if meta_res.status_code == 403:
            return (
                f"Error: Access denied for {repo_full}. "
                "Only public repositories are supported."
            )
        if not meta_res.ok:
            return f"Error: GitHub API returned {meta_res.status_code} for {repo_full}"
        meta = meta_res.json()

        # ── 2. Full recursive file tree ──
        default_branch = meta.get("default_branch", "main")
        tree_res = requests.get(
            f"https://api.github.com/repos/{repo_full}/git/trees/{default_branch}?recursive=1",
            headers=_BASE_HEADERS,
            timeout=_REQUEST_TIMEOUT,
        )
        if not tree_res.ok:
            return f"Error: Failed to fetch file tree (status {tree_res.status_code})"

        all_blobs = [
            item
            for item in tree_res.json().get("tree", [])
            if item["type"] == "blob" and not _SKIP_PATTERN.search(item["path"])
        ]
        all_blobs.sort(key=lambda f: (-_file_priority(f["path"]), len(f["path"])))
        to_fetch = all_blobs[:max_files]

        # ── 3. Fetch file contents ──
        key_files: list[dict[str, str]] = []
        for item in to_fetch:
            try:
                content_res = requests.get(
                    f"https://api.github.com/repos/{repo_full}/contents/{item['path']}",
                    headers=_BASE_HEADERS,
                    timeout=_REQUEST_TIMEOUT,
                )
                if not content_res.ok:
                    continue
                file_json = content_res.json()
                if file_json.get("encoding") != "base64":
                    continue
                raw = base64.b64decode(file_json["content"]).decode("utf-8", errors="replace")
                if len(raw) > _MAX_FILE_BYTES:
                    raw = raw[:_MAX_FILE_BYTES] + f"\n\n... [truncated at {_MAX_FILE_BYTES} bytes]"
                key_files.append({"path": item["path"], "content": raw})
            except Exception as exc:
                logger.warning("Skipping %s — %s", item["path"], exc)
                continue

        # ── 4. Build compact summary ──
        all_paths = [item["path"] for item in all_blobs[:200]]
        language = meta.get("language") or "unknown"
        description = meta.get("description") or ""

        summary_parts: list[str] = []
        if description:
            summary_parts.append(f"Description: {description}")
        summary_parts.append(f"Primary language: {language}")
        summary_parts.append(f"Total visible files: {len(all_blobs)}")
        summary_parts.append(f"Files fetched: {len(key_files)}")
        if all_paths:
            summary_parts.append("\nFile tree (up to 80 paths):")
            summary_parts.extend(f"  {p}" for p in all_paths[:80])

        result = {
            "repo_url": repo_url,
            "repo_name": repo,
            "owner": owner,
            "language": language,
            "description": description,
            "file_tree": all_paths,
            "key_files": key_files,
            "summary": "\n".join(summary_parts),
            "total_files": len(all_blobs),
        }
        return json.dumps(result)

    except requests.exceptions.Timeout:
        return f"Error: Request timed out fetching {repo_full}."
    except requests.exceptions.ConnectionError as exc:
        return f"Error: Connection failed for {repo_full}: {exc}"
    except Exception as exc:
        logger.error("Unexpected error fetching repo %s: %s", repo_full, exc)
        return f"Error: Unexpected failure fetching {repo_full}: {exc}"


if __name__ == "__main__":
    port = int(os.getenv("MCP_GITHUB_TOOLS_PORT", 8013))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
