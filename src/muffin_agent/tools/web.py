"""Web search and crawling tools.

Provides five async LangChain tools backed by self-hosted SearxNG and
Firecrawl OSS services, plus MarkItDown for document conversion:

- ``web_search`` — full-text web search via SearxNG JSON API.
- ``web_scrape`` — scrape a single URL to Markdown via Firecrawl.
- ``web_crawl`` — crawl a site (follow links) and return all page content.
- ``web_map`` — discover URLs on a site without fetching content.
- ``convert_document`` — download a file (PDF, Word, Excel, …) and convert to Markdown.
"""

import asyncio
import os
import tempfile

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from markitdown import MarkItDown
from pydantic import BaseModel

from ..web_config import WebConfiguration

# ── Output schemas ─────────────────────────────────────────────────────────────


class WebSearchResult(BaseModel):
    """Single search result returned by SearxNG."""

    title: str
    url: str
    content: str
    engine: str | None = None


class WebSearchOutput(BaseModel):
    """Output schema for web_search."""

    query: str
    results: list[WebSearchResult]
    total: int


# ── Helpers ────────────────────────────────────────────────────────────────────


def _auth_headers(api_key: str) -> dict[str, str]:
    """Return Firecrawl authorization headers."""
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _infer_suffix(url: str, content_type: str) -> str:
    """Infer a file suffix from URL path or Content-Type header."""
    _ct_map = {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        # fmt: off
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",  # noqa: E501
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/vnd.ms-powerpoint": ".ppt",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",  # noqa: E501
        # fmt: on
        "text/csv": ".csv",
        "application/json": ".json",
        "text/xml": ".xml",
        "application/xml": ".xml",
        "text/html": ".html",
    }
    # Try URL extension first
    path = url.split("?")[0].split("#")[0]
    if "." in path.split("/")[-1]:
        return "." + path.rsplit(".", 1)[-1].lower()
    # Fall back to Content-Type
    base_ct = content_type.split(";")[0].strip().lower()
    return _ct_map.get(base_ct, ".bin")


# ── Tools ──────────────────────────────────────────────────────────────────────


@tool(
    parse_docstring=True,
    extras={"output_schema": WebSearchOutput.model_json_schema()},
)
async def web_search(
    query: str,
    num_results: int = 10,
    category: str = "general",
    runtime: ToolRuntime = None,
) -> dict:
    """Search the web and return titles, URLs, and result snippets.

    Uses a self-hosted SearxNG meta-search engine (aggregates DuckDuckGo,
    Bing, Google, Brave, and others without tracking). Prefer this tool when
    you need to discover relevant URLs before scraping.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return (1–50). Default 10.
        category: SearxNG category: ``general``, ``news``, ``science``,
            ``social media``, ``videos``, or ``files``. Default ``general``.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        A dict with ``query``, ``results`` (list of title/url/content/engine),
        and ``total`` count.
    """
    cfg = WebConfiguration.from_runnable_config(runtime.config if runtime else {})
    params = {
        "q": query,
        "format": "json",
        "category": category,
    }
    if num_results:
        params["limit"] = str(num_results)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{cfg.searxng_url}/search", params=params)
        resp.raise_for_status()
        data = resp.json()

    raw_results = data.get("results", [])[:num_results]
    results = [
        WebSearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            engine=r.get("engine"),
        )
        for r in raw_results
    ]
    return WebSearchOutput(
        query=query,
        results=results,
        total=len(results),
    ).model_dump()


@tool(parse_docstring=True)
async def web_scrape(
    url: str,
    runtime: ToolRuntime = None,
) -> str:
    """Scrape a single URL and return its content as Markdown.

    Uses Firecrawl, which handles JavaScript-rendered pages and cleans the
    output into readable Markdown. Prefer this over ``web_crawl`` when you
    already have the exact URL and only need one page.

    Args:
        url: The full URL to scrape (must include scheme, e.g. ``https://``).
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Page content as Markdown, or an error description if scraping fails.
    """
    cfg = WebConfiguration.from_runnable_config(runtime.config if runtime else {})
    payload = {"url": url, "formats": ["markdown"]}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{cfg.firecrawl_url}/v1/scrape",
            headers=_auth_headers(cfg.firecrawl_api_key),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    result = data.get("data", data)
    return result.get("markdown") or result.get("content") or "(no content returned)"


@tool(parse_docstring=True)
async def web_crawl(
    url: str,
    max_pages: int = 5,
    include_paths: list[str] | None = None,
    runtime: ToolRuntime = None,
) -> list[dict]:
    """Crawl a website by following internal links from a start URL.

    Returns Markdown content for each discovered page. Use this for:
    - IR (Investor Relations) sites with multiple sub-pages
    - Documentation sites
    - Multi-page reports or annual reports hosted on a company domain

    Prefer ``web_scrape`` for a single known URL. ``web_crawl`` is more
    expensive and slower but collects content across an entire section of a site.

    Args:
        url: Start URL. Firecrawl will follow links within the same domain.
        max_pages: Maximum number of pages to scrape (1–50). Default 5.
        include_paths: Optional list of URL path prefixes to restrict crawling
            to (e.g. ``["/investor-relations", "/press-releases"]``). When
            omitted, the full domain is crawled up to ``max_pages``.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        List of dicts, each with ``url``, ``markdown``, and ``metadata`` keys.
        Empty list if the crawl fails.
    """
    cfg = WebConfiguration.from_runnable_config(runtime.config if runtime else {})
    payload: dict = {
        "url": url,
        "limit": max_pages,
        "scrapeOptions": {"formats": ["markdown"]},
    }
    if include_paths:
        payload["includePaths"] = include_paths

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Submit crawl job
        resp = await client.post(
            f"{cfg.firecrawl_url}/v1/crawl",
            headers=_auth_headers(cfg.firecrawl_api_key),
            json=payload,
        )
        resp.raise_for_status()
        job = resp.json()
        job_id = job.get("id") or job.get("jobId")
        if not job_id:
            return []

        # Poll until done (max ~5 minutes)
        for _ in range(60):
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"{cfg.firecrawl_url}/v1/crawl/{job_id}",
                headers=_auth_headers(cfg.firecrawl_api_key),
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            if status_data.get("status") in ("completed", "failed"):
                break

    return status_data.get("data", [])


@tool(parse_docstring=True)
async def web_map(
    url: str,
    runtime: ToolRuntime = None,
) -> list[str]:
    """Discover all URLs on a website without fetching page content.

    Returns a flat list of URLs found on the site. Use this before
    ``web_scrape`` or ``web_crawl`` to understand a site's structure and
    decide which specific pages are worth fetching.

    Args:
        url: Base URL of the site to map (e.g. ``https://ir.nvidia.com``).
        runtime: Injected by LangGraph ToolNode.

    Returns:
        List of discovered URLs. Empty list if the request fails.
    """
    cfg = WebConfiguration.from_runnable_config(runtime.config if runtime else {})
    payload = {"url": url}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{cfg.firecrawl_url}/v1/map",
            headers=_auth_headers(cfg.firecrawl_api_key),
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    return data.get("links", [])


@tool(parse_docstring=True)
async def convert_document(
    url: str,
    runtime: ToolRuntime = None,
) -> str:
    """Download a document from a URL and convert it to Markdown.

    Supports: PDF, Word (.docx/.doc), Excel (.xlsx/.xls), PowerPoint
    (.pptx/.ppt), HTML, CSV, JSON, XML, ZIP archives, audio (transcription),
    and Outlook messages (.msg).

    Use this when the URL points directly to a file, not a rendered web page.
    For HTML pages use ``web_scrape`` instead (it handles JS rendering better).

    Args:
        url: Direct URL to the document file (e.g. an SEC filing PDF or an
            Excel spreadsheet linked from an IR page).
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Document content as Markdown text, or an error description.
    """
    async with httpx.AsyncClient(
        timeout=120.0, follow_redirects=True
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    suffix = _infer_suffix(url, resp.headers.get("content-type", ""))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(resp.content)
            tmp_path = f.name

        md = MarkItDown()
        result = await asyncio.get_event_loop().run_in_executor(
            None, md.convert, tmp_path
        )
        return result.text_content or "(empty document)"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
