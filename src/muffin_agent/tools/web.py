"""Web document conversion tool.

Provides one async LangChain tool for downloading and converting file-based
documents to Markdown:

- ``convert_document`` — download a file (PDF, Word, Excel, …) and convert
  to Markdown using MarkItDown.

Web search is handled by the LangChain SearxNG integration
(``SearxSearchResults`` via ``load_tools``).  Web scraping, crawling, mapping,
and structured extraction are handled by Firecrawl MCP tools.
"""

import asyncio
import os
import tempfile

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from markitdown import MarkItDown

# ── Helpers ────────────────────────────────────────────────────────────────────


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


# ── Tool ───────────────────────────────────────────────────────────────────────


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
    For HTML pages use ``firecrawl_scrape`` instead (it handles JS rendering).

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
