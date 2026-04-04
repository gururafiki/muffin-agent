"""Unit tests for web search and crawling tools."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.tools.web import (
    WebSearchOutput,
    WebSearchResult,
    _infer_suffix,
    convert_document,
    web_crawl,
    web_map,
    web_scrape,
    web_search,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_runtime(
    searxng_url: str = "http://searxng:8080",
    firecrawl_url: str = "http://firecrawl:3002",
    firecrawl_api_key: str = "test-key",
) -> MagicMock:
    """Build a minimal ToolRuntime mock with WebConfiguration baked in."""
    runtime = MagicMock()
    runtime.config = {
        "configurable": {
            "searxng_url": searxng_url,
            "firecrawl_url": firecrawl_url,
            "firecrawl_api_key": firecrawl_api_key,
        }
    }
    return runtime


def _mock_response(
    status_code: int = 200,
    json_data: dict | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = text
    resp.content = text.encode() if text else b""
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


# ── _infer_suffix ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestInferSuffix:
    def test_pdf_from_url(self):
        assert _infer_suffix("https://example.com/report.pdf", "") == ".pdf"

    def test_xlsx_from_url(self):
        assert _infer_suffix("https://example.com/data.xlsx", "") == ".xlsx"

    def test_pdf_from_content_type(self):
        assert _infer_suffix("https://example.com/doc", "application/pdf") == ".pdf"

    def test_docx_from_content_type(self):
        ct = (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
        assert _infer_suffix("https://example.com/file", ct) == ".docx"

    def test_url_extension_beats_content_type(self):
        # URL says .xlsx but Content-Type says pdf — URL wins
        result = _infer_suffix("https://example.com/file.xlsx", "application/pdf")
        assert result == ".xlsx"

    def test_unknown_falls_back_to_bin(self):
        result = _infer_suffix(
            "https://example.com/blob", "application/octet-stream"
        )
        assert result == ".bin"

    def test_query_string_stripped(self):
        assert _infer_suffix("https://example.com/report.pdf?token=abc", "") == ".pdf"


# ── WebSearchOutput schema ────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebSearchOutput:
    def test_schema_title(self):
        assert WebSearchOutput.model_json_schema()["title"] == "WebSearchOutput"

    def test_result_list_serialises(self):
        output = WebSearchOutput(
            query="test",
            results=[WebSearchResult(title="T", url="https://a.com", content="C")],
            total=1,
        )
        d = output.model_dump()
        assert d["total"] == 1
        assert d["results"][0]["url"] == "https://a.com"


# ── web_search ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebSearch:
    @pytest.mark.asyncio
    async def test_calls_searxng_json_endpoint(self):
        searxng_response = {
            "results": [
                {
                    "title": "NVIDIA Q4",
                    "url": "https://nvidia.com/q4",
                    "content": "Strong results",
                    "engine": "bing",
                },
            ]
        }
        mock_resp = _mock_response(json_data=searxng_response)
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            # Call .coroutine() directly to bypass Pydantic schema validation
            # on the injected `runtime` parameter (same pattern as sandbox tests).
            result = await web_search.coroutine(
                query="NVIDIA Q4 earnings", num_results=5, runtime=runtime
            )

        assert result["total"] == 1
        assert result["results"][0]["title"] == "NVIDIA Q4"
        call_kwargs = mock_client.get.call_args
        assert "/search" in call_kwargs[0][0]
        assert call_kwargs[1]["params"]["format"] == "json"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        mock_resp = _mock_response(json_data={"results": []})
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await web_search.coroutine(
                query="obscure query xyz", runtime=runtime
            )

        assert result["total"] == 0
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_respects_num_results_limit(self):
        # Return 10 results from SearxNG but request only 3
        searxng_response = {
            "results": [
                {"title": f"R{i}", "url": f"https://example.com/{i}", "content": "x"}
                for i in range(10)
            ]
        }
        mock_resp = _mock_response(json_data=searxng_response)
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await web_search.coroutine(
                query="test", num_results=3, runtime=runtime
            )

        assert result["total"] == 3


# ── web_scrape ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebScrape:
    @pytest.mark.asyncio
    async def test_returns_markdown_from_firecrawl(self):
        firecrawl_response = {"data": {"markdown": "# Hello World\n\nContent here."}}
        mock_resp = _mock_response(json_data=firecrawl_response)
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await web_scrape.coroutine(
                url="https://nvidia.com", runtime=runtime
            )

        assert "Hello World" in result
        call_kwargs = mock_client.post.call_args
        assert "/v1/scrape" in call_kwargs[0][0]
        assert call_kwargs[1]["json"]["url"] == "https://nvidia.com"

    @pytest.mark.asyncio
    async def test_falls_back_to_content_key(self):
        firecrawl_response = {"data": {"content": "Fallback content"}}
        mock_resp = _mock_response(json_data=firecrawl_response)
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await web_scrape.coroutine(
                url="https://example.com", runtime=runtime
            )

        assert result == "Fallback content"

    @pytest.mark.asyncio
    async def test_sends_auth_header(self):
        mock_resp = _mock_response(json_data={"data": {"markdown": "ok"}})
        runtime = _mock_runtime(firecrawl_api_key="my-secret")

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            await web_scrape.coroutine(url="https://example.com", runtime=runtime)

        headers = mock_client.post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-secret"


# ── web_map ───────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebMap:
    @pytest.mark.asyncio
    async def test_returns_link_list(self):
        firecrawl_response = {
            "links": [
                "https://nvidia.com/about",
                "https://nvidia.com/products",
                "https://nvidia.com/ir",
            ]
        }
        mock_resp = _mock_response(json_data=firecrawl_response)
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await web_map.coroutine(url="https://nvidia.com", runtime=runtime)

        assert len(result) == 3
        assert "https://nvidia.com/ir" in result

    @pytest.mark.asyncio
    async def test_empty_links_returns_empty_list(self):
        mock_resp = _mock_response(json_data={})
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await web_map.coroutine(url="https://example.com", runtime=runtime)

        assert result == []


# ── web_crawl ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestWebCrawl:
    @pytest.mark.asyncio
    async def test_submits_job_and_polls_until_done(self):
        submit_resp = _mock_response(json_data={"id": "job-123"})
        polling_resp = _mock_response(
            json_data={
                "status": "completed",
                "data": [
                    {"url": "https://ir.nvidia.com", "markdown": "# IR Page"},
                ],
            }
        )
        runtime = _mock_runtime()

        with (
            patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls,
            patch("muffin_agent.tools.web.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client.get = AsyncMock(return_value=polling_resp)
            mock_client_cls.return_value = mock_client

            result = await web_crawl.coroutine(
                url="https://ir.nvidia.com", max_pages=3, runtime=runtime
            )

        assert len(result) == 1
        assert result[0]["markdown"] == "# IR Page"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_job_id(self):
        submit_resp = _mock_response(json_data={})
        runtime = _mock_runtime()

        with patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=submit_resp)
            mock_client_cls.return_value = mock_client

            result = await web_crawl.coroutine(
                url="https://example.com", runtime=runtime
            )

        assert result == []


# ── convert_document ──────────────────────────────────────────────────────────


@pytest.mark.unit
class TestConvertDocument:
    @pytest.mark.asyncio
    async def test_downloads_and_converts_pdf(self):
        pdf_bytes = b"%PDF-1.4 fake content"
        download_resp = MagicMock()
        download_resp.content = pdf_bytes
        download_resp.headers = {"content-type": "application/pdf"}
        download_resp.raise_for_status = MagicMock()
        runtime = _mock_runtime()

        mock_result = MagicMock()
        mock_result.text_content = "# Extracted PDF Content"

        with (
            patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls,
            # MarkItDown is imported locally inside the function; patch via sys.modules
            patch("muffin_agent.tools.web.MarkItDown") as mock_md_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=download_resp)
            mock_client_cls.return_value = mock_client

            mock_md_instance = MagicMock()
            mock_md_instance.convert.return_value = mock_result
            mock_md_cls.return_value = mock_md_instance

            result = await convert_document.coroutine(
                url="https://example.com/report.pdf", runtime=runtime
            )

        assert result == "# Extracted PDF Content"

    @pytest.mark.asyncio
    async def test_returns_empty_document_message_when_no_content(self):
        download_resp = MagicMock()
        download_resp.content = b"data"
        download_resp.headers = {"content-type": "application/pdf"}
        download_resp.raise_for_status = MagicMock()
        runtime = _mock_runtime()

        mock_result = MagicMock()
        mock_result.text_content = ""

        with (
            patch("muffin_agent.tools.web.httpx.AsyncClient") as mock_client_cls,
            patch("muffin_agent.tools.web.MarkItDown") as mock_md_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=download_resp)
            mock_client_cls.return_value = mock_client

            mock_md_instance = MagicMock()
            mock_md_instance.convert.return_value = mock_result
            mock_md_cls.return_value = mock_md_instance

            result = await convert_document.coroutine(
                url="https://example.com/empty.pdf", runtime=runtime
            )

        assert result == "(empty document)"
