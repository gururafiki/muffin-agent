"""Unit tests for the convert_document tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from muffin_agent.tools.web import _infer_suffix, convert_document

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_runtime() -> MagicMock:
    """Build a minimal ToolRuntime mock."""
    runtime = MagicMock()
    runtime.config = {}
    return runtime


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
