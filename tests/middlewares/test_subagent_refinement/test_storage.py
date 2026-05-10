"""Tests for the storage layer (path builder + read/write helpers)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from muffin_agent.middlewares.subagent_refinement import CollectionFindings
from muffin_agent.middlewares.subagent_refinement.storage import (
    call_id_path,
    extract_prior_call_id,
    read_findings,
    write_findings,
)


@pytest.mark.unit
class TestCallIdPath:
    def test_basic_path(self):
        assert call_id_path("abc123") == "/scratch/subagent_runs/abc123.json"

    @pytest.mark.parametrize(
        "bad",
        ["", "../../etc/passwd", "ab cd", "ab/cd", "ab.cd", "x" * 100],
    )
    def test_rejects_unsafe_ids(self, bad):
        with pytest.raises(ValueError):
            call_id_path(bad)


@pytest.mark.unit
class TestExtractPriorCallId:
    def test_finds_marker(self):
        assert (
            extract_prior_call_id("Fill remaining gaps. prior_call_id=abc123 thanks")
            == "abc123"
        )

    def test_returns_none_when_absent(self):
        assert extract_prior_call_id("just a normal task") is None

    def test_rejects_unsafe_id(self):
        # Slashes / dots not in the safe charset.
        assert extract_prior_call_id("prior_call_id=a/b") is None

    def test_handles_non_string_input(self):
        assert extract_prior_call_id(None) is None  # type: ignore[arg-type]


@pytest.mark.unit
class TestReadFindings:
    @pytest.mark.asyncio
    async def test_returns_parsed_findings(self):
        backend = MagicMock()
        payload = CollectionFindings(call_id="abc").model_dump_json()
        backend.aread = AsyncMock(
            return_value=MagicMock(
                error=None,
                file_data={"content": payload, "encoding": "utf-8"},
            )
        )
        result = await read_findings(backend, "abc")
        assert result is not None
        assert result.call_id == "abc"

    @pytest.mark.asyncio
    async def test_returns_none_on_error_result(self):
        backend = MagicMock()
        backend.aread = AsyncMock(
            return_value=MagicMock(error="not found", file_data=None)
        )
        assert await read_findings(backend, "abc") is None

    @pytest.mark.asyncio
    async def test_returns_none_on_unparseable_payload(self):
        backend = MagicMock()
        backend.aread = AsyncMock(
            return_value=MagicMock(error=None, file_data={"content": "not json"})
        )
        assert await read_findings(backend, "abc") is None

    @pytest.mark.asyncio
    async def test_returns_none_when_backend_raises(self):
        backend = MagicMock()
        backend.aread = AsyncMock(side_effect=RuntimeError("io fail"))
        assert await read_findings(backend, "abc") is None


@pytest.mark.unit
class TestWriteFindings:
    @pytest.mark.asyncio
    async def test_writes_to_canonical_path(self):
        backend = MagicMock()
        backend.awrite = AsyncMock()
        findings = CollectionFindings(call_id="abc")

        await write_findings(backend, findings)

        backend.awrite.assert_awaited_once()
        path, payload = backend.awrite.call_args.args
        assert path == "/scratch/subagent_runs/abc.json"
        assert '"call_id":"abc"' in payload

    @pytest.mark.asyncio
    async def test_swallows_backend_failure(self):
        backend = MagicMock()
        backend.awrite = AsyncMock(side_effect=RuntimeError("write fail"))
        findings = CollectionFindings(call_id="abc")
        # Must not raise.
        await write_findings(backend, findings)
