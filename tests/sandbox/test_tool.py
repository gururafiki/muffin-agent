"""Unit tests for sandbox tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATCH_AGET = "muffin_agent.sandbox.tools.aget_sandbox"


def _make_execution(stdout_texts=(), stderr_texts=(), error=None, cmd_id="cmd-1"):
    """Build a mock async Execution object."""
    from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage

    logs = ExecutionLogs()
    for t in stdout_texts:
        logs.add_stdout(OutputMessage(text=t, timestamp=0))
    for t in stderr_texts:
        logs.add_stderr(OutputMessage(text=t, timestamp=0))

    return Execution(id=cmd_id, result=[], error=error, logs=logs)


def _make_sandbox(
    *,
    write_raises=False,
    exec_output="42\n",
    exec_exit_code=0,
):
    """Build a mock async Sandbox usable as an async context manager."""
    from opensandbox.models.execd import CommandStatus

    sandbox = MagicMock()
    sandbox.__aenter__ = AsyncMock(return_value=sandbox)
    sandbox.__aexit__ = AsyncMock(return_value=False)

    if write_raises:
        sandbox.files.write_file = AsyncMock(side_effect=PermissionError("denied"))
    else:
        sandbox.files.write_file = AsyncMock()

    run_result = _make_execution(stdout_texts=[exec_output] if exec_output else [])
    cleanup_result = _make_execution()
    sandbox.commands.run = AsyncMock(side_effect=[run_result, cleanup_result])

    sandbox.commands.get_command_status = AsyncMock(
        return_value=CommandStatus(exit_code=exec_exit_code)
    )

    return sandbox


def _make_runtime(store=None):
    """Return a mock ToolRuntime."""
    runtime = MagicMock()
    runtime.config = {"configurable": {}}
    runtime.store = store
    return runtime


async def _invoke(code):
    """Call execute_python's underlying coroutine directly."""
    from muffin_agent.sandbox.tools import execute_python

    runtime = _make_runtime()
    return await execute_python.coroutine(code, runtime)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutePythonTool:
    def test_is_named_execute_python(self):
        from muffin_agent.sandbox.tools import execute_python

        assert execute_python.name == "execute_python"

    @pytest.mark.asyncio
    async def test_successful_execution_returns_output(self):
        sandbox = _make_sandbox(exec_output="hello world\n", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("print('hello world')")

        assert result == "hello world\n"
        sandbox.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_failure_returns_error_message(self):
        sandbox = _make_sandbox(write_raises=True)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("print(1)")

        assert "Failed to write" in result
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_returns_error_with_output(self):
        err_output = "NameError: name 'x' is not defined\n"
        sandbox = _make_sandbox(exec_output=err_output, exec_exit_code=1)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("print(x)")

        assert "Execution failed" in result
        assert "exit 1" in result
        assert "NameError" in result

    @pytest.mark.asyncio
    async def test_no_output_returns_placeholder(self):
        sandbox = _make_sandbox(exec_output="", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("x = 1 + 1")

        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_file(self):
        sandbox = _make_sandbox(exec_output="ok\n", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            await _invoke("print('ok')")

        assert sandbox.commands.run.call_count == 2
        cleanup_cmd = sandbox.commands.run.call_args_list[1].args[0]
        assert cleanup_cmd.startswith("rm -f /tmp/muffin_exec_")

    @pytest.mark.asyncio
    async def test_sandbox_discovered_via_aget_sandbox(self):
        sandbox = _make_sandbox(exec_output="found\n", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)) as mock_aget:
            result = await _invoke("print('found')")

        mock_aget.assert_called_once()
        assert result == "found\n"


# ---------------------------------------------------------------------------
# discover_cached_data tests
# ---------------------------------------------------------------------------


async def _invoke_discover(store=None):
    """Call discover_cached_data's underlying coroutine directly."""
    from muffin_agent.sandbox.tools import discover_cached_data

    runtime = _make_runtime(store=store)
    return await discover_cached_data.coroutine(runtime)


def _make_store_with_entries(entries):
    """Build a mock store with the given entries.

    entries: list of (namespace, key, value) tuples
    """
    store = AsyncMock()

    namespaces = list({ns for ns, _k, _v in entries})
    store.alist_namespaces = AsyncMock(return_value=namespaces)

    def _search(ns):
        items = []
        for ens, key, val in entries:
            if ens == ns:
                item = MagicMock()
                item.value = val
                item.key = key
                items.append(item)
        return items

    store.asearch = AsyncMock(side_effect=_search)
    return store


@pytest.mark.unit
class TestDiscoverCachedDataTool:
    def test_is_named_discover_cached_data(self):
        from muffin_agent.sandbox.tools import discover_cached_data

        assert discover_cached_data.name == "discover_cached_data"

    @pytest.mark.asyncio
    async def test_returns_json_from_store(self):
        """Successful run returns JSON array from store entries."""
        store = _make_store_with_entries([
            (
                ("cache", "tool_a"),
                "abc123",
                {
                    "tool_name": "tool_a",
                    "args": {"x": 1},
                    "cached_at": "2026-03-22T14:00:00",
                    "content_size": 100,
                    "content": "data",
                },
            ),
        ])
        result = await _invoke_discover(store=store)

        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["tool_name"] == "tool_a"
        assert parsed[0]["store_key"] == "abc123"

    @pytest.mark.asyncio
    async def test_empty_store_returns_empty_array(self):
        """No entries returns '[]'."""
        store = AsyncMock()
        store.alist_namespaces = AsyncMock(return_value=[])

        result = await _invoke_discover(store=store)

        assert result == "[]"

    @pytest.mark.asyncio
    async def test_no_store_returns_empty_array(self):
        """runtime.store is None returns '[]'."""
        result = await _invoke_discover(store=None)

        assert result == "[]"


# ---------------------------------------------------------------------------
# write_tool_output_to_backend tests
# ---------------------------------------------------------------------------


async def _invoke_write(tool_name, args_hash, store=None, file_path=None):
    """Call write_tool_output_to_backend's underlying coroutine directly."""
    from muffin_agent.sandbox.tools import write_tool_output_to_backend

    runtime = _make_runtime(store=store)
    return await write_tool_output_to_backend.coroutine(
        tool_name, args_hash, runtime, file_path=file_path,
    )


@pytest.mark.unit
class TestWriteToolOutputToBackendTool:
    def test_is_named_write_tool_output_to_backend(self):
        from muffin_agent.sandbox.tools import write_tool_output_to_backend

        assert write_tool_output_to_backend.name == "write_tool_output_to_backend"

    @pytest.mark.asyncio
    async def test_writes_store_content_to_sandbox_file(self):
        """Store content is written to sandbox via aget_sandbox."""
        store = AsyncMock()
        item = MagicMock()
        item.value = {"content": '{"prices": [1, 2, 3]}'}
        store.aget = AsyncMock(return_value=item)

        sandbox = _make_sandbox()

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke_write("tool_a", "abc123", store=store)

        assert "Data written to" in result
        assert "/data/cache/tool_a/abc123.json" in result
        sandbox.files.write_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_error_when_not_found(self):
        """Store miss returns error message."""
        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)

        result = await _invoke_write("tool_a", "abc123", store=store)

        assert "Error: no cached result" in result

    @pytest.mark.asyncio
    async def test_returns_error_when_no_store(self):
        """runtime.store is None returns error message."""
        result = await _invoke_write("tool_a", "abc123", store=None)

        assert "Error: no store available" in result

    @pytest.mark.asyncio
    async def test_custom_file_path(self):
        """Custom file_path is used instead of default."""
        store = AsyncMock()
        item = MagicMock()
        item.value = {"content": "data"}
        store.aget = AsyncMock(return_value=item)

        sandbox = _make_sandbox()

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke_write(
                "tool_a", "abc123", store=store, file_path="/custom/path.json",
            )

        assert "/custom/path.json" in result
        write_call = sandbox.files.write_file.call_args
        assert write_call.args[0] == "/custom/path.json"

    @pytest.mark.asyncio
    async def test_sandbox_write_failure_returns_error(self):
        """Sandbox write error returns descriptive message."""
        store = AsyncMock()
        item = MagicMock()
        item.value = {"content": "data"}
        store.aget = AsyncMock(return_value=item)

        sandbox = _make_sandbox(write_raises=True)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke_write("tool_a", "abc123", store=store)

        assert "Error writing to sandbox" in result


# ---------------------------------------------------------------------------
# get_tool_output_schema tests
# ---------------------------------------------------------------------------

_PATCH_FIND = "muffin_agent.sandbox.tools._find_python_tool_schema"
_PATCH_CREATE_SESSION = "langchain_mcp_adapters.sessions.create_session"
_PATCH_CONFIG = "muffin_agent.config.Configuration"


async def _invoke_schema(tool_name: str):
    """Call get_tool_output_schema's underlying coroutine directly."""
    from muffin_agent.sandbox.tools import get_tool_output_schema

    runtime = _make_runtime()
    return await get_tool_output_schema.coroutine(tool_name, runtime)


@pytest.mark.unit
class TestGetToolOutputSchemaTool:
    def test_is_named_get_tool_output_schema(self):
        from muffin_agent.sandbox.tools import get_tool_output_schema

        assert get_tool_output_schema.name == "get_tool_output_schema"

    @pytest.mark.asyncio
    async def test_returns_python_tool_schema(self):
        """Python tool with extras['output_schema'] returns its JSON schema."""
        fake_schema = {
            "type": "object",
            "properties": {"slope_10y2y_bps": {"type": "number"}},
        }
        with patch(_PATCH_FIND, return_value=fake_schema):
            result = await _invoke_schema("compute_yield_curve_metrics")

        parsed = json.loads(result)
        assert parsed == fake_schema

    @pytest.mark.asyncio
    async def test_returns_mcp_schema_when_found(self):
        """MCP tool with outputSchema is returned when Python lookup misses."""
        mcp_schema = {"type": "object", "properties": {"data": {}}}

        mock_tool = MagicMock()
        mock_tool.name = "equity_price_historical"
        mock_tool.outputSchema = mcp_schema

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_result.nextCursor = None

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_config = MagicMock()
        mock_config.get_mcp_connections.return_value = {
            "openbb": {"url": "http://test", "transport": "streamable_http"},
        }

        with (
            patch(_PATCH_FIND, return_value=None),
            patch(_PATCH_CREATE_SESSION, return_value=mock_session),
            patch(
                _PATCH_CONFIG + ".from_runnable_config",
                return_value=mock_config,
            ),
        ):
            result = await _invoke_schema("equity_price_historical")

        parsed = json.loads(result)
        assert parsed == mcp_schema

    @pytest.mark.asyncio
    async def test_returns_not_found_message(self):
        """Unknown tool name returns a descriptive message."""
        mock_config = MagicMock()
        mock_config.get_mcp_connections.return_value = {}

        with (
            patch(_PATCH_FIND, return_value=None),
            patch(
                _PATCH_CONFIG + ".from_runnable_config",
                return_value=mock_config,
            ),
        ):
            result = await _invoke_schema("nonexistent_tool")

        assert "No output schema found" in result
        assert "nonexistent_tool" in result
