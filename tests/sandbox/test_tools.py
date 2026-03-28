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
# write_store_data_to_sandbox tests
# ---------------------------------------------------------------------------


def _make_runtime_with_ns(store=None, allowed_namespaces=None):
    """Return a mock ToolRuntime with optional namespace restriction."""
    runtime = MagicMock()
    configurable = {}
    if allowed_namespaces is not None:
        configurable["store_allowed_namespaces"] = allowed_namespaces
    runtime.config = {"configurable": configurable}
    runtime.store = store
    return runtime


@pytest.mark.unit
class TestWriteStoreDataToSandbox:
    def test_is_named(self):
        from muffin_agent.sandbox.tools import write_store_data_to_sandbox

        assert write_store_data_to_sandbox.name == "write_store_data_to_sandbox"

    @pytest.mark.asyncio
    async def test_writes_store_entry_to_sandbox(self):
        from muffin_agent.sandbox.tools import write_store_data_to_sandbox

        item = MagicMock()
        item.value = {"result": "data"}

        store = AsyncMock()
        store.aget = AsyncMock(return_value=item)
        runtime = _make_runtime_with_ns(store=store)

        sandbox = _make_sandbox()

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await write_store_data_to_sandbox.coroutine(
                "computed.dcf", "model_v1", runtime
            )

        assert "Data written to" in result
        assert "/data/store/computed/dcf/model_v1.json" in result
        sandbox.files.write_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_not_found(self):
        from muffin_agent.sandbox.tools import write_store_data_to_sandbox

        store = AsyncMock()
        store.aget = AsyncMock(return_value=None)
        runtime = _make_runtime_with_ns(store=store)

        result = await write_store_data_to_sandbox.coroutine(
            "computed.dcf", "missing", runtime
        )
        assert "Error: no entry found" in result

    @pytest.mark.asyncio
    async def test_no_store(self):
        from muffin_agent.sandbox.tools import write_store_data_to_sandbox

        runtime = _make_runtime_with_ns(store=None)
        result = await write_store_data_to_sandbox.coroutine(
            "computed.dcf", "key", runtime
        )
        assert "Error: no store available" in result

    @pytest.mark.asyncio
    async def test_namespace_denied(self):
        from muffin_agent.sandbox.tools import write_store_data_to_sandbox

        store = AsyncMock()
        runtime = _make_runtime_with_ns(store=store, allowed_namespaces=["cache"])
        result = await write_store_data_to_sandbox.coroutine(
            "secret.data", "key", runtime
        )
        assert "Access denied" in result


# ---------------------------------------------------------------------------
# read_sandbox_file_to_store tests
# ---------------------------------------------------------------------------


def _make_read_sandbox(content_bytes=b'{"nav": 150.5}'):
    """Build a mock sandbox for read operations."""
    sandbox = MagicMock()
    sandbox.__aenter__ = AsyncMock(return_value=sandbox)
    sandbox.__aexit__ = AsyncMock(return_value=False)
    sandbox.files.read_bytes = AsyncMock(return_value=content_bytes)
    return sandbox


@pytest.mark.unit
class TestReadSandboxFileToStore:
    def test_is_named(self):
        from muffin_agent.sandbox.tools import read_sandbox_file_to_store

        assert read_sandbox_file_to_store.name == "read_sandbox_file_to_store"

    @pytest.mark.asyncio
    async def test_reads_json_file_and_stores(self):
        from muffin_agent.sandbox.tools import read_sandbox_file_to_store

        store = AsyncMock()
        store.aput = AsyncMock()
        runtime = _make_runtime_with_ns(store=store)

        content = json.dumps({"nav": 150.5}).encode("utf-8")
        sandbox = _make_read_sandbox(content_bytes=content)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await read_sandbox_file_to_store.coroutine(
                "/data/computed/result.json", "computed.dcf", "v1", runtime
            )

        assert "Stored" in result
        store.aput.assert_awaited_once()
        call_args = store.aput.call_args
        assert call_args.args[0] == ("computed", "dcf")
        assert call_args.args[1] == "v1"
        assert call_args.args[2] == {"nav": 150.5}

    @pytest.mark.asyncio
    async def test_non_json_file_wraps_in_content(self):
        from muffin_agent.sandbox.tools import read_sandbox_file_to_store

        store = AsyncMock()
        store.aput = AsyncMock()
        runtime = _make_runtime_with_ns(store=store)

        sandbox = _make_read_sandbox(content_bytes=b"plain text data")

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await read_sandbox_file_to_store.coroutine(
                "/data/output.txt", "computed.text", "v1", runtime
            )

        assert "Stored" in result
        stored_val = store.aput.call_args.args[2]
        assert stored_val == {"content": "plain text data"}

    @pytest.mark.asyncio
    async def test_no_store(self):
        from muffin_agent.sandbox.tools import read_sandbox_file_to_store

        runtime = _make_runtime_with_ns(store=None)
        sandbox = _make_read_sandbox()

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await read_sandbox_file_to_store.coroutine(
                "/data/file.json", "computed.dcf", "v1", runtime
            )
        assert "Error: no store available" in result

    @pytest.mark.asyncio
    async def test_namespace_denied(self):
        from muffin_agent.sandbox.tools import read_sandbox_file_to_store

        store = AsyncMock()
        runtime = _make_runtime_with_ns(store=store, allowed_namespaces=["cache"])
        sandbox = _make_read_sandbox()

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await read_sandbox_file_to_store.coroutine(
                "/data/file.json", "secret.data", "v1", runtime
            )
        assert "Access denied" in result
