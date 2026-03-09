"""Unit tests for create_python_execution_tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from deepagents.backends.protocol import ExecuteResponse


def _make_write_response(error=None):
    """Return a mock write response (mimics BaseSandbox.awrite return value)."""
    resp = MagicMock()
    resp.error = error
    return resp


def _make_backend(
    *,
    write_error=None,
    exec_output="42\n",
    exec_exit_code=0,
    cleanup_output="",
):
    """Build a mock OpenSandboxBackend with async awrite/aexecute methods."""
    backend = MagicMock()
    backend.awrite = AsyncMock(return_value=_make_write_response(error=write_error))

    exec_results = [
        ExecuteResponse(output=exec_output, exit_code=exec_exit_code),
        ExecuteResponse(output=cleanup_output, exit_code=0),  # rm cleanup call
    ]
    backend.aexecute = AsyncMock(side_effect=exec_results)
    return backend


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreatePythonExecutionTool:
    def test_returns_langchain_tool(self):
        from langchain_core.tools import BaseTool

        from muffin_agent.sandbox.tool import create_python_execution_tool

        backend = _make_backend()
        tool = create_python_execution_tool(backend)
        assert isinstance(tool, BaseTool)

    def test_tool_is_named_execute_python(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        tool = create_python_execution_tool(_make_backend())
        assert tool.name == "execute_python"

    @pytest.mark.asyncio
    async def test_successful_execution_returns_output(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        backend = _make_backend(exec_output="hello world\n", exec_exit_code=0)
        tool = create_python_execution_tool(backend)

        result = await tool.ainvoke({"code": "print('hello world')"})

        assert result == "hello world\n"

    @pytest.mark.asyncio
    async def test_write_failure_returns_error_message(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        backend = _make_backend(write_error="permission_denied")
        tool = create_python_execution_tool(backend)

        result = await tool.ainvoke({"code": "print(1)"})

        assert "Failed to write" in result
        backend.aexecute.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_returns_error_with_output(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        err_output = "Traceback (most recent call last):\n  ...\nNameError: name 'x'\n"
        backend = _make_backend(exec_output=err_output, exec_exit_code=1)
        tool = create_python_execution_tool(backend)

        result = await tool.ainvoke({"code": "print(x)"})

        assert "Execution failed" in result
        assert "exit 1" in result
        assert "NameError" in result

    @pytest.mark.asyncio
    async def test_no_output_returns_placeholder(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        backend = _make_backend(exec_output="", exec_exit_code=0)
        tool = create_python_execution_tool(backend)

        result = await tool.ainvoke({"code": "x = 1 + 1"})

        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_cleanup_runs_after_execution(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        backend = _make_backend(exec_output="ok\n", exec_exit_code=0)
        tool = create_python_execution_tool(backend)

        await tool.ainvoke({"code": "print('ok')"})

        assert backend.aexecute.call_count == 2
        cleanup_cmd = backend.aexecute.call_args_list[1].args[0]
        assert cleanup_cmd.startswith("rm -f /tmp/muffin_exec_")

    @pytest.mark.asyncio
    async def test_uses_unique_file_path_per_call(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        # Two separate backends to get independent call lists
        backend1 = _make_backend(exec_output="a\n")
        backend2 = _make_backend(exec_output="b\n")
        tool1 = create_python_execution_tool(backend1)
        tool2 = create_python_execution_tool(backend2)

        await tool1.ainvoke({"code": "print('a')"})
        await tool2.ainvoke({"code": "print('b')"})

        path1 = backend1.aexecute.call_args_list[0].args[0]
        path2 = backend2.aexecute.call_args_list[0].args[0]
        assert path1 != path2
