"""Unit tests for create_python_execution_tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest


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
    """Build a mock async Sandbox with files and commands services."""
    from opensandbox.models.execd import CommandStatus

    sandbox = MagicMock()

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCreatePythonExecutionTool:
    def test_returns_langchain_tool(self):
        from langchain_core.tools import BaseTool

        from muffin_agent.sandbox.tool import create_python_execution_tool

        sandbox = _make_sandbox()
        tool = create_python_execution_tool(sandbox)
        assert isinstance(tool, BaseTool)

    def test_tool_is_named_execute_python(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        tool = create_python_execution_tool(_make_sandbox())
        assert tool.name == "execute_python"

    @pytest.mark.asyncio
    async def test_successful_execution_returns_output(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        sandbox = _make_sandbox(exec_output="hello world\n", exec_exit_code=0)
        tool = create_python_execution_tool(sandbox)

        result = await tool.ainvoke({"code": "print('hello world')"})

        assert result == "hello world\n"

    @pytest.mark.asyncio
    async def test_write_failure_returns_error_message(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        sandbox = _make_sandbox(write_raises=True)
        tool = create_python_execution_tool(sandbox)

        result = await tool.ainvoke({"code": "print(1)"})

        assert "Failed to write" in result
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_returns_error_with_output(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        err_output = "NameError: name 'x' is not defined\n"
        sandbox = _make_sandbox(exec_output=err_output, exec_exit_code=1)
        tool = create_python_execution_tool(sandbox)

        result = await tool.ainvoke({"code": "print(x)"})

        assert "Execution failed" in result
        assert "exit 1" in result
        assert "NameError" in result

    @pytest.mark.asyncio
    async def test_no_output_returns_placeholder(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        sandbox = _make_sandbox(exec_output="", exec_exit_code=0)
        tool = create_python_execution_tool(sandbox)

        result = await tool.ainvoke({"code": "x = 1 + 1"})

        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_cleanup_runs_after_execution(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        sandbox = _make_sandbox(exec_output="ok\n", exec_exit_code=0)
        tool = create_python_execution_tool(sandbox)

        await tool.ainvoke({"code": "print('ok')"})

        assert sandbox.commands.run.call_count == 2
        cleanup_cmd = sandbox.commands.run.call_args_list[1].args[0]
        assert cleanup_cmd.startswith("rm -f /tmp/muffin_exec_")

    @pytest.mark.asyncio
    async def test_writes_code_to_unique_temp_file(self):
        from muffin_agent.sandbox.tool import create_python_execution_tool

        sandbox1 = _make_sandbox(exec_output="a\n")
        sandbox2 = _make_sandbox(exec_output="b\n")
        tool1 = create_python_execution_tool(sandbox1)
        tool2 = create_python_execution_tool(sandbox2)

        await tool1.ainvoke({"code": "print('a')"})
        await tool2.ainvoke({"code": "print('b')"})

        path1 = sandbox1.commands.run.call_args_list[0].args[0]
        path2 = sandbox2.commands.run.call_args_list[0].args[0]
        assert path1 != path2
        assert path1.startswith("python3 /tmp/muffin_exec_")
