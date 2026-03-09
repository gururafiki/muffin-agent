"""Python execution tool for muffin agents.

Provides a LangChain tool that executes arbitrary Python code in an isolated
OpenSandbox container. Designed for financial calculations, dataframe analysis,
and technical indicator computation using OpenBB methods and TA-Lib.

A fresh async ``Sandbox`` is created for each tool invocation and closed
afterwards. This keeps the implementation stateless — no registry or
persistent connection is required — at the cost of one container-creation
roundtrip per call. Each invocation's container is independent, which is
appropriate because ``execute_python`` calls are self-contained: they write
a temp file, run it, and discard the result.
"""

import uuid
from typing import TYPE_CHECKING

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool

if TYPE_CHECKING:
    from muffin_agent.config import Configuration


def create_python_execution_tool(config: "Configuration") -> BaseTool:
    """Return a LangChain tool that executes Python code in a fresh sandbox.

    Uses the native async OpenSandbox SDK — no sync bridging, no thread pool.
    The tool creates an isolated container, writes code to a temp file, runs
    it with ``python3``, captures all output, and closes the container when
    done.

    Args:
        config: Muffin agent configuration with opensandbox_* fields.

    Returns:
        Async LangChain ``execute_python`` tool bound to *config*.

    Example usage in an agent::

        tool = create_python_execution_tool(config)
        tools = await get_tools(config, MCP_TOOLS, custom_tools=[tool])
    """

    @lc_tool
    async def execute_python(code: str) -> str:
        """Execute Python code in a secure isolated sandbox.

        Use this tool for computations that require Python code execution, such as:
        - Financial formula calculations (DCF valuation, WACC, Gordon Growth Model)
        - Dataframe manipulation and analysis of financial statements or price history
        - Technical indicator computation (moving averages, RSI, Bollinger Bands, etc.)
        - Statistical analysis (correlation, regression, Monte Carlo simulation)
        - Custom OpenBB data transformations

        The sandbox has Python 3 with standard libraries available.
        Use print() to produce output — return values alone are not captured.

        Examples::

            # DCF calculation
            code = '''
            import math
            wacc = 0.10
            fcfs = [100, 110, 121, 133, 146]
            terminal_value = fcfs[-1] * 1.03 / (wacc - 0.03)
            dcf = sum(f / (1 + wacc) ** (i + 1) for i, f in enumerate(fcfs))
            dcf += terminal_value / (1 + wacc) ** len(fcfs)
            print(f"DCF value: {dcf:.2f}")
            '''

        Args:
            code: Python source code to execute. Multi-line strings supported.

        Returns:
            Captured stdout/stderr from the execution, or an error message if
            the process exits with a non-zero code.
        """
        from muffin_agent.sandbox.backend import create_opensandbox_sandbox

        path = f"/tmp/muffin_exec_{uuid.uuid4().hex}.py"

        sandbox = await create_opensandbox_sandbox(config)
        try:
            try:
                await sandbox.files.write_file(path, code)
            except Exception as exc:
                return f"Failed to write code to sandbox: {exc}"

            execution = await sandbox.commands.run(f"python3 {path}")
            await sandbox.commands.run(f"rm -f {path}")

            stdout = "".join(m.text for m in execution.logs.stdout)
            stderr = "".join(m.text for m in execution.logs.stderr)
            output = stdout + stderr

            exit_code: int | None = None
            if execution.id:
                try:
                    status = await sandbox.commands.get_command_status(execution.id)
                    exit_code = status.exit_code
                except Exception:
                    pass
            if execution.error and exit_code is None:
                exit_code = 1

            if exit_code is not None and exit_code != 0:
                return f"Execution failed (exit {exit_code}):\n{output}"

            return output or "(no output)"
        finally:
            await sandbox.close()

    return execute_python
