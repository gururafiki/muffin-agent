"""Python execution tool for muffin agents.

Provides a LangChain tool that executes arbitrary Python code in an isolated
OpenSandbox container. Designed for financial calculations, dataframe analysis,
and technical indicator computation using OpenBB methods and TA-Lib.
"""

import uuid

from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool

from .backend import OpenSandboxBackend


def create_python_execution_tool(backend: OpenSandboxBackend) -> BaseTool:
    """Return a LangChain tool that executes Python code in the sandbox.

    The returned tool writes code to an isolated temp file inside the sandbox
    container, executes it with ``python3``, captures all output, and cleans up.
    Execution is async-safe: the tool is an async function so it does not block
    the event loop.

    Args:
        backend: Connected OpenSandboxBackend to run code in.

    Returns:
        Async LangChain ``execute_python`` tool bound to *backend*.

    Example usage in an agent::

        backend = await create_opensandbox_backend(config)
        tool = create_python_execution_tool(backend)
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
        path = f"/tmp/muffin_exec_{uuid.uuid4().hex}.py"

        write_result = await backend.awrite(path, code)
        if write_result.error:
            return f"Failed to write code to sandbox: {write_result.error}"

        result = await backend.aexecute(f"python3 {path}")
        await backend.aexecute(f"rm -f {path}")

        if result.exit_code is not None and result.exit_code != 0:
            return f"Execution failed (exit {result.exit_code}):\n{result.output}"

        return result.output or "(no output)"

    return execute_python
