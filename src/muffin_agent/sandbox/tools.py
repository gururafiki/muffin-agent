"""Python execution tool for muffin agents.

Provides a LangChain tool that executes arbitrary Python code in an isolated
OpenSandbox container. Designed for financial calculations, dataframe analysis,
and technical indicator computation using OpenBB methods and TA-Lib.

The sandbox is discovered by ``thread_id`` metadata via the OpenSandbox API.
If no running sandbox exists, a new one is created automatically.
"""

import logging
import uuid

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from .factory import aget_sandbox

_log = logging.getLogger(__name__)


@tool
async def execute_python(code: str, runtime: ToolRuntime) -> str:
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
        runtime: Injected by LangGraph ToolNode. Provides config and state.

    Returns:
        Captured stdout/stderr from the execution, or an error message if
        the process exits with a non-zero code.
    """
    sandbox = await aget_sandbox(runtime)
    path = f"/tmp/muffin_exec_{uuid.uuid4().hex}.py"

    async with sandbox:
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
                status = await sandbox.commands.get_command_status(
                    execution.id,
                )
                exit_code = status.exit_code
            except Exception:
                pass
        if execution.error and exit_code is None:
            exit_code = 1

        if exit_code is not None and exit_code != 0:
            return f"Execution failed (exit {exit_code}):\n{output}"

        return output or "(no output)"
