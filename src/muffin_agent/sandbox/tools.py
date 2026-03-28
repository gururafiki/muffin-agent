"""Sandbox-backed LangChain tools for muffin agents.

Provides tools that execute in an isolated OpenSandbox container:

- ``execute_python`` — run arbitrary Python code (financial calcs, dataframes).
- ``write_store_data_to_sandbox`` — materialize store data as sandbox files.
- ``read_sandbox_file_to_store`` — persist sandbox file content to the store.

The sandbox is discovered by ``thread_id`` metadata via the OpenSandbox API.
If no running sandbox exists, a new one is created automatically.
"""

import json
import uuid

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from ..middlewares.store_access.store import AccessControlledStore
from .factory import aget_sandbox


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


# ── Store ↔ Sandbox bridge ───────────────────────────────────────────────��───


@tool(parse_docstring=True)
async def write_store_data_to_sandbox(
    namespace: str,
    key: str,
    runtime: ToolRuntime,
    file_path: str | None = None,
) -> str:
    """Write a store entry to a sandbox file.

    Reads any entry from the store and writes its value as JSON to the
    sandbox filesystem so it can be loaded by ``execute_python``.

    Args:
        namespace: Dot-separated namespace (e.g. ``"computed.dcf_model"``).
        key: Entry key within the namespace.
        file_path: Optional custom sandbox path. Defaults to
            ``/data/store/{namespace}/{key}.json``.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Confirmation message with the file path, or an error message.
    """
    try:
        store = AccessControlledStore.from_runtime(runtime)
        item = await store.aget(namespace, key)
    except ValueError as exc:
        return f"Error: {exc}"

    if item is None:
        return f"Error: no entry found at namespace={namespace!r}, key={key!r}"

    content = json.dumps(item.value)
    target = file_path or f"/data/store/{namespace.replace('.', '/')}/{key}.json"

    sandbox = await aget_sandbox(runtime)
    async with sandbox:
        try:
            await sandbox.files.write_file(target, content)
        except Exception as exc:
            return f"Error writing to sandbox: {exc}"

    return (
        f"Data written to {target} ({len(content)} chars). "
        f"Load it in execute_python with: json.load(open('{target}'))"
    )


@tool(parse_docstring=True)
async def read_sandbox_file_to_store(
    file_path: str,
    namespace: str,
    key: str,
    runtime: ToolRuntime,
) -> str:
    """Read a sandbox file and store its content in the store.

    Reads a file from the sandbox filesystem and puts its content into
    the LangGraph store under the given namespace and key.  Other agents
    sharing the same store can then access the data.

    Args:
        file_path: Absolute path to the file in the sandbox.
        namespace: Dot-separated target namespace (e.g. ``"computed.dcf"``).
        key: Entry key within the namespace.
        runtime: Injected by LangGraph ToolNode.

    Returns:
        Confirmation message with content size, or an error message.
    """
    sandbox = await aget_sandbox(runtime)
    async with sandbox:
        try:
            content_bytes = await sandbox.files.read_bytes(file_path)
            content = content_bytes.decode("utf-8")
        except Exception as exc:
            return f"Error reading from sandbox: {exc}"

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            parsed = {"content": content}
    except (json.JSONDecodeError, TypeError):
        parsed = {"content": content}

    try:
        store = AccessControlledStore.from_runtime(runtime)
        await store.aput(namespace, key, parsed)
    except ValueError as exc:
        return f"Error: {exc}"

    return (
        f"Stored {len(content)} chars from {file_path} "
        f"at namespace={namespace!r}, key={key!r}"
    )
