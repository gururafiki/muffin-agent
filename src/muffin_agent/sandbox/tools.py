"""Store ↔ sandbox bridge tools for muffin agents.

- ``write_store_data_to_sandbox`` — materialize store data as sandbox files.
- ``read_sandbox_file_to_store`` — persist sandbox file content to the store.

Both route through ``AccessControlledStore``, so a namespace an agent may not
touch stays untouchable from inside the sandbox. That permission model is what
keeps these two here rather than in ``langchain-opensandbox`` alongside
``execute_python``.

The sandbox is discovered by ``thread_id`` metadata via the OpenSandbox API.
If no running sandbox exists, a new one is created automatically.
"""

import json

from langchain_core.tools import tool
from langchain_opensandbox.factory import aget_sandbox
from langgraph.prebuilt import ToolRuntime

from ..middlewares.store_access.store import AccessControlledStore


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
