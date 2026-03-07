"""Rich-based streaming output for agent messages."""

import json
from typing import Any

from langchain_core.messages import AIMessageChunk, ToolMessage
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


class StreamPrinter:
    """Print streaming agent messages to the terminal with Rich formatting."""

    def __init__(self) -> None:
        """Initialize console and streaming state."""
        self._console = Console()
        self._in_agent_text = False

    def print_chunk(self, chunk: Any) -> None:
        """Dispatch a streamed chunk to the appropriate handler."""
        if isinstance(chunk, AIMessageChunk):
            self._print_ai_chunk(chunk)
        elif isinstance(chunk, ToolMessage):
            self._end_agent_text()
            self._print_tool_result(chunk)

    def finish(self) -> None:
        """Flush any remaining agent text and print a trailing newline."""
        self._end_agent_text()
        self._console.print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _end_agent_text(self) -> None:
        if self._in_agent_text:
            self._console.print()
            self._in_agent_text = False

    def _print_ai_chunk(self, chunk: AIMessageChunk) -> None:
        """Handle AI message chunks: tool calls or text tokens."""
        if chunk.tool_call_chunks:
            self._end_agent_text()
            for tc in chunk.tool_call_chunks:
                if tc.get("name"):
                    self._console.print(f"\n[bold yellow]>> Tool call: {tc['name']}[/]")
                if tc.get("args"):
                    self._console.print(tc["args"], end="", highlight=False)
        elif chunk.content:
            if not self._in_agent_text:
                self._console.print("\n[bold cyan]\\[Agent][/]")
                self._in_agent_text = True
            self._console.print(chunk.content, end="", highlight=False)

    def _print_tool_result(self, msg: ToolMessage) -> None:
        """Print a tool result in a Rich panel with JSON syntax highlighting."""
        content = self._extract_text(msg.content)
        style = "red" if msg.status == "error" else "magenta"
        label = f"Tool: {msg.name}" if msg.name else "Tool"

        try:
            parsed = json.loads(content)
            formatted = json.dumps(parsed, indent=2)
            if len(formatted) > 3000:
                formatted = formatted[:3000] + "\n... (truncated)"
            body = Syntax(formatted, "json", theme="monokai", word_wrap=True)
        except (json.JSONDecodeError, TypeError):
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            body = content

        self._console.print(Panel(body, title=label, border_style=style))

    @staticmethod
    def _extract_text(content: str | list) -> str:
        """Extract text from MCP content blocks or return plain string."""
        if isinstance(content, str):
            return content
        # MCP format: [{"type": "text", "text": "..."}, ...]
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts) if parts else str(content)
