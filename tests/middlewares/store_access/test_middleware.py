"""Tests for StoreAccessMiddleware."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import ToolMessage

from muffin_agent.middlewares.store_access.middleware import StoreAccessMiddleware


@pytest.mark.unit
class TestStoreAccessMiddleware:
    def test_exposes_five_store_tools(self):
        mw = StoreAccessMiddleware()
        assert len(mw.tools) == 5
        names = {t.name for t in mw.tools}
        assert names == {
            "store_get",
            "store_put",
            "store_delete",
            "store_search",
            "store_list_namespaces",
        }

    @pytest.mark.asyncio
    async def test_pass_through(self):
        mw = StoreAccessMiddleware()
        expected = ToolMessage(content="ok", tool_call_id="tc-1")
        handler = AsyncMock(return_value=expected)
        request = MagicMock()

        result = await mw.awrap_tool_call(request, handler)

        handler.assert_awaited_once_with(request)
        assert result is expected
