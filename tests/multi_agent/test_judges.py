"""Tests for ``StructuredOutputJudge``."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from muffin_agent.multi_agent import judges as judges_mod
from muffin_agent.multi_agent.judges import StructuredOutputJudge

from .conftest import fake_model_config


class _Verdict(BaseModel):
    decision: str
    confidence: float


@pytest.mark.unit
@pytest.mark.asyncio
class TestStructuredOutputJudge:
    async def test_returns_model_dump_of_pydantic_response(self):
        cfg, _ = fake_model_config(_Verdict(decision="buy", confidence=0.8))
        with patch.object(
            judges_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            judge = StructuredOutputJudge(
                name="judge",
                system_prompt_template="multi_agent/_transcript.jinja",
                output_schema=_Verdict,
            )
            result = await judge.adjudicate({}, {})

        assert result == {"decision": "buy", "confidence": 0.8}

    async def test_prompt_includes_rendered_messages_text(self):
        cfg, fake_llm = fake_model_config(
            _Verdict(decision="hold", confidence=0.5)
        )
        with patch.object(
            judges_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            judge = StructuredOutputJudge(
                name="judge",
                system_prompt_template="multi_agent/_transcript.jinja",
                output_schema=_Verdict,
            )
            state = {
                "messages": [
                    AIMessage(content="argue", name="alice"),
                    AIMessage(content="counter", name="bob"),
                ]
            }
            await judge.adjudicate(state, {})

        system_msg, human_msg = fake_llm.invocations[0]
        assert isinstance(system_msg, SystemMessage)
        assert "alice: argue" in system_msg.content
        assert "bob: counter" in system_msg.content
        assert isinstance(human_msg, HumanMessage)
        assert human_msg.content == "Render your verdict now."
