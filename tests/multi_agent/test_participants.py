"""Tests for the LLM-driven participants.

Covers ``LLMParticipant`` and ``LLMMessageParticipant``.

``AgentParticipant`` is covered by ``test_conference.py`` since it's wired
into the conference graph rather than invoked through the ``Participant``
Protocol directly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from muffin_agent.multi_agent import participants as participants_mod
from muffin_agent.multi_agent.participants import (
    LLMMessageParticipant,
    LLMParticipant,
)

from .conftest import ai, fake_model_config


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMParticipant:
    async def test_returns_stripped_content_from_response(self):
        cfg, _ = fake_model_config(ai("  aggressive reply  "))
        with patch.object(
            participants_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            participant = LLMParticipant(
                name="aggressive",
                system_prompt_template="multi_agent/_transcript.jinja",
            )
            content = await participant.speak({}, {})

        assert content == "aggressive reply"

    async def test_prompt_includes_rendered_transcript_text(self):
        cfg, fake_llm = fake_model_config(ai("ok"))
        with patch.object(
            participants_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            participant = LLMParticipant(
                name="aggressive",
                system_prompt_template="multi_agent/_transcript.jinja",
            )
            state = {
                "messages": [
                    AIMessage(content="be careful", name="conservative"),
                    AIMessage(content="scale in", name="neutral"),
                ],
            }
            await participant.speak(state, {})

        system_msg, human_msg = fake_llm.invocations[0]
        assert isinstance(system_msg, SystemMessage)
        assert "conservative: be careful" in system_msg.content
        assert "neutral: scale in" in system_msg.content
        assert isinstance(human_msg, HumanMessage)
        assert human_msg.content == "Take your turn now."

    async def test_empty_messages_renders_opening_placeholder(self):
        cfg, fake_llm = fake_model_config(ai("opener"))
        with patch.object(
            participants_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            participant = LLMParticipant(
                name="aggressive",
                system_prompt_template="multi_agent/_transcript.jinja",
            )
            await participant.speak({"messages": []}, {})

        system_msg = fake_llm.invocations[0][0]
        assert "discussion has not yet begun" in system_msg.content

    async def test_custom_user_prompt(self):
        cfg, fake_llm = fake_model_config(ai("done"))
        with patch.object(
            participants_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            participant = LLMParticipant(
                name="alice",
                system_prompt_template="multi_agent/_transcript.jinja",
                user_prompt="Speak now please.",
            )
            await participant.speak({}, {})

        human_msg = fake_llm.invocations[0][1]
        assert human_msg.content == "Speak now please."


@pytest.mark.unit
@pytest.mark.asyncio
class TestLLMMessageParticipant:
    async def test_materialises_prior_messages_as_message_thread(self):
        cfg, fake_llm = fake_model_config(ai("reply"))
        with patch.object(
            participants_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            participant = LLMMessageParticipant(
                name="alice",
                system_prompt_template="multi_agent/_transcript.jinja",
            )
            state = {
                "messages": [
                    AIMessage(content="first", name="bob"),
                    AIMessage(content="self-reply", name="alice"),
                    AIMessage(content="third", name="carol"),
                ],
            }
            await participant.speak(state, {})

        messages = fake_llm.invocations[0]
        # Expected: [SystemMessage, HumanMessage("[bob]: first"),
        #            AIMessage("self-reply"), HumanMessage("[carol]: third"),
        #            HumanMessage("Take your turn now.")]
        assert len(messages) == 5
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "[bob]: first"
        assert isinstance(messages[2], AIMessage)
        assert messages[2].content == "self-reply"
        assert isinstance(messages[3], HumanMessage)
        assert messages[3].content == "[carol]: third"
        assert isinstance(messages[4], HumanMessage)
        assert messages[4].content == "Take your turn now."

    async def test_empty_messages_just_system_and_user(self):
        cfg, fake_llm = fake_model_config(ai("opener"))
        with patch.object(
            participants_mod.ModelConfiguration,
            "from_runnable_config",
            return_value=cfg,
        ):
            participant = LLMMessageParticipant(
                name="alice",
                system_prompt_template="multi_agent/_transcript.jinja",
            )
            await participant.speak({}, {})

        messages = fake_llm.invocations[0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
