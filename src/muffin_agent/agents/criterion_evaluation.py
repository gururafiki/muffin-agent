"""Criterion evaluation agent.

Deep agent that evaluates a single investment criterion by orchestrating
data collection, validation, and scoring with reflection.
"""

from langchain_core.runnables import RunnableConfig

from ..model_config import ModelConfiguration
from ..utils.agent_builder import MuffinAgentBuilder
from .subagents import build_analysis_subagents


async def create_criterion_evaluation_agent(config: RunnableConfig):
    """Build the criterion evaluation deep agent.

    Create a deep agent that evaluates a single investment criterion by
    collecting targeted data, validating it, scoring the criterion, and
    reflecting on the evaluation quality.
    """
    subagents = await build_analysis_subagents(config)
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("orchestrator")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="criterion_evaluation")
        .with_system_prompt_template("criterion_evaluation.jinja")
        .with_fallback_models(*fallbacks)
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_deep_agent()
