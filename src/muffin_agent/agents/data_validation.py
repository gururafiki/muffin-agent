"""Data validation agent.

Pure reasoning agent that validates collected data against a criterion,
checking sufficiency, relevance, temporal validity, and consistency.
Produces structured confidence/relevance scores.
"""

from langchain_core.runnables import RunnableConfig

from ..model_config import ModelConfiguration
from ..utils.agent_builder import MuffinAgentBuilder


async def create_data_validation_agent(config: RunnableConfig):
    """Build the data validation agent.

    Create a tool-less reasoning agent that evaluates collected data quality
    against a given criterion and returns structured validation scores.
    Goes through ``MuffinAgentBuilder`` so it inherits the universal
    middleware stack (model retry, fallback models when configured, tool
    knowledge with deterministic-fallback lessons, etc.).
    """
    configuration = ModelConfiguration.from_runnable_config(config)
    primary, *fallbacks = configuration.get_llm_for_role("reasoner")
    summariser = configuration.get_summariser()

    builder = (
        MuffinAgentBuilder(primary, name="data_validation")
        .with_system_prompt_template("data_validation.jinja")
        .with_fallback_models(*fallbacks)
    )
    if summariser is not None:
        builder = builder.with_tool_knowledge(summariser)
    return builder.build_react_agent()
