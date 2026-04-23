"""Stock evaluation agent.

Deep agent that orchestrates data collection subagents, validates collected
data, and produces a scored stock assessment with reasoning.
"""

from langchain_core.runnables import RunnableConfig

from ..model_config import ModelConfiguration
from ..utils.agent_builder import MuffinAgentBuilder
from .subagents import build_analysis_subagents


async def create_stock_evaluation_agent(config: RunnableConfig):
    """Build the stock evaluation deep agent.

    Delegates data collection to specialized subagents, then validates,
    analyzes, and scores the stock.  The muffin composite backend provides
    ``/scratch/``, ``/memories/`` and a per-thread sandbox for Python
    computations (DCF, WACC, technical indicators).
    """
    subagents = await build_analysis_subagents(config)
    llm = ModelConfiguration.from_runnable_config(config).get_llm()

    return (
        MuffinAgentBuilder(llm, name="stock_evaluation")
        .with_system_prompt_template("stock_evaluation.jinja")
        .with_sandbox()
        .with_short_term_memory()
        .with_persistent_memory()
        .with_subagents(subagents)
        .build_deep_agent()
    )
