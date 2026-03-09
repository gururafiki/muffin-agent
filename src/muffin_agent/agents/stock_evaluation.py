"""Stock evaluation agent.

Deep agent that orchestrates data collection subagents, validates collected
data, and produces a scored stock assessment with reasoning.
"""

from deepagents import create_deep_agent

from ..config import Configuration
from ..prompts import render_template
from ..sandbox import create_opensandbox_backend
from .subagents import build_analysis_subagents


async def create_stock_evaluation_agent(config: Configuration):
    """Build the stock evaluation deep agent.

    Create a deep agent that delegates data collection to specialized
    subagents, then validates, analyzes, and scores the stock. An isolated
    OpenSandbox container is attached as the backend so the agent can execute
    Python for financial computations (DCF, WACC, technical indicators).
    """
    subagents = await build_analysis_subagents(config)
    prompt = render_template("stock_evaluation.jinja")
    llm = config.get_llm()
    backend = await create_opensandbox_backend(config)

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
        backend=backend,
    )
