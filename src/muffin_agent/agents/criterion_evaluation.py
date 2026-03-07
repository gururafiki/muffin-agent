"""Criterion evaluation agent.

Deep agent that evaluates a single investment criterion by orchestrating
data collection, validation, and scoring with reflection.
"""

from deepagents import create_deep_agent

from ..config import Configuration
from ..prompts import render_template
from .subagents import build_analysis_subagents


async def create_criterion_evaluation_agent(config: Configuration):
    """Build the criterion evaluation deep agent.

    Create a deep agent that evaluates a single investment criterion by
    collecting targeted data, validating it, scoring the criterion, and
    reflecting on the evaluation quality.
    """
    subagents = await build_analysis_subagents(config)
    prompt = render_template("criterion_evaluation.jinja")
    llm = config.get_llm()

    return create_deep_agent(
        model=llm,
        system_prompt=prompt,
        subagents=subagents,
    )
