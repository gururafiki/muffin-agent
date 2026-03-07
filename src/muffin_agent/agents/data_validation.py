"""Data validation agent.

Pure reasoning agent that validates collected data against a criterion,
checking sufficiency, relevance, temporal validity, and consistency.
Produces structured confidence/relevance scores.
"""

from langchain.agents import create_agent

from ..config import Configuration
from ..prompts import render_template


async def create_data_validation_agent(config: Configuration):
    """Build the data validation agent.

    Create a tool-less reasoning agent that evaluates collected data quality
    against a given criterion and returns structured validation scores.
    """
    prompt = render_template("data_validation.jinja")
    llm = config.get_llm()
    return create_agent(model=llm, system_prompt=prompt)
