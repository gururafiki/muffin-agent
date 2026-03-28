"""Data validation agent.

Pure reasoning agent that validates collected data against a criterion,
checking sufficiency, relevance, temporal validity, and consistency.
Produces structured confidence/relevance scores.
"""

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig

from ..model_config import ModelConfiguration
from ..prompts import render_template


async def create_data_validation_agent(config: RunnableConfig):
    """Build the data validation agent.

    Create a tool-less reasoning agent that evaluates collected data quality
    against a given criterion and returns structured validation scores.
    """
    prompt = render_template("data_validation.jinja")
    model_config = ModelConfiguration.from_runnable_config(config)
    llm = model_config.get_llm()
    return create_agent(model=llm, system_prompt=prompt)
