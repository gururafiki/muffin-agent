"""Shared Pydantic models used across investment stage agents."""

from pydantic import BaseModel


class DataSource(BaseModel):
    """Record of data collected from one subagent."""

    subagent: str
    data_retrieved: str
    period: str
