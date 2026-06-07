"""Investor-persona subgraphs.

Each persona is a compiled ``StateGraph`` subgraph
(``collect_data`` ReAct → ``compute_evidence`` Python → ``render_verdict`` LLM)
emitting the shared ``AnalystSignal`` contract. Personas are imported by name
from the council graph and CLI — there is no central registry.
"""

from __future__ import annotations

from .aswath_damodaran import build_aswath_damodaran_agent
from .ben_graham import build_ben_graham_agent
from .bill_ackman import build_bill_ackman_agent
from .cathie_wood import build_cathie_wood_agent
from .charlie_munger import build_charlie_munger_agent
from .michael_burry import build_michael_burry_agent
from .mohnish_pabrai import build_mohnish_pabrai_agent
from .nassim_taleb import build_nassim_taleb_agent
from .peter_lynch import build_peter_lynch_agent
from .phil_fisher import build_phil_fisher_agent
from .rakesh_jhunjhunwala import build_rakesh_jhunjhunwala_agent
from .stanley_druckenmiller import build_stanley_druckenmiller_agent
from .warren_buffett import build_warren_buffett_agent

__all__ = [
    "build_aswath_damodaran_agent",
    "build_ben_graham_agent",
    "build_bill_ackman_agent",
    "build_cathie_wood_agent",
    "build_charlie_munger_agent",
    "build_michael_burry_agent",
    "build_mohnish_pabrai_agent",
    "build_nassim_taleb_agent",
    "build_peter_lynch_agent",
    "build_phil_fisher_agent",
    "build_rakesh_jhunjhunwala_agent",
    "build_stanley_druckenmiller_agent",
    "build_warren_buffett_agent",
]
