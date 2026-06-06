"""Persona council — investor-style verdict ensemble.

v4 architecture: each persona is a compiled :class:`CompiledStateGraph`
subgraph (``collect_data`` ReAct → ``compute_evidence`` Python →
``render_verdict`` LLM call).  Personas are imported by name from the
council graph and CLI — there is no central ``PERSONA_REGISTRY``.

Public exports:

* :func:`build_council_graph` — async builder for the full 13-persona
  council + LLM-mediated judge synthesis.
* :class:`CouncilState`, :class:`CouncilSynthesisOutput` —
  schemas the council reads / writes.
* :class:`AnalystSignal`, :data:`InvestmentSignal` — universal signal
  contract every persona conforms to.
* ``build_<persona>_agent`` — per-persona async factories.
"""

from __future__ import annotations

from .aswath_damodaran import build_aswath_damodaran_agent
from .ben_graham import build_ben_graham_agent
from .bill_ackman import build_bill_ackman_agent
from .cathie_wood import build_cathie_wood_agent
from .charlie_munger import build_charlie_munger_agent
from .council_graph import CouncilState, build_council_graph
from .judge import (
    CouncilJudgeInputState,
    CouncilJudgeOutputState,
    CouncilSynthesisOutput,
    council_judge_node,
)
from .michael_burry import build_michael_burry_agent
from .mohnish_pabrai import build_mohnish_pabrai_agent
from .nassim_taleb import build_nassim_taleb_agent
from .peter_lynch import build_peter_lynch_agent
from .phil_fisher import build_phil_fisher_agent
from .rakesh_jhunjhunwala import build_rakesh_jhunjhunwala_agent
from .schemas import AnalystSignal, InvestmentSignal
from .stanley_druckenmiller import build_stanley_druckenmiller_agent
from .warren_buffett import build_warren_buffett_agent

__all__ = [
    "AnalystSignal",
    "CouncilJudgeInputState",
    "CouncilJudgeOutputState",
    "CouncilState",
    "CouncilSynthesisOutput",
    "InvestmentSignal",
    "build_aswath_damodaran_agent",
    "build_ben_graham_agent",
    "build_bill_ackman_agent",
    "build_cathie_wood_agent",
    "build_charlie_munger_agent",
    "build_council_graph",
    "build_michael_burry_agent",
    "build_mohnish_pabrai_agent",
    "build_nassim_taleb_agent",
    "build_peter_lynch_agent",
    "build_phil_fisher_agent",
    "build_rakesh_jhunjhunwala_agent",
    "build_stanley_druckenmiller_agent",
    "build_warren_buffett_agent",
    "council_judge_node",
]
