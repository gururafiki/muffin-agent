"""Persona council (ai-hedge-fund port) — verdict ensemble + paper-trading + backtester.

Consolidated package: 13 investor personas (``personas/``) + 6 specialists
(``specialists/``) feeding an LLM-mediated judge form the council
(``council_graph``); a deterministic paper-trading pipeline (``portfolio/``)
and a walk-forward backtester (``backtesting/``) consume it. Deterministic
scoring lives in the package-local ``tools/``.

Public exports cover the council surface; import ``portfolio`` / ``backtesting``
submodules directly for the paper-trading / backtesting layers.
"""

from __future__ import annotations

from .council_graph import CouncilState, build_council_graph
from .judge import (
    CouncilJudgeInputState,
    CouncilJudgeOutputState,
    CouncilSynthesisOutput,
    council_judge_node,
)
from .personas import (
    build_aswath_damodaran_agent,
    build_ben_graham_agent,
    build_bill_ackman_agent,
    build_cathie_wood_agent,
    build_charlie_munger_agent,
    build_michael_burry_agent,
    build_mohnish_pabrai_agent,
    build_nassim_taleb_agent,
    build_peter_lynch_agent,
    build_phil_fisher_agent,
    build_rakesh_jhunjhunwala_agent,
    build_stanley_druckenmiller_agent,
    build_warren_buffett_agent,
)
from .schemas import AnalystSignal, InvestmentSignal

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
