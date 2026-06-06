"""Persona council — investor-style verdict ensemble.

Public exports are populated incrementally as the council is built out per
the porting plan.  Phase 1 ships the foundational schemas, the shared
data-collection step, and the registry scaffolding.  Phase 2 adds 13
persona node functions (each self-registers into :data:`PERSONA_REGISTRY`).
Phase 2.4 adds ``build_council_graph`` and the standalone single-persona
graph.
"""

from __future__ import annotations

# ── Persona registrations ─────────────────────────────────────────────────────
# Each persona module self-registers into PERSONA_REGISTRY on import via
# register_persona().  Listed here so importing the personas package
# populates the registry once.  Add new personas to this block.
from . import aswath_damodaran as _aswath_damodaran  # noqa: F401  (side-effect)
from . import ben_graham as _ben_graham  # noqa: F401
from . import bill_ackman as _bill_ackman  # noqa: F401
from . import cathie_wood as _cathie_wood  # noqa: F401
from . import charlie_munger as _charlie_munger  # noqa: F401
from . import michael_burry as _michael_burry  # noqa: F401
from . import mohnish_pabrai as _mohnish_pabrai  # noqa: F401
from . import nassim_taleb as _nassim_taleb  # noqa: F401
from . import peter_lynch as _peter_lynch  # noqa: F401
from . import phil_fisher as _phil_fisher  # noqa: F401
from . import rakesh_jhunjhunwala as _rakesh_jhunjhunwala  # noqa: F401
from . import stanley_druckenmiller as _stanley_druckenmiller  # noqa: F401
from . import warren_buffett as _warren_buffett  # noqa: F401
from ._base import (
    PERSONA_REGISTRY,
    PersonaInputState,
    PersonaNode,
    PersonaOutputState,
    PersonaSpec,
    register_persona,
)
from .council_graph import CouncilState, build_council_graph
from .data import (
    LINE_ITEM_FIELDS,
    InsiderTrade,
    MarketCapHistoryPoint,
    NewsArticle,
    PersonaDataBundle,
    PriceBar,
)
from .data_collection import (
    PersonaDataCollectionInputState,
    PersonaDataCollectionOutputState,
    create_persona_data_collection_agent,
    persona_data_collection_node,
)
from .judge import (
    CouncilJudgeInputState,
    CouncilJudgeOutputState,
    CouncilSynthesisOutput,
    council_judge_node,
)
from .schemas import AnalystSignal, InvestmentSignal, ScoreDetail
from .single_persona_graph import SinglePersonaState, build_single_persona_graph

__all__ = [
    "LINE_ITEM_FIELDS",
    "PERSONA_REGISTRY",
    "AnalystSignal",
    "CouncilJudgeInputState",
    "CouncilJudgeOutputState",
    "CouncilState",
    "CouncilSynthesisOutput",
    "InsiderTrade",
    "InvestmentSignal",
    "MarketCapHistoryPoint",
    "NewsArticle",
    "PersonaDataBundle",
    "PersonaDataCollectionInputState",
    "PersonaDataCollectionOutputState",
    "PersonaInputState",
    "PersonaNode",
    "PersonaOutputState",
    "PersonaSpec",
    "PriceBar",
    "ScoreDetail",
    "SinglePersonaState",
    "build_council_graph",
    "build_single_persona_graph",
    "council_judge_node",
    "create_persona_data_collection_agent",
    "persona_data_collection_node",
    "register_persona",
]
