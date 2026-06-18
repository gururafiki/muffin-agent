"""Package-local deterministic scoring helpers for the persona council.

Pure-Python (math / statistics / typing only) scoring used by the personas
(``scoring_helpers``) and specialists (``technicals``, ``sentiment``,
``fundamentals``, ``growth``, ``valuation_signal``). Distinct from the
top-level :mod:`muffin_agent.tools` package — these are council-internal and
never cross an LLM boundary. Import the submodules directly, e.g.
``from ..tools.scoring_helpers import compute_buffett_3stage_dcf``.
"""

from __future__ import annotations
