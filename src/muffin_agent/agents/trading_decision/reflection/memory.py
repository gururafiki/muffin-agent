"""Per-user persistent storage for trading decisions, outcomes, and reflections.

Wraps :class:`langgraph.store.base.BaseStore` with a typed CRUD API for the
trading-decision pipeline. Decisions live under namespace
``("memories", user_id, "decisions")`` with key ``f"{TICKER}:{YYYY-MM-DD}"``.

Lifecycle:

1. **Write pending** at the end of each run (``decision_writeback_node``).
2. **Resolve** at the start of the next run (``reflector_resolve_node``):
   fetch outcome, generate reflection, transition pending → resolved.
3. **Read** at the start of each run: inject up to *N* same-ticker and
   *M* cross-ticker resolved reflections into the Portfolio Manager prompt.

All operations gracefully degrade on store-side failures (log + continue)
so a flaky memory store can never break the trading-decision pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from ....utils.memory_config import MEMORIES_NAMESPACE_ROOT, MemoryConfiguration
from ..config import TradingDecisionConfiguration
from ..schemas import DecisionRecord, Outcome

logger = logging.getLogger(__name__)


def make_key(ticker: str, date: str) -> str:
    """Construct the deterministic store key for a (ticker, date) pair."""
    return f"{ticker.upper()}:{date}"


def split_key(key: str) -> tuple[str, str]:
    """Inverse of :func:`make_key`. Returns ``(ticker, date)``."""
    ticker, _, date = key.partition(":")
    return ticker, date


class ReflectionMemory:
    """Typed CRUD wrapper for trading-decision reflection storage.

    Construct one per (store, user_id) pair. All methods are async and
    swallow store-side failures (logging at DEBUG level) so the trading
    pipeline never blocks on memory infrastructure.
    """

    NAMESPACE_ROOT: tuple[str, ...] = MEMORIES_NAMESPACE_ROOT
    NAMESPACE_LEAF: tuple[str, ...] = ("decisions",)

    def __init__(self, store: BaseStore, user_id: str) -> None:
        """Bind to *store* under the namespace ``("memories", user_id, "decisions")``.

        Raises ``ValueError`` on empty *user_id* — per-user isolation is the
        whole point of the namespace, so accepting an empty key would silently
        merge every anonymous decision into one shared bucket.
        """
        if not user_id:
            raise ValueError("ReflectionMemory requires a non-empty user_id.")
        self._store = store
        self._user_id = user_id
        self._namespace: tuple[str, ...] = (
            *self.NAMESPACE_ROOT,
            user_id,
            *self.NAMESPACE_LEAF,
        )

    @property
    def namespace(self) -> tuple[str, ...]:
        """The fully-qualified BaseStore namespace this memory writes to."""
        return self._namespace

    # ── Write surface ───────────────────────────────────────────────────────

    async def write_pending(
        self, *, ticker: str, date: str, decision: dict[str, Any]
    ) -> None:
        """Write a new pending decision. Idempotent — overwrites any prior entry.

        Called by ``decision_writeback_node`` at the end of every successful
        trading-decision run. Errored ``portfolio_decision`` payloads (those
        carrying an ``error`` key) are skipped — there is no thesis worth
        learning from in that case.
        """
        if "error" in decision:
            logger.debug(
                "ReflectionMemory.write_pending skipping errored decision for %s/%s",
                ticker,
                date,
            )
            return
        record = DecisionRecord(
            ticker=ticker,
            date=date,
            status="pending",
            decision=decision,
            outcome=None,
            reflection=None,
        )
        try:
            await self._store.aput(
                self._namespace, make_key(ticker, date), record.model_dump()
            )
        except Exception:
            logger.debug(
                "ReflectionMemory.write_pending failed for %s/%s",
                ticker,
                date,
                exc_info=True,
            )

    async def resolve(
        self,
        *,
        ticker: str,
        date: str,
        outcome: Outcome,
        reflection: str,
    ) -> None:
        """Transition a pending entry to resolved + outcome + reflection."""
        key = make_key(ticker, date)
        try:
            existing = await self._store.aget(self._namespace, key)
        except Exception:
            logger.debug(
                "ReflectionMemory.resolve get failed for %s", key, exc_info=True
            )
            return
        if existing is None or not isinstance(existing.value, dict):
            logger.debug(
                "ReflectionMemory.resolve no pending entry for %s — skipping", key
            )
            return
        record = DecisionRecord.model_validate(existing.value)
        record.status = "resolved"
        record.outcome = outcome
        record.reflection = reflection
        try:
            await self._store.aput(self._namespace, key, record.model_dump())
        except Exception:
            logger.debug(
                "ReflectionMemory.resolve put failed for %s", key, exc_info=True
            )

    # ── Read surface ────────────────────────────────────────────────────────

    async def list_pending(self) -> list[DecisionRecord]:
        """Return all pending entries (any ticker)."""
        items = await self._safe_search()
        return [r for r in items if r.status == "pending"]

    async def list_pending_for_ticker(self, ticker: str) -> list[DecisionRecord]:
        """Return pending entries for a specific ticker."""
        ticker_upper = ticker.upper()
        return [
            r for r in await self.list_pending() if r.ticker.upper() == ticker_upper
        ]

    async def list_resolved_for_ticker(
        self, ticker: str, *, limit: int = 5
    ) -> list[DecisionRecord]:
        """Return the *limit* most-recently-dated resolved entries for *ticker*."""
        ticker_upper = ticker.upper()
        all_resolved = [
            r
            for r in await self._safe_search()
            if r.status == "resolved" and r.ticker.upper() == ticker_upper
        ]
        return sorted(all_resolved, key=lambda r: r.date, reverse=True)[:limit]

    async def list_resolved_cross_ticker(
        self, *, exclude_ticker: str, limit: int = 3
    ) -> list[DecisionRecord]:
        """Return the *limit* most-recently-dated resolved entries for OTHER tickers."""
        ticker_upper = exclude_ticker.upper()
        all_resolved = [
            r
            for r in await self._safe_search()
            if r.status == "resolved" and r.ticker.upper() != ticker_upper
        ]
        return sorted(all_resolved, key=lambda r: r.date, reverse=True)[:limit]

    # ── Internals ───────────────────────────────────────────────────────────

    async def _safe_search(self) -> list[DecisionRecord]:
        """Search the namespace, swallowing failures and dropping malformed rows."""
        try:
            items = await self._store.asearch(self._namespace)
        except Exception:
            logger.debug(
                "ReflectionMemory search failed for %s", self._namespace, exc_info=True
            )
            return []
        records: list[DecisionRecord] = []
        for item in items:
            if not isinstance(item.value, dict):
                continue
            try:
                records.append(DecisionRecord.model_validate(item.value))
            except Exception:
                logger.debug("ReflectionMemory dropping malformed entry %s", item.key)
        return records


def render_reflections_block(
    *,
    same_ticker: list[DecisionRecord],
    cross_ticker: list[DecisionRecord],
) -> str:
    """Render past reflections as a Markdown block for the PM prompt.

    Returns an empty string when both lists are empty — the prompt then
    omits the "past lessons" section entirely via Jinja conditional.
    """
    if not same_ticker and not cross_ticker:
        return ""

    sections: list[str] = []

    if same_ticker:
        sections.append("**Past same-ticker decisions and outcomes:**")
        for r in same_ticker:
            sections.append(_render_one(r))

    if cross_ticker:
        sections.append("**Past cross-ticker reflections (for general lessons):**")
        for r in cross_ticker:
            sections.append(_render_one(r, compact=True))

    return "\n\n".join(sections)


def _render_one(record: DecisionRecord, *, compact: bool = False) -> str:
    """Format a single resolved record. ``compact`` skips the decision payload."""
    rating = record.decision.get("rating", "?")
    outcome = record.outcome
    outcome_line = (
        f"raw {outcome.raw_return_pct:+.2f}% / alpha {outcome.alpha_return_pct:+.2f}% "
        f"over {outcome.holding_days}d"
        if outcome is not None
        else "outcome unavailable"
    )
    reflection = record.reflection or "(no reflection text)"
    head = f"- **{record.ticker} {record.date}** → {rating} | {outcome_line}"
    if compact:
        return f"{head}\n  {reflection}"
    return f"{head}\n  {reflection}"


def try_build_reflection_memory(
    config: RunnableConfig,
    store: BaseStore | None,
) -> ReflectionMemory | None:
    """Build a :class:`ReflectionMemory`, or return ``None`` if unavailable.

    Three "unavailable" branches (all return ``None`` silently — these are
    normal operating modes, not errors):

    * *store* is ``None`` (caller didn't wire one).
    * :attr:`TradingDecisionConfiguration.reflection_enabled` is ``False``.
    * ``user_id`` is not resolvable (no ``configurable.user_id`` and no
      ``MEMORY_DEBUG_USER_ID``).

    Both reflection bookend nodes call this as their single up-front gate
    so the per-node guard boilerplate stays a single line.
    """
    if store is None:
        return None
    cfg = TradingDecisionConfiguration.from_runnable_config(config)
    if not cfg.reflection_enabled:
        return None
    user_id = MemoryConfiguration.resolve_user_id(config, allow_missing=True)
    if user_id is None:
        return None
    return ReflectionMemory(store=store, user_id=user_id)
