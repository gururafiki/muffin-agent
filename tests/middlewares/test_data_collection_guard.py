"""Evidence gathering for the criteria anti-hallucination pass.

The failure this guards against is silent and total: if the evidence list is
ever non-empty for an agent that retrieved nothing, ``_reconcile_data_sources``
marks the evaluation ``data_collected=True`` and keeps every fabricated
``data_sources`` entry — the exact behaviour the pass exists to prevent, with no
error anywhere.

That is not hypothetical. ``AutoStrategy`` surfaces the response schema as a
tool, so the synthetic final call that emits the verdict looks identical to a
real tool execution. Without the schema-name exclusion every criterion in the
integration suite came back ``data_collected=True`` while calling zero tools.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from muffin_agent.middlewares.data_collection_guard import (
    DataCollectionGuardMiddleware,
    executed_tool_labels,
)


def _call(tool_id: str, name: str, args: dict | None = None) -> AIMessage:
    call = {"id": tool_id, "name": name, "args": args or {}, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


@pytest.mark.unit
class TestExecutedToolLabels:
    def test_completed_call_is_evidence(self):
        labels = executed_tool_labels(
            [
                HumanMessage(content="go"),
                _call("a", "equity_fundamental_metrics", {"ticker": "AAPL"}),
                ToolMessage(content="{...}", tool_call_id="a"),
            ],
            agent_name="criterion_evaluation",
        )
        assert len(labels) == 1
        assert "equity_fundamental_metrics" in labels[0]
        assert "AAPL" in labels[0]
        assert "criterion_evaluation" in labels[0]

    def test_call_without_a_result_is_not_evidence(self):
        """Intent is not retrieval — and intent is what a fabricator produces."""
        labels = executed_tool_labels([_call("a", "equity_fundamental_metrics")])
        assert labels == []

    def test_response_schema_call_is_excluded(self):
        """The regression that made every evaluation look data-backed."""
        messages = [
            _call("a", "CriterionEvaluationNodeOutput", {"evaluation": {"score": 0.7}}),
            ToolMessage(content="ok", tool_call_id="a"),
        ]
        assert executed_tool_labels(messages) != []  # without the exclusion
        assert (
            executed_tool_labels(
                messages, exclude_tools=frozenset({"CriterionEvaluationNodeOutput"})
            )
            == []
        )

    def test_plumbing_calls_are_excluded(self):
        """An agent that only wrote a todo list collected nothing."""
        labels = executed_tool_labels(
            [
                _call("a", "write_todos", {"todos": []}),
                ToolMessage(content="ok", tool_call_id="a"),
                _call("b", "ls", {}),
                ToolMessage(content="ok", tool_call_id="b"),
            ]
        )
        assert labels == []

    def test_task_delegation_is_kept_with_its_subagent_name(self):
        """`data_sources[].subagent` is matched against the delegation args."""
        labels = executed_tool_labels(
            [
                _call("a", "task", {"subagent_type": "equity-fundamentals"}),
                ToolMessage(content="report", tool_call_id="a"),
            ]
        )
        assert len(labels) == 1
        assert "equity-fundamentals" in labels[0]

    def test_orphan_tool_message_is_ignored(self):
        """A result with no matching call can't be attributed to anything."""
        orphan = ToolMessage(content="x", tool_call_id="ghost")
        assert executed_tool_labels([orphan]) == []


@pytest.mark.unit
class TestMiddleware:
    def test_emits_labels_for_completed_calls(self):
        mw = DataCollectionGuardMiddleware(name="criterion_evaluation")
        update = mw.after_agent(
            {
                "messages": [
                    _call("a", "equity_fundamental_metrics", {"ticker": "AAPL"}),
                    ToolMessage(content="{...}", tool_call_id="a"),
                ]
            },
            None,  # type: ignore[arg-type]
        )
        assert update is not None
        assert len(update["executed_tools"]) == 1

    def test_emits_nothing_when_no_tool_ran(self):
        """No update at all, so the reducer never sees an empty-but-present list."""
        mw = DataCollectionGuardMiddleware(name="criterion_evaluation")
        assert mw.after_agent({"messages": [HumanMessage(content="hi")]}, None) is None  # type: ignore[arg-type]

    def test_honours_the_exclusion_it_was_constructed_with(self):
        mw = DataCollectionGuardMiddleware(
            name="criterion_evaluation",
            exclude_tools=frozenset({"CriterionEvaluationNodeOutput"}),
        )
        update = mw.after_agent(
            {
                "messages": [
                    _call("a", "CriterionEvaluationNodeOutput", {}),
                    ToolMessage(content="ok", tool_call_id="a"),
                ]
            },
            None,  # type: ignore[arg-type]
        )
        assert update is None


@pytest.mark.unit
class TestEnforcement:
    """`max_attempts > 0` — bounce a data-free answer back to the model.

    The hazard here is a livelock, not a wrong answer: every persona collector
    runs under `ModelCallLimitMiddleware(exit_behavior="end")`, so once the call
    budget is spent the model never runs again and the transcript stops gaining
    nudges. A nudge-only ceiling would then never advance and the guard would
    bounce forever (the same bug `_StructuredOutputRetryMiddleware` documents,
    which reached the 10011 recursion limit in prod).
    """

    @staticmethod
    def _guard(max_attempts: int = 2) -> DataCollectionGuardMiddleware:
        return DataCollectionGuardMiddleware(
            name="collector",
            exclude_tools=frozenset({"RawData"}),
            max_attempts=max_attempts,
        )

    def test_data_free_answer_jumps_back_to_the_model(self):
        update = self._guard().after_agent(
            {"messages": [HumanMessage(content="score AAPL")]},
            None,  # type: ignore[arg-type]
        )
        assert update is not None
        assert update["jump_to"] == "model"
        assert update["data_collection_jumps"] == 1
        assert len(update["messages"]) == 1

    def test_a_single_shot_structured_answer_still_counts_as_data_free(self):
        """The exact prod failure: schema tool called, no data tool called."""
        update = self._guard().after_agent(
            {
                "messages": [
                    _call("a", "RawData", {"pe_ratio": 12.0}),
                    ToolMessage(content="ok", tool_call_id="a"),
                ]
            },
            None,  # type: ignore[arg-type]
        )
        assert update is not None and update["jump_to"] == "model"

    def test_a_real_tool_call_passes_straight_through(self):
        update = self._guard().after_agent(
            {
                "messages": [
                    _call("a", "equity_fundamental_metrics", {"ticker": "AAPL"}),
                    ToolMessage(content="{...}", tool_call_id="a"),
                ]
            },
            None,  # type: ignore[arg-type]
        )
        assert update is not None
        assert "jump_to" not in update
        assert len(update["executed_tools"]) == 1

    def test_gives_up_after_max_attempts_rather_than_raising(self):
        """A flagged low-confidence answer beats a failed run."""
        nudge = {"data_collection_nudge": True}
        nudged = [
            HumanMessage(content="x", additional_kwargs=nudge),
            HumanMessage(content="y", additional_kwargs=nudge),
        ]
        update = self._guard(max_attempts=2).after_agent(
            {"messages": nudged},
            None,  # type: ignore[arg-type]
        )
        assert update is None  # accepted; downstream marks data_collected=False

    def test_jump_counter_terminates_when_the_model_can_no_longer_run(self):
        """The livelock backstop: no new nudges appear, but jumps still count."""
        state = {"messages": [], "data_collection_jumps": 2}
        assert self._guard(max_attempts=2).after_agent(state, None) is None  # type: ignore[arg-type]

    def test_bounded_loop_terminates(self):
        """Drive the guard like the runtime would; it must stop on its own."""
        guard = self._guard(max_attempts=2)
        state: dict = {"messages": [], "data_collection_jumps": 0}
        jumps = 0
        for _ in range(10):  # generous ceiling; 2 is expected
            update = guard.after_agent(state, None)  # type: ignore[arg-type]
            if not update or "jump_to" not in update:
                break
            jumps += 1
            # Mirror the reducers: messages append, the counter adds.
            state["messages"] = state["messages"] + update["messages"]
            state["data_collection_jumps"] += update["data_collection_jumps"]
        else:
            pytest.fail("guard never stopped jumping")
        assert jumps == 2

    def test_recording_only_when_enforcement_is_off(self):
        """`max_attempts=0` — some agents legitimately collect nothing."""
        guard = DataCollectionGuardMiddleware(name="x", max_attempts=0)
        assert guard.after_agent({"messages": []}, None) is None  # type: ignore[arg-type]
