from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from muffin_agent.middlewares.agent_capture.middleware import AgentCaptureMiddleware
from muffin_agent.middlewares.agent_capture.tree import (
    build_tree_node,
    merge_subagent_tree,
    node_ids_from_ns,
    resolve_node_id,
)


def test_ns_parsing_root():
    assert node_ids_from_ns("") == ("__root__", None)
    assert node_ids_from_ns(None) == ("__root__", None)


def test_ns_parsing_nested():
    # LangGraph ns segments are pipe-separated "<node>:<task_id>"
    assert node_ids_from_ns("persona:abc") == ("persona:abc", "__root__")
    assert node_ids_from_ns("persona:abc|collect:def") == (
        "persona:abc|collect:def",
        "persona:abc",
    )


def test_ns_parsing_strips_trailing_middleware_segment():
    # Real namespaces (Task-1 spike): the capturing middleware's own node is the
    # trailing segment and must be stripped before deriving id/parent.
    assert node_ids_from_ns(
        "mohnish_pabrai:a|collect_data:b|AgentCaptureMiddleware.after_agent:c"
    ) == ("mohnish_pabrai:a|collect_data:b", "mohnish_pabrai:a")
    assert node_ids_from_ns(
        "ticker_classification:a|AgentCaptureMiddleware.after_agent:b"
    ) == ("ticker_classification:a", "__root__")
    assert node_ids_from_ns("AgentCaptureMiddleware.after_agent:a") == (
        "__root__",
        None,
    )


def test_build_node_summarises_tools():
    runs = [
        {"tool": "news_company", "status": "ok"},
        {"tool": "news_company", "status": "error"},
        {"tool": "equity_price", "status": "ok", "cache_hit": True},
    ]
    n = build_tree_node(
        node_id="p:1",
        parent_id="__root__",
        name="pabrai",
        kind="subgraph",
        tool_runs=runs,
        output={"signal": "hold"},
    )
    assert n["name"] == "pabrai" and n["parent_id"] == "__root__"
    assert n["tool_summary"] == {
        "count": 3,
        "tools": ["news_company", "equity_price"],
        "ok": 2,
        "failed": 1,
        "cached": 1,
    }
    assert n["output_preview"] and n["has_detail"] is True


def test_resolve_node_id_mints_unique_id_on_task_collision():
    resolved = resolve_node_id("p:1", "p:1", "task")
    assert resolved != "p:1"
    assert resolved.startswith("p:1|task:")


def test_resolve_node_id_leaves_distinct_task_id_unchanged():
    assert resolve_node_id("p:1|c:2", "p:1", "task") == "p:1|c:2"


def test_resolve_node_id_leaves_subgraph_collision_unchanged():
    # Only task-kind collisions are minted a unique id; subgraph nesting is
    # validated to always produce a distinct id from its parent already.
    assert resolve_node_id("p:1", "p:1", "subgraph") == "p:1"


def test_reducer_merges_by_id():
    a = {"p:1": {"id": "p:1"}}
    b = {"p:1|c:2": {"id": "p:1|c:2"}}
    assert set(merge_subagent_tree(a, b)) == {"p:1", "p:1|c:2"}


def test_capture_emits_tree_node(monkeypatch):
    monkeypatch.setattr(
        "muffin_agent.middlewares.agent_capture.middleware.get_config",
        lambda: {"configurable": {"checkpoint_ns": "pabrai:1"}},
    )
    mw = AgentCaptureMiddleware(name="pabrai")
    state = {
        "messages": [
            HumanMessage("evaluate AAPL"),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "equity_price", "args": {"t": "AAPL"}, "id": "t1"}
                ],
            ),
            ToolMessage("100", tool_call_id="t1"),
            AIMessage("done"),
        ]
    }
    update = mw._capture(state)
    assert update is not None
    node = update["subagent_tree"]["pabrai:1"]
    assert node["name"] == "pabrai"
    assert node["tool_summary"]["count"] == 1
    assert node["tool_summary"]["tools"] == ["equity_price"]
