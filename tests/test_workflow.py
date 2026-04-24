"""LangGraph workflow: routing, compile, re-plan trace (stubbed), prompts, and source-line helpers used by the UI."""

from __future__ import annotations

from agent.graph import route_after_grade


def test_route_after_grade():
    assert route_after_grade({"grader_sufficient": True}) == "generator"  # type: ignore[arg-type]
    assert (
        route_after_grade({"grader_sufficient": False, "replan_count": 0}) == "bump_replan"  # type: ignore[arg-type]
    )
    assert (
        route_after_grade({"grader_sufficient": False, "replan_count": 1}) == "bump_replan"  # type: ignore[arg-type]
    )
    assert route_after_grade({"grader_sufficient": False, "replan_count": 2}) == "generator"  # type: ignore[arg-type]


def test_workflow_compiles(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from agent.graph import compile_app_graph

    g = compile_app_graph()
    assert g is not None


def test_replan_loop_trace_with_stubbed_nodes(monkeypatch):
    """Grader insufficient then sufficient → trace includes replan + second planner/grader (no LLM)."""
    from agent import graph as gmod

    def fake_planner(state):
        return {
            "plan": [{"action": "rag_search", "query": "stub"}],
            "trace": [{"node": "planner", "detail": {"steps": [["rag_search"]]}, "usage": {}, "ts": 0.0}],
        }

    def fake_retriever(state):
        return {
            "retrieved": [
                {
                    "kind": "rag",
                    "label": "S1",
                    "text": "stub chunk",
                    "metadata": {"source": "doc.pdf", "page": 1, "chunk_index": 0},
                    "score": 0.4,
                }
            ],
            "trace": [{"node": "retriever", "detail": {"chunks": 1}, "usage": {}, "ts": 0.0}],
        }

    n = {"g": 0}

    def fake_grader(state):
        n["g"] += 1
        sufficient = n["g"] >= 2
        return {
            "grader_sufficient": sufficient,
            "grader_reason": "need more context" if not sufficient else "ok",
            "trace": [
                {
                    "node": "grader",
                    "detail": {"sufficient": sufficient, "reason": "mock"},
                    "usage": {},
                    "ts": 0.0,
                }
            ],
        }

    monkeypatch.setattr(gmod, "planner_node", fake_planner)
    monkeypatch.setattr(gmod, "retriever_node", fake_retriever)
    monkeypatch.setattr(gmod, "grader_node", fake_grader)

    from agent.graph import compile_app_graph

    graph = compile_app_graph()
    out = graph.invoke(
        {
            "user_id": "00000000-0000-0000-0000-000000000001",
            "user_query": "vague multi-part question",
            "chat_history": [],
            "replan_count": 0,
            "replan_note": "",
            "trace": [],
            "workflow_started_at": 0.0,
        }
    )

    trace = out.get("trace") or []
    nodes = [t.get("node") for t in trace if isinstance(t, dict)]
    assert nodes.count("planner") >= 2
    assert "replan" in nodes
    assert nodes.count("grader") >= 2
    assert out.get("grader_sufficient") is True
    assert out.get("generation_messages")


def test_generator_prompts_discourage_hallucination_when_context_missing():
    from agent import prompts
    from agent.graph import _evidence_block

    low = prompts.GENERATOR_SYSTEM.lower()
    assert "not enough information" in low or "do not invent" in low
    assert "context" in prompts.GENERATOR_USER.lower()
    assert "no evidence retrieved" in _evidence_block([]).lower()


def test_format_unique_source_lines_collapses_same_page_chunk_different_ids():
    from agent.graph import format_unique_source_lines

    retrieved = [
        {"metadata": {"source": "sample.pdf", "page": 1, "chunk_index": 0, "chunk_id": "a"}, "score": 0.9},
        {"metadata": {"source": "sample.pdf", "page": 1, "chunk_index": 0, "chunk_id": "b"}, "score": 0.5},
        {"metadata": {"source": "sample.pdf", "page": 1, "chunk_index": 0, "chunk_id": "c"}, "score": 0.3},
    ]
    lines = format_unique_source_lines(retrieved, max_lines=5)
    assert len(lines) == 1
    assert "sample.pdf" in lines[0]


def test_format_unique_source_lines_dedupes_ranks_and_caps():
    from agent.graph import format_unique_source_lines

    retrieved = [
        {"metadata": {"source": "low.pdf", "page": 1, "chunk_index": 0}, "score": 0.1},
        {"metadata": {"source": "low.pdf", "page": 1, "chunk_index": 0}, "score": 0.5},
        {"metadata": {"source": "high.pdf", "page": 2, "chunk_index": 1}, "score": 0.9},
        {"metadata": {"source": "mid.pdf", "page": 1, "chunk_index": 2}, "score": 0.3},
    ]
    lines = format_unique_source_lines(retrieved, max_lines=2)
    assert len(lines) == 2
    assert "high.pdf" in lines[0]
    assert "low.pdf" in lines[1]
