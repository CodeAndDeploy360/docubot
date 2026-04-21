"""LangGraph workflow: Planner → Retriever → Grader → (re-plan) → Generator prep."""

from __future__ import annotations

import json
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent import prompts
from agent.state import AgentState
from agent.tools import knowledge_base_search, web_search
from services.llm import build_chat_model, invoke_with_rate_limit_retry


class PlanStep(BaseModel):
    action: Literal["rag_search", "web_search"]
    query: str


class Plan(BaseModel):
    steps: list[PlanStep] = Field(min_length=1, max_length=6)


class Grade(BaseModel):
    sufficient: bool
    reason: str


def _history_block(history: list[tuple[str, str]], limit: int = 6) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for h, a in history[-limit:]:
        lines.append(f"User: {h}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def _usage_from_message(msg: AIMessage) -> dict[str, Any]:
    meta = getattr(msg, "response_metadata", None) or {}
    usage = dict(meta.get("token_usage") or {})
    if getattr(msg, "usage_metadata", None):
        usage.update(dict(msg.usage_metadata))
    return usage


def planner_node(state: AgentState) -> dict[str, Any]:
    replan = state.get("replan_note", "")
    replan_hint = ""
    if replan:
        replan_hint = (
            f"Previous retrieval was insufficient. Reason: {replan}\n"
            "Try different search queries. Use web_search only if the user needs live web facts."
        )

    llm = build_chat_model(temperature=0.2).with_structured_output(Plan)
    sys = SystemMessage(content=prompts.PLANNER_SYSTEM)
    human = HumanMessage(
        content=prompts.PLANNER_USER.format(
            history=_history_block(state.get("chat_history") or []),
            question=state["user_query"],
            replan_hint=replan_hint,
        )
    )
    plan_msg = invoke_with_rate_limit_retry(lambda: llm.invoke([sys, human]))
    usage: dict[str, Any] = {}
    if isinstance(plan_msg, AIMessage):
        usage = _usage_from_message(plan_msg)
    if isinstance(plan_msg, Plan):
        plan_obj = plan_msg
    elif isinstance(plan_msg, dict):
        plan_obj = Plan.model_validate(plan_msg)
    elif isinstance(plan_msg, AIMessage) and isinstance(plan_msg.content, str):
        plan_obj = Plan.model_validate_json(plan_msg.content)
    else:
        raise TypeError(f"Unexpected planner output type: {type(plan_msg)}")
    steps = [s.model_dump() for s in plan_obj.steps]
    trace = {
        "node": "planner",
        "detail": {"steps": steps},
        "usage": usage,
        "ts": time.time(),
    }
    return {"plan": steps, "trace": [trace]}


def _normalize_rag_payload(raw: str, label_prefix: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [
            {
                "kind": "rag",
                "label": f"S{label_prefix}",
                "text": raw[:8000],
                "metadata": {"source": "unknown", "page": -1, "chunk_index": -1},
            }
        ]
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        meta = item.get("metadata") or {}
        out.append(
            {
                "kind": "rag",
                "label": f"S{len(out) + 1}",
                "text": str(item.get("text", "")),
                "metadata": meta,
                "score": float(item.get("score") or 0.0),
            }
        )
    return out


def _normalize_web_payload(raw: str, label_prefix: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [
            {
                "kind": "web",
                "label": f"W{label_prefix}",
                "text": raw[:8000],
                "metadata": {"source": "web", "page": -1, "chunk_index": -1},
            }
        ]
    if isinstance(data, dict) and data.get("error"):
        return [
            {
                "kind": "web",
                "label": f"W{label_prefix}",
                "text": f"web_search error: {data.get('error')}",
                "metadata": {"source": "web", "page": -1, "chunk_index": -1},
            }
        ]
    if not isinstance(data, list):
        return out
    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or ""
        body = item.get("body") or ""
        href = item.get("href") or ""
        text = f"{title}\n{href}\n{body}".strip()
        out.append(
            {
                "kind": "web",
                "label": f"W{len(out) + 1}",
                "text": text,
                "metadata": {"source": "web", "page": -1, "chunk_index": -1},
            }
        )
    return out


def retriever_node(state: AgentState) -> dict[str, Any]:
    plan = state.get("plan") or []
    retrieved: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    def add_batch(batch: list[dict[str, Any]]) -> None:
        import hashlib

        for item in batch:
            h = hashlib.sha256(item["text"].encode("utf-8", errors="ignore")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            retrieved.append(item)

    for step in plan:
        action = step.get("action")
        query = str(step.get("query", "")).strip()
        if not query:
            continue
        if action == "rag_search":
            raw = knowledge_base_search.invoke({"query": query})
            add_batch(_normalize_rag_payload(raw, 1))
        elif action == "web_search":
            raw = web_search.invoke({"query": query})
            add_batch(_normalize_web_payload(raw, 1))

    trace = {
        "node": "retriever",
        "detail": {"steps_executed": len(plan), "chunks": len(retrieved)},
        "usage": {},
        "ts": time.time(),
    }
    return {"retrieved": retrieved, "trace": [trace]}


def _evidence_block(retrieved: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for item in retrieved:
        meta = item.get("metadata") or {}
        head = (
            f"[{item.get('label')}] kind={item.get('kind')} "
            f"source={meta.get('source')} page={meta.get('page')} chunk={meta.get('chunk_index')}"
        )
        parts.append(head + "\n" + str(item.get("text", "")))
    return "\n\n".join(parts) if parts else "(no evidence retrieved)"


def _retrieval_score(item: dict[str, Any]) -> float:
    s = item.get("score")
    if s is not None:
        return float(s)
    meta = item.get("metadata") or {}
    if meta.get("score") is not None:
        return float(meta["score"])
    return 0.0


def format_unique_source_lines(
    retrieved: list[dict[str, Any]],
    *,
    max_lines: int | None = None,
) -> list[str]:
    """
    Distinct chunks for the Sources UI, ranked by hybrid retrieval score, capped for a short list
    (matches the product mockup: 1–3 lines).
    """
    from config import max_sources_display

    cap = max_lines if max_lines is not None else max_sources_display()

    # Dedupe by human-visible slot (file + page + chunk index), not Chroma id: hybrid
    # fusion can surface the same logical chunk under multiple internal ids (e.g. duplicate
    # uploads), which would otherwise repeat the same source line.
    best: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}
    for item in retrieved:
        meta = item.get("metadata") or {}
        src = str(meta.get("source", "unknown"))
        page = meta.get("page", -1)
        chunk = meta.get("chunk_index", -1)
        try:
            page = int(page) if page is not None else -1
        except (TypeError, ValueError):
            page = -1
        try:
            chunk = int(chunk) if chunk is not None else -1
        except (TypeError, ValueError):
            chunk = -1
        key = (src, page, chunk)
        sc = _retrieval_score(item)
        prev = best.get(key)
        if prev is None or sc > prev[0]:
            best[key] = (sc, item)

    ranked = sorted(best.values(), key=lambda x: x[0], reverse=True)
    lines: list[str] = []
    for _sc, it in ranked[:cap]:
        meta = it.get("metadata") or {}
        src = str(meta.get("source", "unknown"))
        page = meta.get("page", -1)
        chunk = meta.get("chunk_index", -1)
        page_part = f"page {page}" if page not in (None, -1) else "page —"
        lines.append(f"{src} — {page_part}, chunk {chunk}")
    return lines


def grader_node(state: AgentState) -> dict[str, Any]:
    llm = build_chat_model(temperature=0.0).with_structured_output(Grade)
    sys = SystemMessage(content=prompts.GRADER_SYSTEM)
    human = HumanMessage(
        content=prompts.GRADER_USER.format(
            question=state["user_query"],
            evidence=_evidence_block(state.get("retrieved") or []),
        )
    )
    grade_msg = invoke_with_rate_limit_retry(lambda: llm.invoke([sys, human]))
    usage = _usage_from_message(grade_msg) if isinstance(grade_msg, AIMessage) else {}
    if isinstance(grade_msg, Grade):
        grade = grade_msg
    elif isinstance(grade_msg, dict):
        grade = Grade.model_validate(grade_msg)
    elif isinstance(grade_msg, AIMessage) and isinstance(grade_msg.content, str):
        grade = Grade.model_validate_json(grade_msg.content)
    else:
        raise TypeError(f"Unexpected grader output type: {type(grade_msg)}")
    sufficient = bool(grade.sufficient)
    reason = str(grade.reason)
    trace = {
        "node": "grader",
        "detail": {"sufficient": sufficient, "reason": reason},
        "usage": usage,
        "ts": time.time(),
    }
    updates: dict[str, Any] = {
        "grader_sufficient": sufficient,
        "grader_reason": reason,
        "trace": [trace],
    }
    if not sufficient:
        updates["replan_note"] = reason
    else:
        updates["replan_note"] = ""
    return updates


def bump_replan_node(state: AgentState) -> dict[str, Any]:
    n = int(state.get("replan_count") or 0) + 1
    trace = {"node": "replan", "detail": {"replan_count": n}, "usage": {}, "ts": time.time()}
    return {"replan_count": n, "trace": [trace]}


def generator_prep_node(state: AgentState) -> dict[str, Any]:
    retrieved = state.get("retrieved") or []
    context = _evidence_block(retrieved)
    sys = SystemMessage(content=prompts.GENERATOR_SYSTEM)
    human = HumanMessage(
        content=prompts.GENERATOR_USER.format(context=context, question=state["user_query"])
    )
    sources = format_unique_source_lines(retrieved)
    trace = {
        "node": "generator",
        "detail": {"mode": "prompt_ready", "sources": sources},
        "usage": {},
        "ts": time.time(),
    }
    return {"generation_messages": [sys, human], "sources": sources, "trace": [trace]}


def route_after_grade(state: AgentState) -> Literal["generator", "bump_replan"]:
    if state.get("grader_sufficient"):
        return "generator"
    if int(state.get("replan_count") or 0) < 2:
        return "bump_replan"
    return "generator"


def build_workflow() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("grader", grader_node)
    g.add_node("bump_replan", bump_replan_node)
    g.add_node("generator", generator_prep_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "grader")
    g.add_conditional_edges(
        "grader",
        route_after_grade,
        {
            "generator": "generator",
            "bump_replan": "bump_replan",
        },
    )
    g.add_edge("bump_replan", "planner")
    g.add_edge("generator", END)
    return g


def compile_app_graph():
    return build_workflow().compile()
