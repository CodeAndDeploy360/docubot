"""Typed workflow state for LangGraph."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    user_query: str
    chat_history: list[tuple[str, str]]
    plan: list[dict[str, Any]]
    retrieved: list[dict[str, Any]]
    grader_sufficient: bool
    grader_reason: str
    replan_count: int
    replan_note: str
    trace: Annotated[list[dict[str, Any]], add]
    generation_messages: list[BaseMessage]
    sources: list[str]
    token_usage: dict[str, Any]
    workflow_started_at: float
