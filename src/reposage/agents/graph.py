from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from reposage.agents.nodes import (
    analyst_node,
    critic_node,
    finalizer_node,
    planner_node,
    retriever_node,
)
from reposage.agents.state import AgentDeps, AgentState
from reposage.logging_setup import get_logger

log = get_logger(__name__)

NODE_PLAN = "plan"
NODE_RETRIEVE = "retrieve"
NODE_ANALYSE = "analyse"
NODE_CRITIQUE = "critique"
NODE_FINALISE = "finalise"


def route_after_plan(state: AgentState) -> str:
    plan = state.get("plan")
    if plan is not None and not plan.needs_retrieval:
        return NODE_ANALYSE
    return NODE_RETRIEVE


def route_after_critique(state: AgentState) -> str:
    critique = state.get("critique")
    if critique is None:
        return NODE_FINALISE
    if critique.needs_refinement:
        return NODE_RETRIEVE
    return NODE_FINALISE


def build_graph(deps: AgentDeps) -> Any:
    builder = StateGraph(AgentState)

    builder.add_node(NODE_PLAN, partial(planner_node, deps=deps))
    builder.add_node(NODE_RETRIEVE, partial(retriever_node, deps=deps))
    builder.add_node(NODE_ANALYSE, partial(analyst_node, deps=deps))
    builder.add_node(NODE_CRITIQUE, partial(critic_node, deps=deps))
    builder.add_node(NODE_FINALISE, partial(finalizer_node, deps=deps))

    builder.add_edge(START, NODE_PLAN)
    builder.add_conditional_edges(
        NODE_PLAN,
        route_after_plan,
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_ANALYSE: NODE_ANALYSE},
    )
    builder.add_edge(NODE_RETRIEVE, NODE_ANALYSE)
    builder.add_edge(NODE_ANALYSE, NODE_CRITIQUE)
    builder.add_conditional_edges(
        NODE_CRITIQUE,
        route_after_critique,
        {NODE_RETRIEVE: NODE_RETRIEVE, NODE_FINALISE: NODE_FINALISE},
    )
    builder.add_edge(NODE_FINALISE, END)

    compiled = builder.compile()
    log.debug("graph.compiled", nodes=5)
    return compiled


def describe_graph() -> str:
    return """graph TD
    START([question]) --> PLAN[plan<br/>decompose into search queries]
    PLAN -->|needs retrieval| RETRIEVE[retrieve<br/>hybrid search + fusion + rerank]
    PLAN -->|map is enough| ANALYSE
    RETRIEVE --> ANALYSE[analyse<br/>draft a cited answer]
    ANALYSE --> CRITIQUE[critique<br/>audit grounding + completeness]
    CRITIQUE -->|refine| RETRIEVE
    CRITIQUE -->|accept| FINALISE[finalise<br/>verify citations, score confidence]
    FINALISE --> DONE([answer])"""
