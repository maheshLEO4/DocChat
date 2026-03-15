"""
graph/nodes.py
~~~~~~~~~~~~~~
Each function is a LangGraph node.
Signature:  (state: AgentState) -> dict   (partial state update)

Node order in the full workflow:
  rewrite_query → check_relevance → research → [verify] → END
"""

from graph.state import AgentState
from utils import get_logger

logger = get_logger(__name__)


# ── Utility: format history for prompts ──────────────────────────────────────

def _format_history(state: AgentState) -> str:
    """
    Convert the last N turns stored in state into a plain-text block.
    Returns an empty string if there is no history.
    """
    history = state.get("conversation_history") or []
    if not history:
        return ""
    lines = [f"{t['role'].capitalize()}: {t['content']}" for t in history]
    return "\n".join(lines)


# ── Node 1 : rewrite_query ────────────────────────────────────────────────────

def rewrite_query_node(state: AgentState) -> dict:
    """
    Rewrite the raw question into a self-contained retrieval query,
    resolving pronouns and references using conversation history.
    """
    from agents.research_agent import ResearchAgent

    logger.info("Node: rewrite_query")
    history = _format_history(state)
    agent = ResearchAgent(
        model_provider=state.get("model_provider"),
        model_name=state.get("model_name"),
    )
    rewritten = agent.rewrite_query(state["question"], history)
    return {"rewritten_query": rewritten}


# ── Node 2 : check_relevance ──────────────────────────────────────────────────

def check_relevance_node(state: AgentState) -> dict:
    """
    Decide whether the retrieved documents can answer the (rewritten) question.
    Sets is_relevant and, on failure, a final draft_answer.
    """
    from agents.relevance_agent import RelevanceAgent

    logger.info("Node: check_relevance")
    history = _format_history(state)
    agent = RelevanceAgent(
        model_provider=state.get("model_provider"),
        model_name=state.get("model_name"),
    )

    label = agent.check(
        question=state["rewritten_query"],
        documents=state["documents"],
        history=history,
    )

    if label in ("CAN_ANSWER", "PARTIAL"):
        return {"is_relevant": True}

    return {
        "is_relevant": False,
        "draft_answer": (
            "❌ The question doesn't appear to be covered by the uploaded documents."
        ),
    }


# ── Node 3 : research ────────────────────────────────────────────────────────

def research_node(state: AgentState) -> dict:
    """
    Generate a draft answer from the retrieved documents,
    using conversation history for context continuity.
    """
    from agents.research_agent import ResearchAgent

    logger.info("Node: research")
    history = _format_history(state)
    agent = ResearchAgent(
        model_provider=state.get("model_provider"),
        model_name=state.get("model_name"),
    )

    result = agent.generate(
        question=state["rewritten_query"],
        documents=state["documents"],
        history=history,
    )

    return {
        "draft_answer": result["draft_answer"],
        "citations": result["citations"],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


# ── Node 4 : verify ──────────────────────────────────────────────────────────

def verify_node(state: AgentState) -> dict:
    """
    Check whether the draft answer is grounded in the retrieved documents.
    On failure the workflow loops back to research (up to MAX_ITERATIONS).
    """
    from agents.verification_agent import VerificationAgent

    logger.info("Node: verify")
    agent = VerificationAgent(
        model_provider=state.get("model_provider"),
        model_name=state.get("model_name"),
    )
    result = agent.check(
        answer=state["draft_answer"],
        documents=state["documents"],
    )
    return {"verification_report": result["verification_report"]}
