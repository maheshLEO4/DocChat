"""
graph/workflow.py
~~~~~~~~~~~~~~~~~
Builds and runs the multi-agent LangGraph workflow.

LangGraph 10-step pattern used here
─────────────────────────────────────
1.  Define State           → graph/state.py  (AgentState TypedDict)
2.  Create Nodes           → graph/nodes.py  (one function per node)
3.  Initialise StateGraph  → AgentWorkflow._build_workflow()
4.  Add Nodes to Graph     → workflow.add_node(...)
5.  Set Entry Point        → workflow.set_entry_point(...)
6.  Add Edges              → workflow.add_edge(...)
7.  Add Conditional Edges  → workflow.add_conditional_edges(...)
8.  Compile the Graph      → workflow.compile()
9.  Invoke / Run           → compiled.invoke(initial_state)
10. Get Final Output       → final_state dict returned to caller
"""

from typing import Any
from langgraph.graph import StateGraph, END

from graph.state import AgentState, Turn
from graph.nodes import (
    rewrite_query_node,
    check_relevance_node,
    research_node,
    verify_node,
)
from config import MAX_ITERATIONS, FINAL_TOP_K
from utils import get_logger

logger = get_logger(__name__)

HISTORY_WINDOW = 4   # keep last 4 user+assistant pairs = 8 turns total


class AgentWorkflow:
    """
    Orchestrates the full RAG pipeline via LangGraph.

    Fast mode  (enable_verification=False):
        rewrite_query → research → END

    Full mode  (enable_verification=True):
        rewrite_query → check_relevance → research → verify → [loop|END]
    """

    def __init__(self, enable_verification: bool = False):
        self.enable_verification = enable_verification
        self.app = self._build_workflow()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3–8 : build the graph
    # ─────────────────────────────────────────────────────────────────────────

    def _build_workflow(self):
        # Step 3: initialise
        workflow = StateGraph(AgentState)

        if self.enable_verification:
            # Step 4: add nodes
            workflow.add_node("rewrite_query",    rewrite_query_node)
            workflow.add_node("check_relevance",  check_relevance_node)
            workflow.add_node("research",         research_node)
            workflow.add_node("verify",           verify_node)

            # Step 5: entry point
            workflow.set_entry_point("rewrite_query")

            # Step 6: linear edges
            workflow.add_edge("rewrite_query", "check_relevance")
            workflow.add_edge("research", "verify")

            # Step 7: conditional edges
            workflow.add_conditional_edges(
                "check_relevance",
                self._after_relevance,
                {"relevant": "research", "irrelevant": END},
            )
            workflow.add_conditional_edges(
                "verify",
                self._after_verify,
                {"re_research": "research", "end": END},
            )

        else:
            # Fast path — no relevance check, no verification
            workflow.add_node("rewrite_query", rewrite_query_node)
            workflow.add_node("research",      research_node)

            workflow.set_entry_point("rewrite_query")
            workflow.add_edge("rewrite_query", "research")
            workflow.add_edge("research", END)

        # Step 8: compile
        return workflow.compile()

    # ─────────────────────────────────────────────────────────────────────────
    # Conditional edge functions
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _after_relevance(state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.info(f"Relevance gate → {decision}")
        return decision

    @staticmethod
    def _after_verify(state: AgentState) -> str:
        report = state.get("verification_report", "")
        iterations = state.get("iteration_count", 0)

        if iterations >= MAX_ITERATIONS:
            logger.info("Max iterations reached → end")
            return "end"

        # Re-run research if verification found unsupported claims or irrelevance
        if "Supported: NO" in report or "Relevant: NO" in report:
            logger.info("Verification failed → re_research")
            return "re_research"

        logger.info("Verification passed → end")
        return "end"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 9–10 : public pipeline entry
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        retriever: Any,
        conversation_history: list[Turn] | None = None,
    ) -> dict:
        """
        Run the full pipeline for one user turn.

        Args:
            question:             raw user question
            retriever:            HybridRetriever instance
            conversation_history: list of Turn dicts from session state

        Returns:
            {
                "draft_answer":        str,
                "citations":           list[str],
                "verification_report": str,
                "updated_history":     list[Turn],  # window-trimmed, ready to store
            }
        """
        history = list(conversation_history or [])

        # Retrieve documents using the *current* raw question first;
        # the graph will rewrite it internally for agent calls.
        try:
            documents = retriever.invoke(question)
        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            return {
                "draft_answer": "❌ Error retrieving documents. Please re-index your PDFs.",
                "citations": [],
                "verification_report": "",
                "updated_history": history,
            }

        logger.info(f"Retrieved {len(documents)} document(s) for: '{question}'")

        # Step 9: build initial state and invoke
        initial_state: AgentState = {
            "question": question,
            "rewritten_query": question,         # will be overwritten by rewrite node
            "conversation_history": history,
            "documents": documents,
            "is_relevant": True,
            "draft_answer": "",
            "citations": [],
            "verification_report": (
                "⚡ Verification disabled for faster responses"
                if not self.enable_verification
                else ""
            ),
            "retriever": retriever,
            "iteration_count": 0,
            "enable_verification": self.enable_verification,
        }

        try:
            # Step 10: get final output
            final_state = self.app.invoke(initial_state)
        except Exception as exc:
            logger.error(f"Workflow execution error: {exc}")
            return {
                "draft_answer": f"❌ Workflow error: {exc}",
                "citations": [],
                "verification_report": "",
                "updated_history": history,
            }

        answer = final_state.get("draft_answer", "")

        # ── Update conversation history (rolling window of 4 Q+A pairs) ──────
        history.append(Turn(role="user",      content=question))
        history.append(Turn(role="assistant", content=answer))
        # Keep only the last HISTORY_WINDOW pairs = HISTORY_WINDOW * 2 turns
        trimmed = history[-(HISTORY_WINDOW * 2):]

        return {
            "draft_answer":        answer,
            "citations":           final_state.get("citations", []),
            "verification_report": final_state.get("verification_report", ""),
            "updated_history":     trimmed,
        }
