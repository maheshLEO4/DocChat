from typing import Any, TypedDict
from langchain_core.documents import Document


# ─────────────────────────────────────────────────────────────────────────────
# Conversation turn — stored in history
# ─────────────────────────────────────────────────────────────────────────────
class Turn(TypedDict):
    role: str    # "user" | "assistant"
    content: str


# ─────────────────────────────────────────────────────────────────────────────
# Shared state that flows through every LangGraph node
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    question: str                    # raw user question (current turn)
    rewritten_query: str             # query after rewriting for retrieval
    conversation_history: list[Turn] # last N turns (user + assistant pairs)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    documents: list[Document]        # chunks returned by hybrid retriever

    # ── Agent outputs ─────────────────────────────────────────────────────────
    is_relevant: bool                # set by relevance node
    draft_answer: str                # set by research node
    citations: list[str]             # source filenames extracted from docs
    verification_report: str         # set by verification node

    # ── Control ───────────────────────────────────────────────────────────────
    retriever: Any                   # HybridRetriever instance (passed through)
    iteration_count: int             # tracks research→verify loops
    enable_verification: bool        # toggle slower verification path
    model_provider: str              # "groq" | "gemini"
    model_name: str                  # selected model name
