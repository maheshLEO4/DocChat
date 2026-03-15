from langchain_core.documents import Document
from agents.base_agent import BaseAgent
from utils import get_logger
import re

logger = get_logger(__name__)


class ResearchAgent(BaseAgent):
    """
    Synthesises a draft answer from retrieved documents.
    Uses conversation history to handle follow-up questions correctly.
    Also performs query rewriting when history is present.
    """

    def __init__(self, model_provider: str | None = None, model_name: str | None = None):
        super().__init__(
            prompt_file="research.txt",
            temperature=0.1,
            max_tokens=600,
            model_provider=model_provider,
            model_name=model_name,
        )

        # Load query-rewrite prompt from same prompts/ directory
        from pathlib import Path
        rewrite_path = Path(__file__).parent / "prompts" / "query_rewrite.txt"
        self.rewrite_template = rewrite_path.read_text(encoding="utf-8")

    # ── Query rewriting ───────────────────────────────────────────────────────

    def rewrite_query(self, question: str, history: str) -> str:
        """
        Rewrite the question into a self-contained search query.
        Falls back to the original question on any error.
        """
        if not history or history == "(no prior conversation)":
            return question  # nothing to resolve

        prompt = self.rewrite_template.format(
            history=history,
            question=question,
        )
        try:
            rewritten = self._call_llm(prompt).strip().strip('"').strip("'")
            logger.info(f"Query rewrite: '{question}' → '{rewritten}'")
            return rewritten or question
        except Exception as exc:
            logger.warning(f"Query rewrite failed ({exc}), using original")
            return question

    # ── Answer generation ─────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        documents: list[Document],
        history: str,
    ) -> dict:
        """
        Generate a draft answer.

        Returns:
            {
                "draft_answer": str,
                "citations": list[str],   # unique source filenames
            }
        """
        if not documents:
            return {
                "draft_answer": "Sorry, I don't have enough information to answer that.",
                "citations": [],
            }

        context = "\n\n".join(
            f"[Source: {d.metadata.get('file_name', 'unknown')}]\n{d.page_content}"
            for d in documents
        )

        prompt = self.prompt_template.format(
            history=history or "(no prior conversation)",
            question=question,
            context=context,
        )

        try:
            answer = self._call_llm(prompt)
        except Exception as exc:
            logger.error(f"ResearchAgent LLM error: {exc}")
            return {
                "draft_answer": "Sorry, I encountered an error while generating the answer.",
                "citations": [],
            }

        citations = list({
            d.metadata["file_name"]
            for d in documents
            if d.metadata.get("file_name")
        })

        return {"draft_answer": answer, "citations": citations}
