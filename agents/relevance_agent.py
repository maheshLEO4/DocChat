from langchain_core.documents import Document
from agents.base_agent import BaseAgent
from utils import get_logger

logger = get_logger(__name__)

VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}


class RelevanceAgent(BaseAgent):
    """
    Classifies whether retrieved passages can answer the user's question,
    taking conversation history into account.
    """

    def __init__(self, model_provider: str | None = None, model_name: str | None = None):
        super().__init__(
            prompt_file="relevance.txt",
            temperature=0.0,
            max_tokens=10,
            model_provider=model_provider,
            model_name=model_name,
        )

    def check(self, question: str, documents: list[Document], history: str) -> str:
        """
        Returns one of: "CAN_ANSWER", "PARTIAL", "NO_MATCH".

        Args:
            question:  current user question (already rewritten for retrieval)
            documents: top retrieved document chunks
            history:   formatted conversation history string
        """
        if not documents:
            logger.debug("No documents — returning NO_MATCH")
            return "NO_MATCH"

        passages = "\n\n".join(d.page_content for d in documents)

        prompt = self.prompt_template.format(
            history=history or "(no prior conversation)",
            question=question,
            passages=passages,
        )

        try:
            raw = self._call_llm(prompt).upper()
        except Exception as exc:
            logger.error(f"RelevanceAgent LLM error: {exc}")
            return "NO_MATCH"

        # Accept any response that contains a valid label
        for label in VALID_LABELS:
            if label in raw:
                logger.info(f"Relevance → {label}")
                return label

        logger.warning(f"Unexpected relevance response '{raw}' — defaulting to NO_MATCH")
        return "NO_MATCH"
