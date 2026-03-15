import os
from pathlib import Path
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL
from utils import get_logger

logger = get_logger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


class BaseAgent:
    """
    Shared base for all LLM-backed agents.
    Loads a prompt template from agents/prompts/<name>.txt and
    provides a ChatGroq client.
    """

    def __init__(self, prompt_file: str, temperature: float = 0.0, max_tokens: int = 512):
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or Streamlit secrets."
            )

        self.llm = ChatGroq(
            model_name=LLM_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            groq_api_key=GROQ_API_KEY,
        )

        prompt_path = PROMPT_DIR / prompt_file
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        logger.info(f"{self.__class__.__name__} ready (model={LLM_MODEL})")

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content.strip()
