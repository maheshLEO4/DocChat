import os
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    LLM_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
)
from utils import get_logger

logger = get_logger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


class BaseAgent:
    """
    Shared base for all LLM-backed agents.
    Loads a prompt template from agents/prompts/<name>.txt and
    provides a ChatGroq client.
    """

    def __init__(
        self,
        prompt_file: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        model_provider: str | None = None,
        model_name: str | None = None,
    ):
        provider = (model_provider or DEFAULT_PROVIDER).lower()
        model = model_name or DEFAULT_MODEL or LLM_MODEL

        if provider == "groq":
            if not GROQ_API_KEY:
                raise EnvironmentError(
                    "GROQ_API_KEY is not set. "
                    "Add it to your .env file or Streamlit secrets."
                )

            self.llm = ChatGroq(
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=GROQ_API_KEY,
            )
        elif provider == "gemini":
            if not GEMINI_API_KEY:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. "
                    "Add it to your .env file or Streamlit secrets."
                )

            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=GEMINI_API_KEY,
            )
        else:
            raise ValueError(f"Unknown model provider: {provider}")

        prompt_path = PROMPT_DIR / prompt_file
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        logger.info(
            f"{self.__class__.__name__} ready (provider={provider}, model={model})"
        )

    def _call_llm(self, prompt: str) -> str:
        response = self.llm.invoke(prompt)
        return response.content.strip()
