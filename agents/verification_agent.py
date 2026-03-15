from langchain_core.documents import Document
from agents.base_agent import BaseAgent
from utils import get_logger

logger = get_logger(__name__)


class VerificationAgent(BaseAgent):
    """
    Checks whether the draft answer is grounded in the retrieved documents.
    """

    def __init__(self):
        super().__init__(prompt_file="verification.txt", temperature=0.0, max_tokens=220)

    def check(self, answer: str, documents: list[Document]) -> dict:
        """
        Verify *answer* against *documents*.

        Returns a formatted string report.
        """
        if not documents:
            return {"verification_report": self._default_report("No documents available.")}

        context = "\n\n".join(d.page_content for d in documents)
        prompt = self.prompt_template.format(answer=answer, context=context)

        try:
            raw = self._call_llm(prompt)
        except Exception as exc:
            logger.error(f"VerificationAgent LLM error: {exc}")
            return {"verification_report": self._default_report(f"Model error: {exc}")}

        parsed = self._parse(raw)
        report = self._format(parsed)
        return {"verification_report": report}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse(self, text: str) -> dict:
        result = {
            "Supported": "NO",
            "Unsupported Claims": [],
            "Contradictions": [],
            "Relevant": "NO",
            "Additional Details": "",
        }
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip().lower(), value.strip()

            if key == "supported":
                result["Supported"] = "YES" if "yes" in value.lower() else "NO"
            elif key == "unsupported claims":
                result["Unsupported Claims"] = self._parse_list(value)
            elif key == "contradictions":
                result["Contradictions"] = self._parse_list(value)
            elif key == "relevant":
                result["Relevant"] = "YES" if "yes" in value.lower() else "NO"
            elif key == "additional details":
                result["Additional Details"] = value
        return result

    @staticmethod
    def _parse_list(value: str) -> list[str]:
        value = value.strip("[]")
        if not value:
            return []
        return [item.strip().strip("'\"") for item in value.split(",") if item.strip()]

    @staticmethod
    def _format(v: dict) -> str:
        lines = [
            f"**Supported:** {v['Supported']}",
            f"**Unsupported Claims:** {', '.join(v['Unsupported Claims']) or 'None'}",
            f"**Contradictions:** {', '.join(v['Contradictions']) or 'None'}",
            f"**Relevant:** {v['Relevant']}",
            f"**Additional Details:** {v['Additional Details'] or 'None'}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _default_report(reason: str) -> str:
        return (
            f"**Supported:** NO\n"
            f"**Unsupported Claims:** None\n"
            f"**Contradictions:** None\n"
            f"**Relevant:** NO\n"
            f"**Additional Details:** {reason}"
        )
