"""
ingestion/embedding.py
~~~~~~~~~~~~~~~~~~~~~~
Configures the global LlamaIndex embedding model and pre-warms it.

HF Spaces fixes:
- HF_HOME is set in Dockerfile → /data/hf_cache (persistent disk).
  The model is downloaded once and reused across restarts.
- _warm_up() runs a dummy encode after loading so the first real
  batch doesn't pay the JIT / tokenizer init cost during indexing.
"""

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import EMBED_MODEL, EMBED_BATCH_SIZE
from utils import get_logger

logger = get_logger(__name__)

# Module-level singleton so configure_embedding() is idempotent
_embed_model = None


def configure_embedding():
    """
    Set the global LlamaIndex embedding model.
    Safe to call multiple times — only initialises once per process.
    """
    global _embed_model

    if _embed_model is not None:
        # Already initialised in this process — reuse, don't reload
        Settings.embed_model = _embed_model
        logger.info("Embedding model reused from cache (no reload)")
        return

    logger.info(f"Loading embedding model '{EMBED_MODEL}'…")
    Settings.llm = None  # disable OpenAI LLM shim

    _embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    Settings.embed_model = _embed_model

    # Pre-warm: run one dummy encode so the first real batch doesn't
    # pay tokenizer JIT cost during the timed indexing window.
    try:
        _embed_model.get_text_embedding("warm up")
        logger.info("Embedding model warmed up successfully")
    except Exception as exc:
        logger.warning(f"Warm-up encode failed (non-fatal): {exc}")

    logger.info(
        f"Embedding model ready: '{EMBED_MODEL}', batch_size={EMBED_BATCH_SIZE}"
    )