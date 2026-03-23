from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import EMBED_MODEL, EMBED_BATCH_SIZE
from utils import get_logger

logger = get_logger(__name__)


def configure_embedding():
    """
    Configure the global LlamaIndex embedding model.
    Call once before building or loading any index.
    """
    Settings.llm = None  # disable OpenAI LLM shim
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL, 
        embed_batch_size=EMBED_BATCH_SIZE
    )
    logger.info(f"Embedding model set to '{EMBED_MODEL}' with batch size {EMBED_BATCH_SIZE}")
