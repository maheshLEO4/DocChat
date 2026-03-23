"""
ingestion/splitter.py
~~~~~~~~~~~~~~~~~~~~~
Splits LlamaIndex documents into nodes (chunks).

Fix: chunk_size=384 matches all-MiniLM-L6-v2's max token length exactly.
     The original 600-token chunks were silently truncated by the model,
     meaning the tail of each chunk was never embedded — causing retrieval
     to miss content that appeared in the latter half of large paragraphs.
"""

from llama_index.core.node_parser import SentenceSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import get_logger

logger = get_logger(__name__)


def split_documents(docs: list) -> list:
    """Split LlamaIndex documents into nodes."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,       # 384 — matches MiniLM max context
        chunk_overlap=CHUNK_OVERLAP, # 64 — preserves cross-boundary context
    )
    nodes = splitter.get_nodes_from_documents(docs, show_progress=True)
    logger.info(f"Split {len(docs)} doc(s) into {len(nodes)} chunk(s) "
                f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return nodes