from llama_index.core.node_parser import SentenceSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils import get_logger

logger = get_logger(__name__)


def split_documents(docs: list) -> list:
    """Split LlamaIndex documents into nodes (chunks)."""
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents(docs)
    logger.info(f"Split into {len(nodes)} chunk(s)")
    return nodes
