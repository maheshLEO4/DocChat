from llama_index.core import VectorStoreIndex

from config import BATCH_SIZE, INDEX_DIR
from utils import get_logger

logger = get_logger(__name__)


def build_index(nodes: list) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from nodes.
    Processes in batches for large corpora.
    """
    total = len(nodes)

    if total <= BATCH_SIZE:
        logger.info(f"Building index from {total} node(s) in one pass")
        index = VectorStoreIndex(nodes)
    else:
        logger.info(f"Large corpus ({total} nodes) - batching in groups of {BATCH_SIZE}")
        index = VectorStoreIndex(nodes[:BATCH_SIZE])
        for start in range(BATCH_SIZE, total, BATCH_SIZE):
            batch = nodes[start : start + BATCH_SIZE]
            index.insert_nodes(batch)
            done = min(start + BATCH_SIZE, total)
            logger.info(f"  indexed {done}/{total} nodes")

    index.storage_context.persist(persist_dir=INDEX_DIR)
    logger.info(f"Index persisted to {INDEX_DIR}")
    return index


def ingest_pdfs(progress_callback=None):
    """
    Full ingestion pipeline: load -> split -> embed -> index.

    Args:
        progress_callback: optional (progress: float, message: str) callable
    """
    from ingestion.embedding import configure_embedding
    from ingestion.loader import load_pdfs
    from ingestion.splitter import split_documents

    def _cb(progress, message):
        if progress_callback:
            progress_callback(progress, message)
        logger.info(message)

    _cb(0.05, "Configuring embedding model...")
    configure_embedding()

    _cb(0.10, "Loading PDF documents...")
    docs = load_pdfs()
    if not docs:
        raise RuntimeError("No PDF documents found in upload directory.")

    _cb(0.30, f"Loaded {len(docs)} document(s). Splitting into chunks...")
    nodes = split_documents(docs)
    total = len(nodes)

    _cb(0.50, f"Created {total} chunk(s). Building vector index...")
    build_index(nodes)

    _cb(1.00, f"Indexed {total} chunks successfully!")
