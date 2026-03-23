import os
import shutil
from llama_index.core import VectorStoreIndex
from config import BATCH_SIZE, get_index_dir
from utils import get_logger

logger = get_logger(__name__)

def build_index(nodes: list, collection_name: str, progress_callback=None) -> VectorStoreIndex:
    def _cb(p, m):
        if progress_callback:
            progress_callback(p, m)
        logger.info(m)

    index_dir = get_index_dir(collection_name)
    total = len(nodes)
    logger.info(f"Building index from {total} nodes")
    
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)

    _cb(0.1, f"Embedding {total} chunks...")

    index = VectorStoreIndex(
        nodes,
        show_progress=True,
        insert_batch_size=BATCH_SIZE,
    )

    _cb(0.9, "Persisting index to disk...")
    index.storage_context.persist(persist_dir=index_dir)
    _cb(1.0, f"Index built and saved ({total} chunks)")
    logger.info(f"Index persisted to {index_dir}")
    return index

def ingest_pdfs(collection_name: str, progress_callback=None):
    from ingestion.embedding import configure_embedding
    from ingestion.loader import load_pdfs
    from ingestion.splitter import split_documents

    def _cb(p, m):
        if progress_callback:
            progress_callback(p, m)
        logger.info(m)

    _cb(0.05, "Loading embedding model...")
    configure_embedding()

    _cb(0.10, "Loading PDF documents...")
    docs = load_pdfs(collection_name)

    _cb(0.25, f"Loaded {len(docs)} pages(s). Splitting into chunks...")
    nodes = split_documents(docs)
    total = len(nodes)

    _cb(0.35, f"{total} chunks ready. Embedding...")

    def _build_cb(p, m):
        _cb(0.35 + p * 0.60, m)

    build_index(nodes, collection_name, progress_callback=_build_cb)
    _cb(1.00, f"Done! Indexed {total} chunks.")
