"""
ingestion/index_builder.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Builds and persists a VectorStoreIndex from LlamaIndex nodes.

Fixes:
- Wipes INDEX_DIR before rebuilding so re-uploads never mix stale
  and new vectors (was the cause of wrong/irrelevant retrieval results).
- insert_batch_size=BATCH_SIZE keeps peak RAM bounded on HF Spaces.
- Progress callback covers the full 0→1 range so the Streamlit bar
  never appears frozen.
"""

import os
import shutil

from llama_index.core import VectorStoreIndex

from config import BATCH_SIZE, INDEX_DIR
from utils import get_logger

logger = get_logger(__name__)


def build_index(nodes: list, progress_callback=None) -> VectorStoreIndex:
    """
    Build and persist a VectorStoreIndex from *nodes*.

    Args:
        nodes:             LlamaIndex TextNode list
        progress_callback: optional (float, str) callable for UI updates
    """
    def _cb(p, m):
        if progress_callback:
            progress_callback(p, m)
        logger.info(m)

    total = len(nodes)
    logger.info(f"Building index from {total} nodes (insert_batch_size={BATCH_SIZE})")

    # Always wipe old index — stale vectors from previous uploads cause
    # irrelevant retrieval results on re-indexing.
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR, exist_ok=True)

    _cb(0.1, f"Embedding {total} chunks… (this takes the longest on CPU)")

    index = VectorStoreIndex(
        nodes,
        show_progress=True,
        insert_batch_size=BATCH_SIZE,
    )

    _cb(0.9, "Persisting index to disk…")
    index.storage_context.persist(persist_dir=INDEX_DIR)

    _cb(1.0, f"Index built and saved ({total} chunks)")
    logger.info(f"Index persisted to {INDEX_DIR}")
    return index


def ingest_pdfs(progress_callback=None):
    """
    Full ingestion pipeline: configure → load → split → embed → index.

    Progress milestones:
        0.05  configuring embedding model
        0.10  loading PDFs
        0.25  splitting into chunks
        0.35  starting index build  (slow — embedding all chunks on CPU)
        0.95  persisting to disk
        1.00  done
    """
    from ingestion.embedding import configure_embedding
    from ingestion.loader import load_pdfs
    from ingestion.splitter import split_documents

    def _cb(p, m):
        if progress_callback:
            progress_callback(p, m)
        logger.info(m)

    _cb(0.05, "Loading embedding model (cached after first run)…")
    configure_embedding()

    _cb(0.10, "Loading PDF documents…")
    docs = load_pdfs()
    if not docs:
        raise RuntimeError("No PDF documents found in the upload directory.")

    _cb(0.25, f"Loaded {len(docs)} document(s). Splitting into chunks…")
    nodes = split_documents(docs)
    total = len(nodes)

    _cb(0.35, f"{total} chunks ready. Embedding on CPU — please wait…")

    # Scale build progress into the 0.35 → 0.95 band
    def _build_cb(p, m):
        _cb(0.35 + p * 0.60, m)

    build_index(nodes, progress_callback=_build_cb)

    _cb(1.00, f"✅ Done! Indexed {total} chunks.")