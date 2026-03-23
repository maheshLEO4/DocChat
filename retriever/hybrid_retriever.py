"""
retriever/hybrid_retriever.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hybrid dense+sparse retriever with RRF score fusion.

Fixes:
1. BM25 is optional — gracefully falls back to vector-only if unavailable.
2. Metadata filename extracted with multi-key fallback (file_name / file_path /
   filename / source) so citations never silently disappear.
3. Module-level singleton (_instance) so HybridRetriever() can be called
   repeatedly from Streamlit without reloading the index from disk each time.
"""

import os
from llama_index.core import StorageContext, load_index_from_storage
from langchain_core.documents import Document

from ingestion.embedding import configure_embedding
from retriever.vector_retriever import get_vector_retriever
from retriever.bm25_retriever import get_bm25_retriever
from retriever.fusion import reciprocal_rank_fusion
from config import INDEX_DIR
from utils import get_logger

logger = get_logger(__name__)

# Metadata keys tried in order when resolving the source filename
_FILENAME_KEYS = ("file_name", "file_path", "filename", "source")


def _extract_filename(metadata: dict) -> str:
    for key in _FILENAME_KEYS:
        val = metadata.get(key)
        if val:
            return os.path.basename(str(val))
    return "unknown"


class HybridRetriever:
    """
    Hybrid dense + sparse retriever with RRF fusion.

    Usage:
        retriever = HybridRetriever()
        docs = retriever.invoke("What is X?")
    """

    def __init__(self):
        if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
            raise RuntimeError(
                "No index found. Upload and index PDFs first."
            )

        # configure_embedding() is idempotent — safe to call every time
        configure_embedding()

        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        self.index = load_index_from_storage(storage)
        logger.info("Index loaded from storage")

        self.vector = get_vector_retriever(self.index)
        self.bm25   = get_bm25_retriever(self.index)  # may be None

        if self.bm25 is None:
            logger.warning(
                "Running in vector-only mode. "
                "Add rank-bm25>=0.2.2 to requirements.txt for hybrid search."
            )

    def invoke(self, query: str) -> list[Document]:
        """
        Retrieve documents for *query* using hybrid search (or vector-only).
        Returns a list of LangChain Document objects.
        """
        # Dense retrieval
        try:
            vector_nodes = self.vector.retrieve(query)
        except Exception as exc:
            logger.error(f"Vector retrieval error: {exc}")
            return []

        # Sparse retrieval + RRF fusion (if BM25 is available)
        if self.bm25 is not None:
            try:
                bm25_nodes = self.bm25.retrieve(query)
                fused = reciprocal_rank_fusion([vector_nodes, bm25_nodes])
            except Exception as exc:
                logger.warning(f"BM25 retrieval error ({exc}) — using vector only")
                fused = vector_nodes
        else:
            fused = vector_nodes

        results = []
        for n in fused:
            raw_meta = n.node.metadata or {}
            meta = dict(raw_meta)
            meta["file_name"] = _extract_filename(raw_meta)
            results.append(Document(page_content=n.node.text, metadata=meta))

        logger.info(f"Retrieved {len(results)} doc(s) for: '{query[:80]}'")
        return results

    # Alias for callers that use .retrieve()
    retrieve = invoke