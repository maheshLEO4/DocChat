"""
retriever/bm25_retriever.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sparse BM25 retriever.

Fixes:
- from_defaults(index=index) was loading ALL nodes into RAM via an
  internal retrieve-all call. Replaced with from_defaults(nodes=nodes)
  which reads the docstore directly — same data, no extra round-trip.
- Returns None on failure so hybrid_retriever degrades to vector-only
  instead of crashing (rank-bm25 not installed, empty docstore, etc).

requirements.txt must include:
    rank-bm25>=0.2.2   ← was missing; BM25Retriever depends on it
"""

from llama_index.retrievers.bm25 import BM25Retriever
from config import TOP_K
from utils import get_logger

logger = get_logger(__name__)


def get_bm25_retriever(index) -> "BM25Retriever | None":
    """
    Return a BM25 retriever over *index*, or None if setup fails.
    """
    try:
        nodes = list(index.docstore.docs.values())
        if not nodes:
            logger.warning("Docstore is empty — BM25 skipped")
            return None

        retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=TOP_K,
        )
        logger.info(f"BM25 retriever ready over {len(nodes)} nodes (top_k={TOP_K})")
        return retriever

    except Exception as exc:
        logger.error(
            f"BM25 init failed: {exc}. "
            "Ensure rank-bm25>=0.2.2 is in requirements.txt. "
            "Falling back to vector-only retrieval."
        )
        return None