from llama_index.retrievers.bm25 import BM25Retriever
from config import TOP_K
from utils import get_logger

logger = get_logger(__name__)


def get_bm25_retriever(index) -> BM25Retriever:
    """Return a sparse BM25 retriever over the given index."""
    retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=TOP_K)
    logger.info(f"BM25 retriever ready (top_k={TOP_K})")
    return retriever
