from llama_index.core.retrievers import VectorIndexRetriever
from config import TOP_K
from utils import get_logger

logger = get_logger(__name__)


def get_vector_retriever(index) -> VectorIndexRetriever:
    """Return a dense vector retriever over the given index."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=TOP_K)
    logger.info(f"Vector retriever ready (top_k={TOP_K})")
    return retriever
