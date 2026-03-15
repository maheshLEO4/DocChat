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


class HybridRetriever:
    """
    Hybrid dense+sparse retriever with RRF score fusion.

    Usage:
        retriever = HybridRetriever()
        docs = retriever.invoke("What is Mahesh's experience?")
    """

    def __init__(self):
        if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
            raise RuntimeError("No index found. Upload and index PDFs first.")

        configure_embedding()

        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        self.index = load_index_from_storage(storage)
        logger.info("Index loaded from storage")

        self.vector = get_vector_retriever(self.index)
        self.bm25   = get_bm25_retriever(self.index)

    def invoke(self, query: str) -> list[Document]:
        """
        Retrieve documents for *query* using RRF-fused hybrid search.

        Returns a list of LangChain Document objects.
        """
        try:
            vector_nodes = self.vector.retrieve(query)
            bm25_nodes   = self.bm25.retrieve(query)
        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            return []

        fused = reciprocal_rank_fusion([vector_nodes, bm25_nodes])

        return [
            Document(page_content=n.node.text, metadata=n.node.metadata or {})
            for n in fused
        ]
