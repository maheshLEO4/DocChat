import os
from llama_index.core import StorageContext, load_index_from_storage
from langchain_core.documents import Document

from ingestion.embedding import configure_embedding
from retriever.vector_retriever import get_vector_retriever
from retriever.bm25_retriever import get_bm25_retriever
from retriever.fusion import reciprocal_rank_fusion
from config import get_index_dir
from utils import get_logger

logger = get_logger(__name__)

_FILENAME_KEYS = ("file_name", "file_path", "filename", "source")

def _extract_filename(metadata: dict) -> str:
    for key in _FILENAME_KEYS:
        val = metadata.get(key)
        if val:
            return os.path.basename(str(val))
    return "unknown"

class HybridRetriever:
    def __init__(self, collection_name: str):
        index_dir = get_index_dir(collection_name)
        if not os.path.exists(index_dir) or not os.listdir(index_dir):
            raise RuntimeError(
                f"No index found for collection '{collection_name}'. Upload and index PDFs first."
            )

        configure_embedding()

        storage = StorageContext.from_defaults(persist_dir=index_dir)
        self.index = load_index_from_storage(storage)
        logger.info(f"Index loaded from storage: {index_dir}")

        self.vector = get_vector_retriever(self.index)
        self.bm25   = get_bm25_retriever(self.index)  # may be None

        if self.bm25 is None:
            logger.warning("Running in vector-only mode.")

    def invoke(self, query: str) -> list[Document]:
        try:
            vector_nodes = self.vector.retrieve(query)
        except Exception as exc:
            logger.error(f"Vector retrieval error: {exc}")
            return []

        if self.bm25 is not None:
            try:
                bm25_nodes = self.bm25.retrieve(query)
                fused = reciprocal_rank_fusion([vector_nodes, bm25_nodes])      
            except Exception as exc:
                logger.warning(f"BM25 retrieval error ({exc}) using vector only")                                                                                           
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

    retrieve = invoke
