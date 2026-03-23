from llama_index.core import SimpleDirectoryReader
from config import UPLOAD_DIR
from utils import get_logger

logger = get_logger(__name__)


def load_pdfs() -> list:
    """Load all PDFs from UPLOAD_DIR and return LlamaIndex Document objects."""
    docs = SimpleDirectoryReader(
        UPLOAD_DIR,
        required_exts=[".pdf"],
        filename_as_id=True,
        recursive=False
    ).load_data()

    logger.info(f"Loaded {len(docs)} document(s) from {UPLOAD_DIR}")
    return docs
