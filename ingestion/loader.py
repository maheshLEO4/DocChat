import os
import fitz  # PyMuPDF
from llama_index.core import Document
from config import get_upload_dir
from utils import get_logger
import re

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """Basic cleanup to prevent OCR/pdf shadow text blowups."""
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with maximum 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing
    return text.strip()


def load_pdfs(collection_name: str) -> list:
    """Load all PDFs from the collection's upload dir using PyMuPDF and return LlamaIndex Document objects.""" 
    docs = []
    upload_dir = get_upload_dir(collection_name)
    
    for filename in os.listdir(upload_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(upload_dir, filename)
            try:
                # Use PyMuPDF (fitz) - much better at preventing overlapping text bugs than pypdf
                with fitz.open(filepath) as doc:
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text = page.get_text("text") or ""
                        
                        text = clean_text(text)
                        
                        if text:
                            # Create a Document for each page
                            docs.append(
                                Document(
                                    text=text,
                                    metadata={"source": filename, "page": page_num + 1}
                                )
                            )
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    if docs:
        total_chars = sum(len(d.text) for d in docs)
        logger.info(f"Loaded {len(docs)} page(s) from {upload_dir}. Total characters: {total_chars}")
        
    return docs
