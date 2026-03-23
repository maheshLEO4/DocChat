import os
import re

import fitz  # PyMuPDF
from llama_index.core import Document, SimpleDirectoryReader

from config import get_upload_dir
from utils import get_logger

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
    """Load PDFs from a collection using PyMuPDF; fall back to SimpleDirectoryReader if needed."""
    docs = []
    upload_dir = get_upload_dir(collection_name)
    pdf_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]
    errors = []

    for filename in pdf_files:
        filepath = os.path.join(upload_dir, filename)
        try:
            # PyMuPDF is preferred for reliable text extraction.
            with fitz.open(filepath) as doc:
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text("text") or ""
                    text = clean_text(text)

                    if text:
                        docs.append(
                            Document(
                                text=text,
                                metadata={"source": filename, "page": page_num + 1},
                            )
                        )
        except Exception as exc:
            errors.append(f"{filename}: {exc}")
            logger.error(f"Error loading {filename}: {exc}")

    # Fallback to LlamaIndex reader if PyMuPDF produced no text.
    if not docs and pdf_files:
        try:
            reader = SimpleDirectoryReader(
                upload_dir,
                required_exts=[".pdf"],
                recursive=False,
            )
            docs = reader.load_data()
        except Exception as exc:
            errors.append(f"fallback: {exc}")

    if docs:
        total_chars = sum(len(d.text) for d in docs)
        logger.info(
            f"Loaded {len(docs)} page(s) from {upload_dir}. Total characters: {total_chars}"
        )

        # Guard against huge duplicate extracts (often from PDF overlay artifacts).
        if total_chars > 5_000_000:
            raise RuntimeError(
                "Extracted text is unusually large. The PDF may contain duplicated text layers. "
                "Try splitting the PDF or using a cleaner copy."
            )

    if not docs:
        if not pdf_files:
            raise RuntimeError("No PDF files found in the collection upload folder.")
        if errors:
            raise RuntimeError(
                "PDFs were found but no extractable text was produced. "
                "If the PDFs are scanned images, run OCR first. "
                f"Errors: {errors[:3]}"
            )
        raise RuntimeError(
            "PDFs were found but no extractable text was produced. "
            "If the PDFs are scanned images, run OCR first."
        )

    return docs
