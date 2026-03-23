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


def _dedupe_lines(text: str) -> str:
    """Remove duplicate lines while preserving first occurrence order."""
    seen = set()
    deduped = []
    for line in text.splitlines():
        key = line.strip()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)
    return "\n".join(deduped)


def load_pdfs() -> list:
    """Load PDFs from the shared upload folder using PyMuPDF; fall back to SimpleDirectoryReader if needed."""
    docs = []
    upload_dir = get_upload_dir()
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

        # Auto-clean when extracts look suspiciously large.
        if total_chars > 5_000_000:
            logger.warning(
                "Extracted text is unusually large. Attempting auto-clean by de-duplicating lines."
            )
            cleaned_docs = []
            for doc in docs:
                cleaned_docs.append(
                    Document(
                        text=_dedupe_lines(doc.text),
                        metadata=getattr(doc, "metadata", None),
                    )
                )
            docs = cleaned_docs
            total_chars = sum(len(d.text) for d in docs)
            logger.info(
                f"Post-clean character count: {total_chars}"
            )

    if not docs:
        if not pdf_files:
            raise RuntimeError("No PDF files found in the upload folder.")
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
