import hashlib
import json
import os
import shutil

from llama_index.core import VectorStoreIndex

from config import BATCH_SIZE, INDEX_DIR, UPLOAD_DIR
from utils import get_logger

logger = get_logger(__name__)
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest() -> dict:
    files = []
    for name in sorted(os.listdir(UPLOAD_DIR)):
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path) or not name.lower().endswith(".pdf"):
            continue
        files.append(
            {
                "name": name,
                "size": os.path.getsize(path),
                "sha256": _hash_file(path),
            }
        )
    return {"files": files}


def _load_manifest() -> dict | None:
    if not os.path.exists(MANIFEST_PATH):
        return None
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning(f"Failed to read index manifest: {exc}")
        return None


def _save_manifest(manifest: dict) -> None:
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _clear_index_dir() -> None:
    for entry in os.listdir(INDEX_DIR):
        path = os.path.join(INDEX_DIR, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def build_index(nodes: list, progress_callback=None) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from nodes.
    Process large corpora in batches and report progress after each batch.
    """
    total = len(nodes)
    if total == 0:
        raise RuntimeError("No nodes were generated for indexing.")

    logger.info(f"Building index from {total} node(s) in batches of {BATCH_SIZE}")
    index = None

    for start in range(0, total, BATCH_SIZE):
        batch = nodes[start : start + BATCH_SIZE]
        if index is None:
            index = VectorStoreIndex(batch)
        else:
            index.insert_nodes(batch)

        done = min(start + len(batch), total)
        logger.info(f"  indexed {done}/{total} nodes")
        if progress_callback:
            progress_callback(done, total)

    assert index is not None
    index.storage_context.persist(persist_dir=INDEX_DIR)
    logger.info(f"Index persisted to {INDEX_DIR}")
    return index


def ingest_pdfs(progress_callback=None):
    """
    Full ingestion pipeline: load -> split -> embed -> index.

    Args:
        progress_callback: optional (progress: float, message: str) callable
    """
    from ingestion.embedding import configure_embedding
    from ingestion.loader import load_pdfs
    from ingestion.splitter import split_documents

    def _cb(progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(progress, message)
        logger.info(message)

    manifest = _build_manifest()
    if not manifest["files"]:
        raise RuntimeError("No PDF documents found in upload directory.")

    existing_manifest = _load_manifest()
    if existing_manifest == manifest and os.listdir(INDEX_DIR):
        _cb(1.00, "PDF set unchanged. Reusing existing index.")
        return

    _cb(0.05, "Configuring embedding model...")
    configure_embedding()

    _cb(0.10, "Loading PDF documents...")
    docs = load_pdfs()
    if not docs:
        raise RuntimeError("No PDF documents found in upload directory.")

    _cb(0.30, f"Loaded {len(docs)} document(s). Splitting into chunks...")
    nodes = split_documents(docs)
    total = len(nodes)

    _cb(0.45, "Refreshing stored index...")
    _clear_index_dir()

    _cb(0.50, f"Created {total} chunk(s). Building vector index...")
    build_index(
        nodes,
        progress_callback=lambda done, overall: _cb(
            0.50 + (0.45 * (done / overall)),
            f"Building vector index... {done}/{overall} chunks processed",
        ),
    )
    _save_manifest(manifest)

    _cb(1.00, f"Indexed {total} chunks successfully.")
