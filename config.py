import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR   = os.getenv("APP_DATA_DIR", os.path.join(BASE_DIR, "data"))
UPLOAD_DIR = os.path.join(DATA_DIR, "raw_pdfs")
INDEX_DIR  = os.path.join(DATA_DIR, "llamaindex")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)

# ── Embedding ─────────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 22 MB, 384-dim, fast on CPU.
# HF_HOME → /data so the model is cached on the persistent disk in HF Spaces
# and NOT re-downloaded on every cold start.
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 32   # safe for 2-vCPU HF Spaces free tier

# ── Chunking ──────────────────────────────────────────────────────────────────
# 384 tokens = MiniLM's max context length → no truncation, best embeddings.
# Smaller chunks = more precise retrieval (less irrelevant text per chunk).
CHUNK_SIZE    = 384
CHUNK_OVERLAP = 64

# ── Indexing ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64   # VectorStoreIndex insert_batch_size

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K       = 8   # candidates per retriever before RRF fusion
FINAL_TOP_K = 5   # docs sent to agents after fusion
RRF_K       = 60

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL      = "llama-3.1-8b-instant"

GROQ_FREE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]
GEMINI_FREE_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL    = GROQ_FREE_MODELS[0]

# ── Workflow ──────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 3