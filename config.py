import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(BASE_DIR, "data")
UPLOAD_DIR  = os.path.join(DATA_DIR, "raw_pdfs")
INDEX_DIR   = os.path.join(DATA_DIR, "llamaindex")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K         = 6      # docs returned per retriever before fusion
FINAL_TOP_K   = 5      # docs kept after RRF fusion
RRF_K         = 60     # RRF constant (higher = smoother rank blending)

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 256
CHUNK_OVERLAP = 25
BATCH_SIZE    = 1000   # nodes per indexing batch for large PDFs

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
LLM_MODEL     = "llama-3.1-8b-instant"

# ── Workflow ──────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 2     # max research→verify loops before forcing end
