import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR    = os.getenv("APP_DATA_DIR", os.path.join(BASE_DIR, "data"))
UPLOAD_DIR  = os.path.join(DATA_DIR, "raw_pdfs")
INDEX_DIR   = os.path.join(DATA_DIR, "llamaindex")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR,  exist_ok=True)

# ── Embedding ────────────────────────────────────────────────────────────────
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K         = 6      # docs returned per retriever before fusion
FINAL_TOP_K   = 5      # docs kept after RRF fusion
RRF_K         = 60     # RRF constant (higher = smoother rank blending)

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 768
CHUNK_OVERLAP = 80
BATCH_SIZE    = 256    # nodes per indexing batch for large PDFs

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL     = "llama-3.1-8b-instant"

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
DEFAULT_MODEL = GROQ_FREE_MODELS[0]

# ── Workflow ──────────────────────────────────────────────────────────────────
MAX_ITERATIONS = 3     # max research→verify loops before forcing end
