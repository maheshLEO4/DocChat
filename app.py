import os
import streamlit as st

from ingestion import ingest_pdfs
from retriever import HybridRetriever
from graph import AgentWorkflow, Turn
from config import (
    UPLOAD_DIR,
    INDEX_DIR,
    GROQ_FREE_MODELS,
    GEMINI_FREE_MODELS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    GROQ_API_KEY,
    GEMINI_API_KEY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title("📚 Multi-Agent Hybrid RAG Chatbot")

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    # Each item: {"user": str, "assistant": str, "citations": list, "verification": str}
    "chat_history":          [],
    # Rolling window of Turn dicts passed into the graph
    "conversation_history":  [],
    "retriever":             None,
    "files_indexed":         False,
    "uploaded_file_names":   set(),
    "model_provider":        DEFAULT_PROVIDER,
    "model_name":            DEFAULT_MODEL,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    enable_verification = st.checkbox(
        "Enable Verification",
        value=False,
        help="🐌 Slower but validates answer accuracy. ⚡ Disable for 3× faster responses.",
    )
    st.info(
        "⚡ **Fast Mode** (default): ~2–3 s\n\n"
        "🔍 **Verification Mode**: ~6–10 s — checks answer quality"
    )
    st.divider()
    st.subheader("Model")

    provider_labels = ["Groq", "Gemini"]
    provider_index = 0 if st.session_state.model_provider == "groq" else 1
    provider_label = st.selectbox("Provider", provider_labels, index=provider_index)
    model_provider = provider_label.lower()

    model_options = GROQ_FREE_MODELS if model_provider == "groq" else GEMINI_FREE_MODELS
    if st.session_state.model_name not in model_options:
        st.session_state.model_name = model_options[0]

    model_name = st.selectbox(
        "Model",
        model_options,
        index=model_options.index(st.session_state.model_name),
    )

    st.session_state.model_provider = model_provider
    st.session_state.model_name = model_name
    st.divider()
    st.caption("Conversation memory: last **4** Q&A pairs")

# ─────────────────────────────────────────────────────────────────────────────
# PDF upload & indexing
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📄 Upload Documents")
st.info("💡 **Tip**: For large PDFs (>50 MB / 200+ pages), split them first for faster processing.")

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_names = {f.name for f in uploaded_files}
    if current_names != st.session_state.uploaded_file_names:
        st.session_state.files_indexed = False
        st.session_state.uploaded_file_names = current_names
        # New files → reset retriever and conversation so history is not stale
        st.session_state.retriever = None
        st.session_state.conversation_history = []

    if not st.session_state.files_indexed:
        total_mb = sum(f.size for f in uploaded_files) / (1024 * 1024)
        if total_mb > 50:
            st.warning(f"⚠️ Large upload ({total_mb:.1f} MB) — indexing may take a few minutes.")

        if st.button("📑 Index PDFs", type="primary", use_container_width=True):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            incoming_names = {f.name for f in uploaded_files}
            for name in os.listdir(UPLOAD_DIR):
                path = os.path.join(UPLOAD_DIR, name)
                if os.path.isfile(path) and name.lower().endswith(".pdf") and name not in incoming_names:
                    os.remove(path)
            for f in uploaded_files:
                dest = os.path.join(UPLOAD_DIR, f.name)
                with open(dest, "wb") as fh:
                    fh.write(f.getbuffer())

            progress_bar = st.progress(0)
            status_text  = st.empty()

            try:
                ingest_pdfs(
                    progress_callback=lambda p, m: (
                        progress_bar.progress(p), status_text.text(m)
                    )
                )
                st.session_state.files_indexed = True
                st.session_state.retriever = None          # force reload
                progress_bar.empty()
                status_text.empty()
                st.success("✅ PDFs indexed! You can now ask questions.")
                st.rerun()
            except MemoryError:
                progress_bar.empty(); status_text.empty()
                st.error("❌ File too large — split the PDF into smaller parts.")
                st.session_state.files_indexed = False
            except Exception as exc:
                progress_bar.empty(); status_text.empty()
                st.error(f"❌ Indexing failed: {exc}")
                st.session_state.files_indexed = False
    else:
        st.success(f"✅ {len(uploaded_files)} file(s) indexed.")
        if st.button("🔄 Re-index PDFs"):
            st.session_state.files_indexed = False
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Render existing chat history
# ─────────────────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["assistant"])

    if msg.get("citations"):
        st.caption("📎 Sources: " + " · ".join(f"`{c}`" for c in msg["citations"]))

    if msg.get("verification"):
        with st.expander("🔍 Verification Report", expanded=False):
            st.markdown(msg["verification"])

# ─────────────────────────────────────────────────────────────────────────────
# Chat input
# ─────────────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about your uploaded PDFs…")

if question:
    if st.session_state.model_provider == "groq" and not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your secrets or .env file.")
        st.stop()
    if st.session_state.model_provider == "gemini" and not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY is not set. Add it to your secrets or .env file.")
        st.stop()

    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        st.warning("⚠️ Please upload and index PDFs first.")
        st.stop()

    # Lazy-load retriever (cached across turns)
    if st.session_state.retriever is None:
        with st.spinner("Loading retriever…"):
            st.session_state.retriever = HybridRetriever()

    workflow = AgentWorkflow(enable_verification=enable_verification)

    spinner_msg = "🤔 Thinking…" if not enable_verification else "🤔 Thinking and verifying…"
    with st.spinner(spinner_msg):
        result = workflow.run(
            question=question,
            retriever=st.session_state.retriever,
            conversation_history=st.session_state.conversation_history,
            model_provider=st.session_state.model_provider,
            model_name=st.session_state.model_name,
        )

    # ── Persist updated history window back to session ────────────────────
    st.session_state.conversation_history = result["updated_history"]

    # ── Save full display record ──────────────────────────────────────────
    st.session_state.chat_history.append({
        "user":         question,
        "assistant":    result["draft_answer"],
        "citations":    result.get("citations", []),
        "verification": result.get("verification_report", ""),
    })

    # ── Render current turn ───────────────────────────────────────────────
    st.chat_message("user").write(question)
    st.chat_message("assistant").write(result["draft_answer"])

    if result.get("citations"):
        st.caption("📎 Sources: " + " · ".join(f"`{c}`" for c in result["citations"]))

    if result.get("verification_report"):
        with st.expander("🔍 Verification Report", expanded=False):
            st.markdown(result["verification_report"])

# ─────────────────────────────────────────────────────────────────────────────
# Clear conversation button (sidebar)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history         = []
        st.session_state.conversation_history = []
        st.rerun()
