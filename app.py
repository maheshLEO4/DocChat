import os
import streamlit as st

from ingestion import ingest_pdfs
from retriever import HybridRetriever
from graph import AgentWorkflow
from config import (
    get_upload_dir,
    get_index_dir,
    GROQ_FREE_MODELS,
    GEMINI_FREE_MODELS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
)

st.set_page_config(page_title="Docchat", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: "Space Grotesk", sans-serif;
    }

    .stApp {
        background: radial-gradient(1100px 700px at 15% 10%, #0f172a 0%, #111827 45%, #0b1220 100%);
        color: #e5e7eb;
    }

    .main, .block-container {
        background: transparent;
        color: #e5e7eb;
    }

    .hero-title {
        font-family: "IBM Plex Serif", serif;
        font-size: 2.2rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #cbd5f5;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.82rem;
        color: #a5b4fc;
        margin-bottom: 0.6rem;
    }

    .chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: #1f2a44;
        color: #c7d2fe;
        font-size: 0.75rem;
        margin-left: 0.4rem;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #334155;
        background: #6366f1;
        color: #ffffff;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #4f46e5;
        border-color: #4f46e5;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stFileUploader label {
        color: #e5e7eb;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #0f172a;
        border: 1px solid #334155;
    }

    .stFileUploader {
        background: #0f172a;
        border: 1px dashed #334155;
        border-radius: 12px;
        padding: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">Multi-Agent Hybrid RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Upload PDFs, index them, and chat with grounded answers.</p>',
    unsafe_allow_html=True,
)

defaults = {
    "chat_history":          [],
    "conversation_history":  [],
    "retriever":             None,
    "model_provider":        DEFAULT_PROVIDER,
    "model_name":            DEFAULT_MODEL,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.header("Settings")
    enable_verification = st.checkbox("Enable Verification", value=False)
    
    st.divider()
    st.subheader("Model")
    provider_labels = ["Groq", "Gemini"]
    provider_index = 0 if st.session_state.model_provider == "groq" else 1
    provider_label = st.selectbox("Provider", provider_labels, index=provider_index)
    model_provider = provider_label.lower()

    model_options = GROQ_FREE_MODELS if model_provider == "groq" else GEMINI_FREE_MODELS
    if st.session_state.model_name not in model_options:
        st.session_state.model_name = model_options[0]

    model_name = st.selectbox("Model", model_options, index=model_options.index(st.session_state.model_name))
    st.session_state.model_provider = model_provider
    st.session_state.model_name = model_name

upload_dir = get_upload_dir()
index_dir = get_index_dir()

# Upload Manager
st.markdown("<div class='section-title'>Upload files</div>", unsafe_allow_html=True)
st.markdown("### Upload folder <span class='chip'>shared</span>", unsafe_allow_html=True)
col_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]

if col_files:
    col_a, col_b = st.columns([0.8, 0.2])
    col_a.caption("Files currently in the upload folder")
    if col_b.button("Clear all", key="clear_uploads"):
        for f in col_files:
            os.remove(os.path.join(upload_dir, f))
        st.session_state.retriever = None
        st.rerun()

    for f in col_files:
        row1, row2 = st.columns([0.85, 0.15])
        row1.write(f"{f}")
        if row2.button("Remove", key=f"del_{f}"):
            os.remove(os.path.join(upload_dir, f))
            # Delete removes item from the upload dir, prompt re-index
            st.session_state.retriever = None
            st.rerun()
else:
    st.info("No documents in the upload folder.")

st.markdown("<div class='section-title'>Add documents</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Add PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    saved_any = False
    for f in uploaded_files:
        dest = os.path.join(upload_dir, f.name)
        if not os.path.exists(dest):
            with open(dest, "wb") as fh:
                fh.write(f.getbuffer())
            saved_any = True
    if saved_any:
        st.success("Files uploaded! Click 'Index PDFs' to apply changes.")
        st.rerun()

colbase_has_pdf = len(os.listdir(upload_dir)) > 0
index_exists = os.path.exists(index_dir) and len(os.listdir(index_dir)) > 0

if colbase_has_pdf:
    if st.button("Index PDFs", type="primary"):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        try:
            ingest_pdfs(
                progress_callback=lambda p, m: (progress_bar.progress(p), status_text.text(m))
            )
            st.session_state.retriever = None
            progress_bar.empty()
            status_text.empty()
            st.success("Index ready! You can now ask questions.")
            st.rerun()
        except Exception as exc:
            progress_bar.empty(); status_text.empty()
            st.error(f"Indexing failed: {exc}")

st.divider()

# Chat
st.markdown("<div class='section-title'>Conversation</div>", unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["assistant"])
    if msg.get("citations"):
        st.caption("Sources: " + "  ".join(f"{c}" for c in msg["citations"]))
    if msg.get("verification"):
        with st.expander("Verification Report", expanded=False):
            st.markdown(msg["verification"])

question = st.chat_input("Ask about your PDFs...")

if question:
    if not index_exists:
        st.warning("Please index the PDFs first before asking questions.")
        st.stop()

    if st.session_state.retriever is None:
        with st.spinner("Loading retriever..."):
            try:
                st.session_state.retriever = HybridRetriever()
            except Exception as e:
                st.error(str(e))
                st.stop()

    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        status = st.empty()
        status.info("Reasoning...")

        wf = AgentWorkflow(enable_verification=enable_verification)
        final_state = wf.run(
            question=question,
            retriever=st.session_state.retriever,
            conversation_history=st.session_state.conversation_history.copy(),
            model_provider=st.session_state.model_provider,
            model_name=st.session_state.model_name,
        )

        status.empty()

        ans = final_state.get("draft_answer", "")
        if not ans:
            st.warning("Could not generate an answer.")
            st.stop()

        st.write(ans)
        citations = final_state.get("citations", [])
        if citations:
            unique_citations = list(dict.fromkeys(citations))
            st.caption("Sources: " + "  ".join(f"{c}" for c in unique_citations))

        vr = final_state.get("verification_report")
        if vr:
            with st.expander("Verification Report"):
                st.markdown(vr)

        st.session_state.chat_history.append({
            "user": question,
            "assistant": ans,
            "citations": citations,
            "verification": vr,
        })

        st.session_state.conversation_history = final_state.get(
            "updated_history",
            st.session_state.conversation_history,
        )
