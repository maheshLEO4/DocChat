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

st.set_page_config(page_title="🐥 DocChat", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: "Space Grotesk", sans-serif;
    }

    .stApp {
        background: radial-gradient(1100px 700px at 15% 10%, #0a0a0a 0%, #0e0e0e 45%, #121212 100%);
        color: #e5e7eb;
    }

    .main, .block-container {
        background: transparent;
        color: #e5e7eb;
    }

    .hero-title {
        font-family: "IBM Plex Serif", serif;
        font-size: 5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        color: #cfcfcf;
        margin-top: 0;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.82rem;
        color: #bdbdbd;
        margin-bottom: 0.6rem;
    }

    .chip {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        background: #1f1f1f;
        color: #e0e0e0;
        font-size: 0.75rem;
        margin-left: 0.4rem;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #2d2d2d;
        background: #1c1c1c;
        color: #ffffff;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: #2a2a2a;
        border-color: #2a2a2a;
    }

    button[kind="primary"] {
        background: #b91c1c;
        border-color: #7f1d1d;
        color: #ffffff;
    }

    button[kind="primary"]:hover {
        background: #991b1b;
        border-color: #991b1b;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select, .stFileUploader label {
        color: #e5e7eb;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #0f0f0f;
        border: 1px solid #2d2d2d;
    }

    .stFileUploader {
        background: #0f0f0f;
        border: 1px dashed #2d2d2d;
        border-radius: 12px;
        padding: 0.75rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #1a1a1a;
        border: 1px solid #2d2d2d;
        color: #e5e7eb;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-title">🐥 DocChat</div>', unsafe_allow_html=True)
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

    st.divider()
    st.subheader("Upload")
    st.caption("Shared upload folder")

    upload_dir = get_upload_dir()
    index_dir = get_index_dir()

    col_files = [f for f in os.listdir(upload_dir) if f.lower().endswith(".pdf")]

    if col_files:
        if st.button("Clear all", key="clear_uploads"):
            for f in col_files:
                os.remove(os.path.join(upload_dir, f))
            st.session_state.retriever = None
            st.rerun()

        for f in col_files:
            row1, row2 = st.columns([0.86, 0.14])
            row1.write(f"{f}")
            if row2.button("🗑", key=f"del_{f}"):
                os.remove(os.path.join(upload_dir, f))
                st.session_state.retriever = None
                st.rerun()
    else:
        st.info("No documents in the upload folder.")

colbase_has_pdf = len(os.listdir(get_upload_dir())) > 0
index_exists = os.path.exists(get_index_dir()) and len(os.listdir(get_index_dir())) > 0

st.markdown("<div class='section-title'>Add Files</div>", unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Add PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    saved_any = False
    upload_dir = get_upload_dir()
    for f in uploaded_files:
        dest = os.path.join(upload_dir, f.name)
        if not os.path.exists(dest):
            with open(dest, "wb") as fh:
                fh.write(f.getbuffer())
            saved_any = True
    if saved_any:
        st.success("Files uploaded! Click 'Index PDFs' to apply changes.")
        st.rerun()

if colbase_has_pdf:
    if st.button("Index PDFs", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
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
            progress_bar.empty()
            status_text.empty()
            st.error(f"Indexing failed: {exc}")

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
        status.markdown("<span class='status-pill'>Thinking...</span>", unsafe_allow_html=True)

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
