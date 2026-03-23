import os
import shutil
import streamlit as st

from ingestion import ingest_pdfs
from retriever import HybridRetriever
from graph import AgentWorkflow, Turn
from config import (
    COLLECTIONS_DIR,
    get_upload_dir,
    get_index_dir,
    GROQ_FREE_MODELS,
    GEMINI_FREE_MODELS,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
)

st.set_page_config(page_title="Multi-Agent RAG", layout="wide")
st.title(" Multi-Agent Hybrid RAG Chatbot")

def get_all_collections():
    if not os.path.exists(COLLECTIONS_DIR):
        return ["default"]
    cols = [d for d in os.listdir(COLLECTIONS_DIR) if os.path.isdir(os.path.join(COLLECTIONS_DIR, d))]
    if "default" not in cols:
        cols.append("default")
    return sorted(list(set(cols)))

os.makedirs(COLLECTIONS_DIR, exist_ok=True)

defaults = {
    "chat_history":          [],
    "conversation_history":  [],
    "retriever":             None,
    "model_provider":        DEFAULT_PROVIDER,
    "model_name":            DEFAULT_MODEL,
    "active_collection":     "default"
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

with st.sidebar:
    st.header(" Settings")
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
    st.subheader(" Collections")
    all_collections = get_all_collections()
    
    selected_col = st.selectbox(
        "Current Collection", 
        all_collections, 
        index=all_collections.index(st.session_state.active_collection) if st.session_state.active_collection in all_collections else 0
    )
    
    # If collection changed:
    if selected_col != st.session_state.active_collection:
        st.session_state.active_collection = selected_col
        st.session_state.retriever = None
        st.session_state.chat_history = []
        st.session_state.conversation_history = []
        st.rerun()
    
    c_new = st.text_input("New Collection Name")
    if st.button("Create Collection"):
        if c_new and c_new.strip() not in all_collections:
            get_upload_dir(c_new.strip())
            st.session_state.active_collection = c_new.strip()
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.rerun()
            
    if selected_col != "default":
        if st.button(f" Delete '{selected_col}'"):
            shutil.rmtree(os.path.join(COLLECTIONS_DIR, selected_col))
            st.session_state.active_collection = "default"
            st.session_state.retriever = None
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.rerun()

current_col = st.session_state.active_collection
upload_dir = get_upload_dir(current_col)
index_dir = get_index_dir(current_col)

# Collection Manager
st.markdown(f"###  Documents in [{current_col}]")
col_files = [f for f in os.listdir(upload_dir) if f.lower().endswith('.pdf')]

if col_files:
    for f in col_files:
        col1, col2 = st.columns([0.9, 0.1])
        col1.write(f" {f}")
        if col2.button("", key=f"del_{f}"):
            os.remove(os.path.join(upload_dir, f))
            # Delete removes item from the upload dir, prompt re-index
            st.session_state.retriever = None 
            st.rerun()
else:
    st.info("No documents in this collection.")

uploaded_files = st.file_uploader(f"Add PDFs to '{current_col}'", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    saved_any = False
    for f in uploaded_files:
        dest = os.path.join(upload_dir, f.name)
        if not os.path.exists(dest):
            with open(dest, "wb") as fh:
                fh.write(f.getbuffer())
            saved_any = True
    if saved_any:
        st.success("Files uploaded! Click 'Index Collection' to apply changes.")
        st.rerun()

colbase_has_pdf = len(os.listdir(upload_dir)) > 0
index_exists = os.path.exists(index_dir) and len(os.listdir(index_dir)) > 0

if colbase_has_pdf:
    if st.button(" Index / Re-index Collection", type="primary"):
        progress_bar = st.progress(0)
        status_text  = st.empty()
        try:
            ingest_pdfs(
                collection_name=current_col,
                progress_callback=lambda p, m: (progress_bar.progress(p), status_text.text(m))
            )
            st.session_state.retriever = None
            progress_bar.empty()
            status_text.empty()
            st.success(" Collection indexed! You can now ask questions.")
            st.rerun()
        except Exception as exc:
            progress_bar.empty(); status_text.empty()
            st.error(f" Indexing failed: {exc}")

st.divider()

# Chat
for msg in st.session_state.chat_history:
    st.chat_message("user").write(msg["user"])
    st.chat_message("assistant").write(msg["assistant"])
    if msg.get("citations"):
        st.caption(" Sources: " + "  ".join(f"{c}" for c in msg["citations"]))
    if msg.get("verification"):
        with st.expander(" Verification Report", expanded=False):
            st.markdown(msg["verification"])

question = st.chat_input(f"Ask about '{current_col}'...")

if question:
    if not index_exists:
        st.warning(" Please index the collection first before asking questions.")
        st.stop()

    if st.session_state.retriever is None:
        with st.spinner("Loading retriever..."):
            try:
                st.session_state.retriever = HybridRetriever(collection_name=current_col)
            except Exception as e:
                st.error(str(e))
                st.stop()

    st.chat_message("user").write(question)
    
    with st.chat_message("assistant"):
        status = st.empty()
        status.info("Retrieving documents...")

        retrieved_docs = st.session_state.retriever.invoke(question)
        if not retrieved_docs:
            status.warning("No relevant chunk found.")
            st.stop()

        status.info("Reasoning...")
        turn = Turn(
            question=question,
            retrieved_docs=retrieved_docs,
            conversation_history=st.session_state.conversation_history.copy(),
            enable_verification=enable_verification,
            provider=st.session_state.model_provider,
            model=st.session_state.model_name
        )

        wf = AgentWorkflow()
        final_state = wf.run(turn)

        status.empty()

        ans = final_state.get("final_answer", "")
        if not ans:
            st.warning("Could not generate an answer.")
            st.stop()

        st.write(ans)
        citations = [d.metadata.get('file_name', 'unknown') for d in retrieved_docs]
        unique_citations = list(dict.fromkeys(citations))
        st.caption(" Sources: " + "  ".join(f"{c}" for c in unique_citations))

        vr = final_state.get("verification_result")
        if vr:
            with st.expander(" Verification Report"):
                st.markdown(vr)

        st.session_state.chat_history.append({
            "user": question,
            "assistant": ans,
            "citations": unique_citations,
            "verification": vr
        })

        st.session_state.conversation_history.append({"role": "user", "content": question})
        st.session_state.conversation_history.append({"role": "assistant", "content": ans})
        if len(st.session_state.conversation_history) > 8:
            st.session_state.conversation_history = st.session_state.conversation_history[-8:]
