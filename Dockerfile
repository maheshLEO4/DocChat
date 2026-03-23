FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# ── Pin HuggingFace model cache to /data ──────────────────────────────────────
# In HF Spaces, /data is the persistent disk volume.
# Without this, the 22 MB MiniLM model is re-downloaded on every cold start,
# adding ~30-60 s to the first indexing run of each session.
ENV HF_HOME=/data/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/data/hf_cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/data/hf_cache/transformers

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT}"]