---
title: Multi-Agent Hybrid RAG
emoji: "📝"
colorFrom: "yellow"
colorTo: "red"
sdk: docker
pinned: false
---

# Multi-Agent Hybrid RAG

Upload PDFs, index them, and chat with a multi-agent RAG workflow.

## Setup (Hugging Face Spaces - Docker)

1. Create a new Space and choose **Docker**.
2. Upload this repository contents.
3. Add a secret named `GROQ_API_KEY` or `GEMINI_API_KEY` in **Settings → Secrets**.
4. Choose the provider and model in the app sidebar.

## Setup (Render)

1. Create a new **Web Service** from this repository.
2. Deploy it with the included `Dockerfile`.
3. Add `GROQ_API_KEY` or `GEMINI_API_KEY` as an environment variable.
4. Add `APP_DATA_DIR=/var/data` as an environment variable.
5. Attach a persistent disk and mount it at `/var/data`.

The container binds to Render's `PORT` automatically.

## Notes

- Indexing is done at runtime after upload.
- Use a persistent disk in production so uploaded PDFs and indexes survive restarts.
- Large PDFs may require a few minutes to index.
