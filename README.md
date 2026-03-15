---
title: Multi-Agent Hybrid RAG
emoji: ""
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
3. Add a secret named `GROQ_API_KEY` in **Settings → Secrets**.
4. The app will start automatically.

## Notes

- Indexing is done at runtime after upload.
- Large PDFs may require a few minutes to index.
