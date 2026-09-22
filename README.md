

# 🚀 DocChat: Multi-Agent Hybrid RAG

Upload your PDFs, index them on the fly, and interact with a sophisticated multi-agent Retrieval-Augmented Generation (RAG) workflow. Powered by Streamlit, LangGraph/custom agents, and support for high-speed LLM providers like Groq and Google Gemini.

---

## 🌟 Features

* **Multi-Agent Architecture:** Specialized agents coordinate to handle query understanding, document retrieval, and response synthesis.
* **Hybrid RAG Retrieval:** Combines semantic vector search with keyword-based retrieval for high accuracy.
* **Dynamic Runtime Indexing:** Upload PDFs directly via the UI; documents are parsed and indexed at runtime.
* **Flexible LLM Providers:** Easily switch between **Groq** and **Gemini** via the app sidebar.
* **Cloud Ready:** Fully configured for deployment on Hugging Face Spaces (Docker) and Render with persistent storage support.

---

## 📂 Project Structure

```text
DocChat/
├── agents/          # Multi-agent logic and orchestration
├── graph/           # RAG workflow graph definitions
├── ingestion/       # PDF parsing and chunking pipelines
├── retriever/       # Hybrid search and retrieval logic
├── utils/           # Helper functions and utilities
├── .streamlit/      # Streamlit configuration
├── app.py           # Main Streamlit application entry point
├── config.py        # Application configuration settings
├── Dockerfile       # Docker container configuration
├── requirements.txt # Python dependencies
└── runtime.txt      # Python runtime specification
```

---

## 💻 Local Development & Setup

### 1. Clone the repository

```bash
git clone https://github.com/maheshLEO4/DocChat.git
cd DocChat
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 🌐 Deployment

### Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Select **Docker** as the SDK.
3. Upload the repository files.
4. Go to **Settings → Secrets**.
5. Add your API key as a secret:

   * `GROQ_API_KEY`
   * or `GEMINI_API_KEY`
6. Launch the Space.

### Render

1. Create a new **Web Service** connected to this repository.
2. Select **Docker** as the environment.
3. Add the required environment variables.
4. Attach a persistent disk if uploaded documents and indexes need to survive restarts.
5. Deploy the service.

---

## 📝 Notes & Tips

* **Runtime Indexing:** Documents are parsed, chunked, embedded, and indexed after upload.
* **Large PDFs:** Processing large documents may take some time.
* **Persistence:** Persistent storage is recommended when deploying on platforms where containers can restart.
* **Security:** Never commit API keys or `.env` files to GitHub.

---

## 🔗 Repository

GitHub: https://github.com/maheshLEO4/DocChat
**
