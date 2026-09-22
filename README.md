---

title: Multi-Agent Hybrid RAG
emoji: "📝"
colorFrom: "yellow"
colorTo: "red"
sdk: docker
pinned: false
-------------

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

Create a `.env` file in the root directory and add your API keys:

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

### Option 1: Hugging Face Spaces (Docker)

1. Create a new **Space** on Hugging Face and choose **Docker** as the SDK.
2. Upload the contents of this repository to your Space.
3. Go to **Settings → Secrets** and add a secret named either:

   * `GROQ_API_KEY`
   * `GEMINI_API_KEY`
4. Launch your Space and select your preferred provider and model from the app sidebar.

### Option 2: Render

1. Create a new **Web Service** connected to this repository.
2. Select **Docker** as the environment using the included `Dockerfile`.
3. Add the following environment variables under **Environment**:

```text
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
APP_DATA_DIR=/var/data
```

4. Attach a **Persistent Disk** and mount it at `/var/data` to ensure uploaded PDFs and vector indexes persist across container restarts.
5. The container will automatically bind to Render's dynamic `PORT`.

---

## 📝 Notes & Tips

* **Runtime Indexing:** Indexing occurs dynamically after you upload your documents. Large PDFs may take a few minutes to fully process.
* **Persistence:** Always use a persistent disk in production environments such as Render so that uploaded documents and indexes survive restarts.
* **API Keys:** Never commit your `.env` file or API keys to GitHub. Add `.env` to your `.gitignore`.

---

## 🔗 Repository

**GitHub:** https://github.com/maheshLEO4/DocChat.git
