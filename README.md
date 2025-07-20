# 🎥 Video Transcript AI — RAG Based LLM

This project demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline that extracts YouTube video transcripts and lets you **ask questions** about the video using **LLMs and embeddings**.

---

## 📂 Repository Contents

- **`RAG Based LLM Model.ipynb`**
  - An interactive Jupyter Notebook that builds the RAG pipeline step-by-step.
  - Loads a YouTube transcript.
  - Splits the transcript into chunks.
  - Embeds text chunks.
  - Stores embeddings in a vector store (**FAISS**).
  - Uses a Language Model to generate answers based on retrieved context.

- **`UI.py`**
  - A **Streamlit web app** that provides an intuitive user interface.
  - Users can:
    - Enter a YouTube URL.
    - Select transcript language.
    - See the video thumbnail.
    - Automatically extract and embed the transcript.
    - Ask questions about the video.
  - Uses local embedding models (`SentenceTransformer`) and a local LLM (`Ollama`).

---

## 🚀 **How It Works**

1️⃣ **Extract Transcript**  
Extracts video transcript using `youtube-transcript-api`.

2️⃣ **Split & Embed**  
Splits transcript into overlapping text chunks → Generates embeddings using `SentenceTransformer`.

3️⃣ **Vector Store**  
Stores embeddings in a **FAISS** index for fast similarity search.

4️⃣ **Retrieve Relevant Chunks**  
Searches vector store for the most relevant chunks for a user’s question.

5️⃣ **Generate Answer**  
Uses a local LLM (`Ollama`) or remote (`HuggingFace`) to generate context-grounded answers.

---

## 🛠️ **Tech Stack**

- **Python**, **Jupyter Notebook**, **Streamlit**
- **LangChain** components for chaining RAG steps.
- **SentenceTransformers** for local embeddings.
- **FAISS** for vector similarity search.
- **Ollama** or **HuggingFace** for LLM completion.
- **YouTube Transcript API** for transcript fetching.

---


   
