# AI Chatbot of MYself — RAG + Streamlit + FastAPI + Ollama + Docker

This project is an AI Chat Bot called **Alyx**, representing Alexis Mandario.  
It uses a RAG (Retrieval Augmented Generation) pipeline powered by Docker container + Ollama + FastAPI + Pinecode backend and a Streamlit frontend.

---

✅ backend (FastAPI + retrieval)
✅ frontend (Streamlit)
✅ ollama (LLM server)
✅ Docker (Container)
✅ Pinecode (Vector Database)
…all in one repo, running via docker-compose.

###  Usig docker-compose

```bash
docker-compose up --build
• Backend: http://localhost:8000
• Frontend: http://localhost:8501
• Ollama: http://localhost:11434

---

NOTES:
1. LLM models are save in docker volume (you can add models similar architecture to gemma)
2. Embeding models is saved in docker volume
3. The dataset (jsonl share gpt style) and chroma (vector database) are bind mounted to local host (for easy editing)
4. You can create your own datase (ex. root/data/.jsonl) must be in sharegpt style then run scripts/ingest/py that will create a chroma vector database.
5. Must have docker and ollama 



