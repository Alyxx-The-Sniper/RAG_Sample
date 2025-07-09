#!/usr/bin/env python3
"""
retrieval_chat.py

Query Pinecone for Q/A pairs and generate completions
using DeepInfra’s OpenAI-compatible chat + embedding APIs.
"""

import os
import re
import httpx
from pinecone import Pinecone
from dotenv import load_dotenv



# Load .env (optional if you already have env vars)
load_dotenv(override=True)

# --- Configuration ---
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
NAMESPACE           = os.getenv("PINECONE_NAMESPACE", None)
INDEX_NAME          = os.getenv("PINECONE_INDEX_NAME", "rag-about-me")
SIM_THRESHOLD       = float(os.getenv("SIM_THRESHOLD", 0.3))
RETRIEVE_K          = int(os.getenv("RETRIEVE_K"))

DEEPINFRA_API_KEY   = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_LLM_MODEL = os.getenv("DEEPINFRA_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
DEEPINFRA_CHAT_URL  = os.getenv("DEEPINFRA_CHAT_URL", "https://api.deepinfra.com/v1/openai/chat/completions")

DEEPINFRA_EMBEDDING_MODEL = os.getenv("DEEPINFRA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEEPINFRA_EMBEDDING_URL = os.getenv("DEEPINFRA_EMBEDDING_URL", "https://api.deepinfra.com/v1/openai/embeddings")


# --- Pinecone client ---
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


### 1.  Embed text/query/prompt
async def embed_query(text):
    
    # 1.) payload
    payload = {
        "input": text,
        "model": DEEPINFRA_EMBEDDING_MODEL ,
        "encoding_format": "float"
    }
    # 2.) headers
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    # 3.) 
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(DEEPINFRA_EMBEDDING_URL,
                                json=payload,
                                headers=headers)
        
        return resp.json()["data"][0]["embedding"]




### 2.  RETRIEVE top k in pinecone
async def retrieve(query, k=RETRIEVE_K):
    vec = await embed_query(query)
    ns  = NAMESPACE or None

    # query Pinecone
    resp = index.query(
        vector=vec,
        top_k=k,
        include_metadata=True,
        namespace=ns
    )

    # get matches
    matches = resp.matches

    good = []
    for m in matches:
        if m.score >= SIM_THRESHOLD:
            print(f"PASS: {m.score:.4f} >= {SIM_THRESHOLD} ID: {m.id}")
            good.append(m)
        else:
            print(f"SKIP: {m.score:.4f} >= {SIM_THRESHOLD} ID: {m.id}")

    paired = [
        (m.metadata["question"], m.metadata["answer"])
        for m in good
    ]
    return paired


### extra function Clean the output of LLM 
def clean_llm_output(text):
    p1      = re.compile(r'^(\W+\s*)?(Final Answer:|Final Response:)\s*',
                         flags=re.IGNORECASE)
    p2      = re.compile(r'^[^\w\s]+ ')
    lines   = text.splitlines()
    cleaned = [p2.sub("", p1.sub("", line)).strip() for line in lines]
    unique  = dict.fromkeys(filter(None, cleaned))
    return " ".join(unique).strip()


### 3. The LLM MOdel
async def call_deepinfra_llm(prompt, context_str):

    messages = [
    {
        "role": "system",
        "content": (
            "You are Ayx, an AI-agent and ML/AI enthusiast. "
            "Answer **only** using the CONTEXT provided. "
            "Be friendly, approachable, and robotic sound like; you may insert emojis when it adds to the tone. "
            "Keep your answers engaging and grounded in best practices or real-world knowledge. "
            "Always share my LinkedIn profile at the end: "
            "https://www.linkedin.com/in/alexis-mandario-b546881a8"
            "Strictly dont change my url links."
        )
    },
    {
        "role": "system",
        "content": f"CONTEXT:\n{context_str}" # see context in main.py
    },
    {
        "role": "user",
        "content": prompt
    }
    ]


    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPINFRA_LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
        "stop": ["</s>", "<|im_end|>"]
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp   = await client.post(DEEPINFRA_CHAT_URL,
                                   json=payload,
                                   headers=headers)
        data   = resp.json()
        result = data["choices"][0]["message"]["content"].strip()
        return clean_llm_output(result) 


