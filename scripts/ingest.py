#!/usr/bin/env python3

import os
import json
import uuid
from pathlib import Path

import requests
import pinecone
from pinecone import ServerlessSpec
from tqdm import tqdm

# --- Configuration ---
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
ENV           = os.getenv("PINECONE_ENV", "us-east-1-aws")
INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "rag-about-me")
METRIC        = os.getenv("PINECONE_METRIC", "cosine")
EMBED_DIM     = 384
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", 100))

JSONL_PATH    = Path(__file__).resolve().parent.parent / "data" / "me.jsonl"

DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY") 
DEEPINFRA_EMBEDDING_MODEL = os.getenv("DEEPINFRA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEEPINFRA_EMBEDDING_URL = os.getenv("DEEPINFRA_EMBEDDING_URL", "https://api.deepinfra.com/v1/openai/embeddings")

# --- Helper to parse ENV into region & cloud ---
def parse_env(env):
    parts = env.split("-")
    return "-".join(parts[:-1]), parts[-1]

region, cloud = parse_env(ENV)

# --- Initialize Pinecone client ---
client = pinecone.Pinecone(PINECONE_API_KEY=PINECONE_API_KEY, environment=ENV)

# --- Create index if it does not exist ---
if INDEX_NAME not in client.list_indexes().names():
    client.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=cloud, region=region)
    )

index = client.Index(INDEX_NAME)

# --- Load Q/A data from JSONL file --- must be a sharegpt style format
with open(JSONL_PATH, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# --- Extract questions and answers ---
questions = []
metadatas = []
for obj in records:
    msgs = obj["messages"]
    q = msgs[0]["content"]
    a = msgs[1]["content"] if len(msgs) > 1 else ""  # ✅ 2 → dahil 2 > 1 → ito ay TRUE → kukunin ang msgs[1]["content"] as a. if not else ""
                                                     # we can raise error handling here if wrong jsonl formal (be sure to asharegpt style)
    
    questions.append(q)
    metadatas.append({"question": q, "answer": a})


# --- Function to get embeddings from Deepinfra ---
def get_embeddings(texts):

    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": DEEPINFRA_EMBEDDING_MODEL,
        "input": texts
    }

    response = requests.post(DEEPINFRA_EMBEDDING_URL, 
                             headers=headers, 
                             json=payload
                             )
    
    return [d["embedding"] for d in response.json()["data"]]

# --- Get embeddings ---
embeddings = get_embeddings(questions) # embed the questions

# --- Build vectors with IDs, embeddings, and metadata ---
vectors = [
    (str(uuid.uuid4()), emb, meta)
    for emb, meta in zip(embeddings, metadatas)
]

# --- Upsert vectors into Pinecone in batches ---
for i in tqdm(range(0, len(vectors), BATCH_SIZE), desc="Upserting"):
    batch = vectors[i : i + BATCH_SIZE]
    index.upsert(vectors=batch)
