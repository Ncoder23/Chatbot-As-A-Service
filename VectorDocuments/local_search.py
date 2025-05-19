#!/usr/bin/env python3
"""
local_search.py

– Loads FAISS index + id_map.pkl
– Connects to SQLite metadata.db
– Embeds your query & does a k-NN search
– Fetches the exact chunk by chunk_id
"""

import pickle
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIG
INDEX_FILE = "vector.index"
SQLITE_DB = "metadata.db"
ID_MAP_FILE = "id_map.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# 1) Load ID map
with open(ID_MAP_FILE, "rb") as f:
    id_map = pickle.load(f)

# 2) Load FAISS index
index = faiss.read_index(INDEX_FILE)

# 3) Connect to SQLite
conn = sqlite3.connect(SQLITE_DB)

# 4) Load embedder
embedder = SentenceTransformer(EMBED_MODEL)


def search(query: str, top_k: int = 5):
    # 5) Embed
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype="float32"), top_k)

    for dist, idx in zip(D[0], I[0]):
        try:
            chunk_id = id_map[idx]
        except IndexError:
            print(f"⚠️  No chunk_id for vector index {idx}")
            continue

        row = conn.execute(
            "SELECT text, source, position FROM chunks WHERE id = ?",
            (chunk_id,)
        ).fetchone()
        if not row:
            print(f"⚠️  No metadata for chunk_id {chunk_id}")
            continue

        text, source, pos = row
        print(
            f"--- Result (score {dist:.3f}) from {chunk_id} (source={source}, pos={pos}) ---")
        print(text[:300].replace("\n", " "), "\n")


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is FastAPI?"
    search(q, top_k=5)
