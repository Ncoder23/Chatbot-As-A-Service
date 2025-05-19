#!/usr/bin/env python3
"""
local_ingest.py

– Scans uploads/ for PDF/DOCX/HTML
– Extracts & chunks text
– Generates embeddings
– Upserts into FAISS (vector.index)
– Records metadata in SQLite (metadata.db)
"""

import pickle
import os
import uuid
import sqlite3
from tika import parser
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# CONFIGURATION
UPLOAD_DIR = "uploads"
INDEX_FILE = "vector.index"
SQLITE_DB = "metadata.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1500      # words
CHUNK_OVERLAP = 110    # words

# load or init id_map
if os.path.exists("id_map.pkl"):
    with open("id_map.pkl", "rb") as f:
        id_map = pickle.load(f)
else:
    id_map = []

# Initialize embedder
embedder = SentenceTransformer(EMBEDDING_MODEL)
dim = embedder.get_sentence_embedding_dimension()

# --- 1. Initialize / load FAISS index ---
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    # Flat L2 index; for production consider IVF or HNSW
    index = faiss.IndexFlatL2(dim)

# --- 2. Initialize / connect SQLite ---
conn = sqlite3.connect(SQLITE_DB)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    source TEXT,
    text TEXT,
    position INTEGER,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)""")
conn.commit()


def extract_text(filepath: str) -> str:
    """Extract raw text from any file via Apache Tika."""
    parsed = parser.from_file(filepath)
    return parsed.get("content", "") or ""


def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split `text` into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + size])
        chunks.append(chunk)
    return chunks


def upsert_file(filepath: str):
    """
    Process one file: extract text, chunk, embed, upsert into FAISS,
    record metadata in SQLite, and update the id_map.
    """
    print(f"[+] Processing {filepath}")
    # 1. Extract
    text = extract_text(filepath)
    if not text.strip():
        print("    ⚠️ No text found; skipping.")
        return

    # 2. Chunk
    chunks = chunk_text(text)

    # 3. Embed all chunks at once
    embeddings = embedder.encode(chunks, show_progress_bar=True)

    # 4. Upsert each chunk
    for pos, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        chunk_id = str(uuid.uuid4())

        # 4a. Add vector to FAISS
        index.add(np.array([emb], dtype="float32"))

        # 4b. Track the mapping from FAISS index → chunk_id
        id_map.append(chunk_id)

        # 4c. Insert metadata into SQLite
        c.execute(
            "INSERT INTO chunks (id, source, text, position) VALUES (?, ?, ?, ?)",
            (chunk_id, os.path.basename(filepath), chunk, pos)
        )

    # 5. Persist FAISS index and SQLite DB
    faiss.write_index(index, INDEX_FILE)
    conn.commit()

    # 6. Persist the updated id_map
    with open("id_map.pkl", "wb") as f:
        pickle.dump(id_map, f)

    print(f"    ✅ Indexed {len(chunks)} chunks.")


def main():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    seen = set()
    # Keep track of which files we've already ingested
    for row in c.execute("SELECT source FROM chunks"):
        seen.add(row[0])

    for fname in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, fname)
        if fname in seen:
            continue
        upsert_file(path)

    print("[*] Done ingesting new files.")


if __name__ == "__main__":
    main()
