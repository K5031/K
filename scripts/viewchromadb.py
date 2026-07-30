#!/usr/bin/env python3
"""View all memories stored in the ChromaDB long-term memory collection."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BASE_DIR

import chromadb

CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma")
COLLECTION_NAME = "k_memory"

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    print("No memory collection found.")
    sys.exit(0)

results = collection.get(include=["metadatas"])

if not results["ids"]:
    print("No memories stored.")
    sys.exit(0)

for i, (id_, meta) in enumerate(zip(results["ids"], results["metadatas"]), 1):
    meta = meta or {}
    text = meta.get("data") or meta.get("memory") or meta.get("text") or "(no text field found)"
    print(f"[{i}] {id_[:8]}")
    print(f"  text: {text}")
    other = {k: v for k, v in meta.items() if k not in ("data", "memory", "text")}
    if other:
        print(f"  meta: {other}")
    print()

print(f"Total: {len(results['ids'])} memories")