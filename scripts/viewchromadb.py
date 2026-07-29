#!/usr/bin/env python3
"""View all memories stored in the ChromaDB long-term memory collection."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BASE_DIR

import chromadb

CHROMA_PATH = os.path.join(BASE_DIR, "memory", "chroma")
COLLECTION_NAME = "cybrex"

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception:
    print("No memory collection found.")
    sys.exit(0)

results = collection.get(include=["documents"])

if not results["ids"]:
    print("No memories stored.")
    sys.exit(0)

for i, (id_, doc) in enumerate(zip(results["ids"], results["documents"]), 1):
    print(f"[{i}] {id_[:8]}")
    print(doc)
    print()

print(f"Total: {len(results['ids'])} memories")
