import chromadb
import sys

client = chromadb.PersistentClient(path="memory/chroma")
collection = client.get_collection("cybrex")
results = collection.get(include=["metadatas"])

if not results["ids"]:
    print("No memories stored.")
    sys.exit()

for id, m in zip(results["ids"], results["metadatas"]):
    print(f"[{id[:8]}] {m.get('data')}")

print(f"\nTotal: {len(results['ids'])}")