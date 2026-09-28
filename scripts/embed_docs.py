"""
embed_docs.py
Reads data/chunks.jsonl, embeds each chunk with Voyage AI, and stores vectors
plus metadata in a local Chroma collection. Safe to re-run: it rebuilds the
collection from scratch each time.

Run from the project root:
    python scripts/embed_docs.py
"""
import json
import os
import time
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()

CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION = "zenpharma_docs"
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-4-large")
BATCH_SIZE = 64
# Optional throttling for Voyage accounts without a payment method (3 RPM, 10K TPM):
#     VOYAGE_TPM=10000 VOYAGE_RPM=3 python scripts/embed_docs.py
VOYAGE_TPM = int(os.getenv("VOYAGE_TPM", "0"))
VOYAGE_RPM = int(os.getenv("VOYAGE_RPM", "3"))


def estimate_tokens(text):
    # Conservative (~3 chars/token) so batches don't overshoot the TPM budget.
    return len(text) // 3 + 1


def make_batches(chunks):
    """Group chunks by count and, when throttling, by an estimated token budget."""
    budget = VOYAGE_TPM * 0.8 if VOYAGE_TPM else float("inf")
    batch, tokens = [], 0
    for c in chunks:
        t = estimate_tokens(c["text"])
        if batch and (len(batch) >= BATCH_SIZE or tokens + t > budget):
            yield batch
            batch, tokens = [], 0
        batch.append(c)
        tokens += t
    if batch:
        yield batch


def main():
    if not CHUNKS_FILE.exists():
        raise SystemExit("data/chunks.jsonl not found. Run scripts/chunk_docs.py first.")

    chunks = [json.loads(line) for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(chunks)} chunks")

    # Works with keys created in MongoDB Atlas or on the Voyage platform;
    # the client picks the right endpoint from the key format.
    vo = voyageai.Client(max_retries=3, timeout=60)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    done = 0
    for batch in make_batches(chunks):
        tokens = sum(estimate_tokens(c["text"]) for c in batch)
        while True:
            try:
                result = vo.embed([c["text"] for c in batch], model=VOYAGE_MODEL, input_type="document")
                break
            except voyageai.error.RateLimitError:
                print("Rate limited; waiting 60s...")
                time.sleep(60)
        collection.add(
            ids=[c["id"] for c in batch],
            embeddings=result.embeddings,
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "section": c["section"]} for c in batch],
        )
        done += len(batch)
        print(f"Embedded {done}/{len(chunks)}")
        if VOYAGE_TPM and done < len(chunks):
            # Stay under both the per-minute token and request limits.
            time.sleep(max(60 * tokens / VOYAGE_TPM, 60 / VOYAGE_RPM) + 1)

    print(f"Done. Collection '{COLLECTION}' now holds {collection.count()} chunks at {CHROMA_PATH}")


if __name__ == "__main__":
    main()