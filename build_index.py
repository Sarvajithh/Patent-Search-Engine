"""
build_index.py

1. Downloads a small subset of the BigPatent dataset from Hugging Face
   (streaming mode so we never pull the full ~50GB dataset - only the
   first N examples of one CPC category, kept well under 500MB).
2. Cleans the text and saves it locally to data/patents.jsonl.
3. Encodes the patent abstracts with a Sentence-Transformers bi-encoder.
4. Builds a FAISS index over the embeddings and saves it to models/.

Run:
    python build_index.py --n 800 --category g
"""

import argparse
import itertools
import os

import faiss
import numpy as np
from tqdm import tqdm

from utils import (
    INDEX_FILE,
    MODELS_DIR,
    PATENTS_FILE,
    clean_text,
    load_bi_encoder,
    save_meta,
    save_patents,
)

# BigPatent CPC top-level categories, e.g. 'g' = Physics, 'a' = Human Necessities
DEFAULT_CATEGORY = "g"
DEFAULT_N = 800


def download_subset(category: str, n: int) -> list:
    """Stream a small slice of BigPatent so we never touch the full dataset."""
    from datasets import load_dataset

    print(f"Streaming BigPatent category '{category}' (first {n} examples)...")
    stream = load_dataset(
        "big_patent",
        category,
        split="train",
        streaming=True,
    )

    records = []
    for i, ex in enumerate(itertools.islice(stream, n)):
        abstract = clean_text(ex.get("abstract", ""))
        description = clean_text(ex.get("description", ""))[:1000]  # keep it small
        if not abstract:
            continue
        records.append({
            "id": f"{category}_{i}",
            "category": category,
            "abstract": abstract,
            "description": description,
        })
    return records


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Cosine similarity via inner product on L2-normalized vectors."""
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def main():
    parser = argparse.ArgumentParser(description="Build the patent FAISS index.")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                         help="Number of patents to download (keeps dataset small).")
    parser.add_argument("--category", type=str, default=DEFAULT_CATEGORY,
                         help="BigPatent CPC category letter (a-y).")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    records = download_subset(args.category, args.n)
    print(f"Downloaded {len(records)} patents.")
    save_patents(records, PATENTS_FILE)

    print("Loading bi-encoder model (sentence-transformers/all-MiniLM-L6-v2)...")
    model = load_bi_encoder()

    texts = [r["abstract"] for r in records]
    print("Encoding patent abstracts...")
    embeddings = model.encode(
        texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    index = build_faiss_index(embeddings)
    faiss.write_index(index, INDEX_FILE)
    save_meta(records)

    print(f"Saved FAISS index -> {INDEX_FILE}")
    print(f"Saved metadata     -> models/patents_meta.json")
    print(f"Saved raw patents  -> {PATENTS_FILE}")


if __name__ == "__main__":
    main()
