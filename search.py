"""
search.py

Command-line interface for the Patent Semantic Search Engine.

Pipeline:
    1. Encode the user's query with the bi-encoder.
    2. Retrieve Top-K candidates from the FAISS index (fast, approximate).
    3. Rerank those candidates with a Cross-Encoder for higher precision.
    4. Print the reranked results.

Run:
    python search.py --k 10
"""

import argparse

import faiss
import numpy as np

from utils import (
    INDEX_FILE,
    load_bi_encoder,
    load_cross_encoder,
    load_meta,
)


def retrieve(query: str, model, index, meta: list, top_k: int):
    """Takes the query as input and returns the top_k similar vectors stored in FAISS index."""
    # encode the query
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)
    # store the results 
    scores, ids = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        record = dict(meta[idx])
        record["bi_score"] = float(score)
        results.append(record)
    return results


def rerank(query: str, candidates: list, cross_encoder):
    """We re-rank the vectors we retrieved through retrieve function using cross encoder."""
    pairs = [(query, c["abstract"]) for c in candidates]
    scores = cross_encoder.predict(pairs)
    for c, s in zip(candidates, scores):
        c["cross_score"] = float(s)
    return sorted(candidates, key=lambda c: c["cross_score"], reverse=True)


def print_results(results: list):
    for rank, r in enumerate(results, start=1):
        print(f"\n#{rank}  id={r['id']}  category={r['category']}  "
              f"cross_score={r.get('cross_score', 0):.3f}  bi_score={r.get('bi_score', 0):.3f}")
        print(f"    {r['abstract'][:280]}...")


def main():
    parser = argparse.ArgumentParser(description="Search similar patents.")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return.")
    parser.add_argument("--candidates", type=int, default=30,
                         help="Number of candidates fetched from FAISS before reranking.")
    args = parser.parse_args()

    print("Loading index and models (this happens once)...")
    index = faiss.read_index(INDEX_FILE)
    meta = load_meta()
    bi_encoder = load_bi_encoder()
    cross_encoder = load_cross_encoder()

    print("\nPatent Semantic Search Engine")
    print("Type a patent description and press Enter. Type 'quit' to exit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue

        candidates = retrieve(query, bi_encoder, index, meta, args.candidates)
        if not candidates:
            print("No results found. Did you run build_index.py?")
            continue

        results = rerank(query, candidates, cross_encoder)[: args.k]
        print_results(results)


if __name__ == "__main__":
    main()
