"""
evaluate.py

Evaluates retrieval quality with Recall@10, MRR and nDCG@10.

Ground truth strategy (kept simple, no external labels needed):
    For each sampled patent, we build a query from its *description*
    snippet (a different field than the one we indexed, the abstract).
    The correct answer is that same patent. Other patents that share
    its CPC category are treated as partially relevant, which lets us
    compute a graded nDCG as well as an exact-match Recall/MRR.

Run:
    python evaluate.py --samples 100
"""

import argparse
import random

import faiss
import numpy as np

from search import retrieve
from utils import INDEX_FILE, load_bi_encoder, load_meta


def reciprocal_rank(ranked_ids: list, correct_id: str) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == correct_id:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: list, correct_id: str, k: int) -> float:
    return 1.0 if correct_id in ranked_ids[:k] else 0.0


def relevance(rid: str, correct_id: str, correct_category: str, meta_by_id: dict) -> float:
    if rid == correct_id:
        return 2.0  # exact match: highly relevant
    if meta_by_id[rid]["category"] == correct_category:
        return 1.0  # same technology category: somewhat relevant
    return 0.0


def ndcg_at_k(ranked_ids: list, correct_id: str, correct_category: str,
              meta_by_id: dict, k: int) -> float:
    gains = [relevance(rid, correct_id, correct_category, meta_by_id) for rid in ranked_ids[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    ideal_gains = sorted(gains, reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate the search pipeline.")
    parser.add_argument("--samples", type=int, default=100, help="Number of queries to evaluate.")
    parser.add_argument("--k", type=int, default=10, help="Cutoff K for the metrics.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    index = faiss.read_index(INDEX_FILE)
    meta = load_meta()
    meta_by_id = {r["id"]: r for r in meta}
    bi_encoder = load_bi_encoder()

    usable = [r for r in meta if r.get("description")]
    sample = random.sample(usable, min(args.samples, len(usable)))

    recalls, mrrs, ndcgs = [], [], []

    for record in sample:
        query = record["description"][:300]  # simulate a user's free-text query
        candidates = retrieve(query, bi_encoder, index, meta, top_k=max(args.k, 20))
        ranked_ids = [c["id"] for c in candidates]

        recalls.append(recall_at_k(ranked_ids, record["id"], args.k))
        mrrs.append(reciprocal_rank(ranked_ids, record["id"]))
        ndcgs.append(ndcg_at_k(ranked_ids, record["id"], record["category"], meta_by_id, args.k))

    print(f"Evaluated on {len(sample)} queries (k={args.k})")
    print(f"Recall@{args.k}: {np.mean(recalls):.4f}")
    print(f"MRR:       {np.mean(mrrs):.4f}")
    print(f"nDCG@{args.k}:   {np.mean(ndcgs):.4f}")


if __name__ == "__main__":
    main()
