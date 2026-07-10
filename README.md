# Patent Semantic Search Engine

A small, self-contained semantic search engine for patents: embed patent
abstracts with a free Sentence-Transformers model, index them with FAISS,
retrieve the Top-K most similar patents for a user query, and rerank the
candidates with a Cross-Encoder for better precision.

## Acknowledgement

This project was built for learning purposes and takes inspiration from the
NLP course repository [nlpia2](https://gitlab.com/tangibleai/nlpia2/) by
Tangible AI, which demonstrates practical NLP pipelines (tokenization,
embeddings, vector search, etc.) in Python. The code, structure and design
here are original and were written from scratch for this patent-search use
case; no code was copied from nlpia2.

## How it works

```
query text
   │
   ▼
Sentence-Transformers bi-encoder (all-MiniLM-L6-v2)
   │  encodes query into a vector
   ▼
FAISS IndexFlatIP  ──►  Top-N candidate patents (fast, approximate)
   │
   ▼
Cross-Encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
   │  scores each (query, candidate) pair jointly for higher precision
   ▼
Top-K final results
```

## Project structure

```
project/
├── data/               # local patent subset (created by build_index.py)
├── models/             # FAISS index + metadata (created by build_index.py)
├── build_index.py      # download subset, embed, build FAISS index
├── search.py            # CLI: query -> retrieve -> rerank -> print
├── evaluate.py           # Recall@10, MRR, nDCG@10
├── utils.py              # shared helpers (I/O, model loading, cleaning)
├── requirements.txt
└── README.md
```

## Dataset

We use a small streamed subset of the [BigPatent](https://huggingface.co/datasets/big_patent)
dataset (one CPC category, a few hundred patents), which keeps the local
data well under 500MB — no need to download the full ~50GB corpus.

## Models (all free, via Hugging Face, no paid APIs)

- Bi-encoder (for indexing/retrieval): `sentence-transformers/all-MiniLM-L6-v2`
- Cross-encoder (for reranking): `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Build the index (downloads a small BigPatent subset and embeds it):

```bash
python build_index.py --n 800 --category g
```

2. Search interactively:

```bash
python search.py --k 10
```

```
Query> a device for filtering water using a porous ceramic membrane
```

3. Evaluate retrieval quality:

```bash
python evaluate.py --samples 100 --k 10
```

Reports **Recall@10**, **MRR**, and **nDCG@10**.

## Notes

- Everything runs on CPU with small, free models — no OpenAI, no paid APIs.
- No LangChain, Haystack, Docker, or MLflow — just plain Python scripts.
- Embeddings use cosine similarity (L2-normalized vectors + FAISS inner-product index).
