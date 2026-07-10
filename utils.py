"""
utils.py
Shared helper functions for the Patent Semantic Search Engine.

Keeps I/O, text cleaning and model-loading logic in one place so that
build_index.py, search.py and evaluate.py stay short and focused.
"""

import json
import os
import re

DATA_DIR = "data"
MODELS_DIR = "models"

PATENTS_FILE = os.path.join(DATA_DIR, "patents.jsonl")
INDEX_FILE = os.path.join(MODELS_DIR, "patents.faiss")
META_FILE = os.path.join(MODELS_DIR, "patents_meta.json")

# Small, free, CPU-friendly models (Hugging Face)
BI_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def clean_text(text: str) -> str:
    """Light cleanup: collapse whitespace, strip odd characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def load_patents(path: str = PATENTS_FILE) -> list:
    """Load the local patent subset saved as JSON Lines."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run build_index.py first "
            f"to download and prepare the patent subset."
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_patents(records: list, path: str = PATENTS_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_bi_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BI_ENCODER_NAME)


def load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(CROSS_ENCODER_NAME)


def load_meta(path: str = META_FILE) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_meta(records: list, path: str = META_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
