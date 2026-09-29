# -*- coding: utf-8 -*-
"""Fase 3 — Embeddings com MiniLM (congelado) + índice FAISS da base canônica."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from .config import CANONICAL_PARQUET, EMBEDDING_MODEL_NAME, FAISS_INDEX, MODELS_DIR


def load_model(name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name, device="cpu")


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return emb.astype("float32")


def build_canonical_index(model: SentenceTransformer) -> tuple[np.ndarray, pd.DataFrame]:
    """Gera embeddings dos textos canônicos e o índice FAISS."""
    import faiss

    base = pd.read_parquet(CANONICAL_PARQUET)
    textos = base["texto"].fillna("").astype(str).tolist()
    # trechos curtos de contexto também ajudam no ranking
    clipes = [t[:2000] for t in textos]
    emb = encode(model, clipes)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # embeddings normalizados -> produto interno
    index.add(emb)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX))
    return emb, base


def load_faiss_index() -> "faiss.Index":
    import faiss

    return faiss.read_index(str(FAISS_INDEX))
