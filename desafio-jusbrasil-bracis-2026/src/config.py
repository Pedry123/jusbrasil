# -*- coding: utf-8 -*-
"""Configuração central do projeto: caminhos e constantes."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
TXT_DIR = BASE_DIR / "txt"

GOLDENSET_CSV = BASE_DIR / "goldenset_offsets.csv"
DATASET_XLSX = BASE_DIR / "dataset.xlsx"
CANONICAL_DB = BASE_DIR / "desafio1_bracis.db"
COMPARTILHADO_DIR = BASE_DIR / "dataset_compartilhado"

MERGED_PARQUET = DATA_DIR / "merged_citacoes.parquet"
CANONICAL_PARQUET = DATA_DIR / "canonical_base.parquet"
SPLITS_DIR = DATA_DIR / "splits"
FAISS_INDEX = MODELS_DIR / "faiss.index"
CLASSIFIER_STATE = MODELS_DIR / "classifier.pt"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CLASSES = ("real", "inventada", "incompleta")
NIVEL_WEIGHTS = {1: 1.0, 2: 2.0}
