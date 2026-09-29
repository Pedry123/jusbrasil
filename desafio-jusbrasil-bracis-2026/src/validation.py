"""Split por documento, com duplicatas textuais removidas apenas do treino."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .data_quality import text_key


def train_val_split(df: pd.DataFrame, val_frac: float = 0.2,
                    seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    # Textos completos equivalentes devem permanecer no mesmo grupo, mesmo
    # quando aparecem com documento_id diferente.
    doc_keys = {}
    for doc, group in df.groupby("documento_id", sort=False):
        texts = group["_texto"].unique()
        if len(texts) != 1:
            raise ValueError(f"Documento {doc} possui textos divergentes")
        doc_keys[doc] = hashlib.sha256(text_key(texts[0]).encode()).hexdigest()
    groups = df.documento_id.map(doc_keys).to_numpy()
    if len(set(groups)) < 2:
        raise ValueError("São necessários ao menos dois documentos distintos para separar os dados")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train, val = next(splitter.split(df, groups=groups))
    # Compartilhar uma lei muito citada não deve unir quase todo o corpus em
    # um único grupo. Preservamos a validação completa e purgamos os trechos
    # equivalentes do lado do treino.
    keys = df.trecho.map(text_key).to_numpy()
    heldout = set(keys[val])
    train = np.asarray([i for i in train if keys[i] not in heldout], dtype=np.int64)
    if not len(train):
        raise ValueError("Treino vazio após remover citações equivalentes às da validação")
    return train, val


def split_three_way(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    development, val = train_val_split(df, val_frac, seed)
    fit, calibration = train_val_split(df.iloc[development].reset_index(drop=True), 0.2, seed + 1)
    fit, calibration = development[fit], development[calibration]
    if set(df.iloc[fit].classificacao) != {"real", "inventada", "incompleta"}:
        raise ValueError("O treino deve conter as três classes; revise a divisão dos documentos")
    return fit, calibration, val
