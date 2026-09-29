# -*- coding: utf-8 -*-
"""Fase 5 — Resolução do id_canonico.

Sinais (em ordem de força):
  1. Número de processo normalizado (dígitos) casando o cabeçalho do acórdão.
  2. Súmula / dispositivo (natureza na base é minúscula): número + tribunal + texto.
  3. Similaridade de embedding (FAISS) com bônus de tribunal/ano.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import CANONICAL_PARQUET
from .preprocess import (
    extract_ano,
    extract_process_digits,
    extract_tribunal,
    norm_digits,
    tem_artigo,
    tem_sumula,
)

_PROC_TOK = re.compile(r"\d[\d.\-/]{3,}")


def _extract_proc_tokens(texto: str) -> set[str]:
    """Tokens numéricos com separadores no cabeçalho (candidatos a nº de processo)."""
    toks = set()
    for m in _PROC_TOK.finditer(texto[:4000]):
        d = norm_digits(m.group(0))
        if len(d) >= 5:
            toks.add(d)
    return toks


def _extract_numero(trecho: str, pat: re.Pattern) -> str:
    m = pat.search(trecho)
    return norm_digits(m.group(0)) if m else ""


class CanonicalResolver:
    def __init__(self, base: pd.DataFrame):
        self.base = base.reset_index(drop=True)
        self.base["_trib"] = self.base["tribunal"].fillna("").astype(str).str.lower()
        self.base["_nat"] = self.base["natureza"].fillna("").astype(str).str.lower()
        self.base["_texto_lower"] = self.base["texto"].fillna("").astype(str).str.lower()

        # índice: token de processo -> lista de índices de linha
        self.proc_index: dict[str, list[int]] = {}
        for i, texto in enumerate(self.base["texto"].fillna("").astype(str)):
            for tok in _extract_proc_tokens(texto):
                self.proc_index.setdefault(tok, []).append(i)

        self.ids = self.base["id"].astype(str).tolist()

    # ------------------------------------------------------------------ scoring
    def _score_doc(self, trecho: str, proc: str, trib: str, ano: int | None,
                   num_sumula: str, num_artigo: str, emb: np.ndarray | None,
                   i: int) -> float:
        score = 0.0
        row = self.base.iloc[i]
        nat = row["_nat"]

        if proc:
            if proc in self.proc_index and i in self.proc_index[proc]:
                score += 100.0
        if tem_sumula(trecho) and num_sumula:
            if nat == "sumula" and num_sumula in row["_texto_lower"]:
                score += 80.0
        if tem_artigo(trecho) and num_artigo:
            if nat == "dispositivo" and num_artigo in row["_texto_lower"]:
                score += 80.0

        if trib and row["_trib"] == trib:
            score += 8.0
        if ano is not None and row["ano"] == ano:
            score += 4.0

        if emb is not None:
            score += float(emb @ self._canonical_emb[i]) * 10.0
        return score

    def set_canonical_embeddings(self, emb: np.ndarray):
        self._canonical_emb = emb.astype("float32")

    # ------------------------------------------------------------------ resolve
    def resolve(self, trecho: str, emb: np.ndarray | None = None,
                topk: int = 50) -> dict:
        """Retorna {id_canonico, confianca, indice, tipo_match}.

        Prioridade: casamento exato de identificador (processo/súmula/artigo)
        sobre TODA a base; só então fallback semântico por embedding.
        """
        proc = extract_process_digits(trecho)
        trib = extract_tribunal(trecho)
        ano = extract_ano(trecho)
        num_sumula = _extract_numero(trecho, re.compile(r"[sS5]ú?mula\s+(?:[vV]inculante\s+)?(\d+)"))
        num_artigo = _extract_numero(trecho, re.compile(r"\b(?:art|artigo)\.?\s+(\d+)"))

        # 1) candidatos por identificador forte (sobre toda a base)
        cands: list[int] = []
        if proc and proc in self.proc_index:
            cands = self.proc_index[proc]
        if tem_sumula(trecho) and num_sumula:
            cands += [i for i in range(len(self.base))
                      if self.base.iloc[i]["_nat"] == "sumula"
                      and num_sumula in self.base.iloc[i]["_texto_lower"]]
        if tem_artigo(trecho) and num_artigo:
            cands += [i for i in range(len(self.base))
                      if self.base.iloc[i]["_nat"] == "dispositivo"
                      and num_artigo in self.base.iloc[i]["_texto_lower"]]

        if cands:
            ordem = list(dict.fromkeys(cands))
            tipo_match = "identificador"
        elif emb is not None and hasattr(self, "_canonical_emb"):
            sims = self._canonical_emb @ emb
            ordem = list(np.argsort(-sims)[:topk])
            tipo_match = "semantico"
        else:
            ordem = list(range(len(self.base)))
            tipo_match = "nenhum"

        best_i, best_s = -1, -1.0
        for i in ordem:
            s = self._score_doc(trecho, proc, trib, ano, num_sumula, num_artigo, emb, i)
            if s > best_s:
                best_i, best_s = i, s

        conf = min(1.0, max(0.0, best_s / 110.0))
        return {
            "id_canonico": self.ids[best_i] if best_i >= 0 else "",
            "confianca": conf,
            "indice": best_i,
            "tipo_match": tipo_match,
            "score": best_s,
        }


def load_resolver() -> CanonicalResolver:
    base = pd.read_parquet(CANONICAL_PARQUET)
    return CanonicalResolver(base)
