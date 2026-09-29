# -*- coding: utf-8 -*-
"""Pipeline de inferência: .txt -> citações (classe + id_canonico + confiança)."""
from __future__ import annotations

import numpy as np
import torch

from .config import CLASSES
from .embed import build_canonical_index, encode, load_model
from .preprocess import build_features, extract_candidates, tem_artigo, tem_processo, tem_sumula
from .resolve import CanonicalResolver, load_resolver
from .train import compute_search_features, load_classifier


def refine_class(classe: str, resol: dict, trecho: str) -> tuple[str, str]:
    """Regra de decisão final (identificador determina 'real').

    - real exige casamento forte de identificador (processo/súmula/artigo);
    - identificador suficiente sem casamento -> inventada;
    - sem identificadores -> incompleta.
    """
    if classe != "real":
        return classe, ""
    if resol.get("tipo_match") == "identificador":
        return "real", resol.get("id_canonico", "")
    if tem_processo(trecho) or tem_sumula(trecho) or tem_artigo(trecho):
        return "inventada", ""
    return "incompleta", ""


class CitationPipeline:
    def __init__(self):
        self.model = load_model()
        self.clf = load_classifier()
        self.net = self.clf["net"]
        self.state = self.clf["state"]
        self.resolver = load_resolver()
        canon_emb, _ = build_canonical_index(self.model)
        self.resolver.set_canonical_embeddings(canon_emb)

    @classmethod
    def from_components(cls, model, net, state, resolver):
        """Avalia componentes em memória usando exatamente a inferência final."""
        pipe = cls.__new__(cls)
        pipe.model = model
        pipe.net = net
        pipe.state = state
        pipe.clf = {"net": net, "state": state}
        pipe.resolver = resolver
        return pipe

    def _predict_probs(self, trecho: str, emb: np.ndarray) -> np.ndarray:
        ident = np.array(list(build_features(trecho).values()), dtype="float32")
        search = np.array(list(compute_search_features(self.resolver, trecho, emb).values()),
                          dtype="float32")
        X = np.hstack([emb.astype("float32"), ident, search]).reshape(1, -1)
        X = (X - self.state["scaler_mean"]) / self.state["scaler_scale"]
        with torch.no_grad():
            logits = self.net(torch.tensor(X, dtype=torch.float32)).numpy()[0]
        logits = logits / self.state["temperature"]
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

    def process(self, text: str) -> list[dict]:
        candidatos = extract_candidates(text)
        if not candidatos:
            return []
        trechos = [c["trecho"] for c in candidatos]
        embs = encode(self.model, trechos)

        saida = []
        for i, (c, emb) in enumerate(zip(candidatos, embs)):
            trecho = c["trecho"]
            probs = self._predict_probs(trecho, emb)
            classe = CLASSES[int(probs.argmax())]
            conf = float(probs.max())

            resolucao = None
            if classe == "real":
                r = self.resolver.resolve(trecho, emb)
                classe, idc = refine_class("real", r, trecho)
                if classe == "real":
                    resolucao = {"fonte": "jusbrasil", "id_canonico": idc}
                conf = min(conf, 0.5 + 0.5 * r["confianca"])

            saida.append({
                "id": f"c{i + 1}",
                "inicio": int(c["inicio"]),
                "fim": int(c["fim"]),
                "trecho": trecho,
                "tipo": "lei" if tem_artigo(trecho) else "jurisprudencia",
                "classificacao": classe,
                "resolucao": resolucao,
                "confianca": round(conf, 4),
            })
        return saida

    def process_document(self, texto: str, documento_id: str | None = None) -> dict:
        return {
            "schema_version": "1.2",
            "documento_id": documento_id or "",
            "citacoes": self.process(texto),
        }
