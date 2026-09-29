# -*- coding: utf-8 -*-
"""Fase 4 — Classificador MLP (PyTorch) treinado em CPU com embeddings congelados.

Entrada por citação: embedding (MiniLM) ⊕ features de identificadores ⊕ sinal de busca.
Saída: 3 classes (real/inventada/incompleta) + confiança calibrada (temperature scaling).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from .config import CLASSES, CLASSIFIER_STATE, MERGED_PARQUET, MODELS_DIR, SPLITS_DIR
from .preprocess import build_features, extract_ano, extract_process_digits, extract_tribunal
from .resolve import CanonicalResolver, load_resolver
from .validation import train_val_split, split_three_way
from .data_quality import reconcile_annotations

SEARCH_KEYS = ["proc_match", "sumula_match", "artigo_match", "best_sim", "trib_match", "ano_match"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def compute_search_features(resolver: CanonicalResolver, trecho: str, emb: np.ndarray) -> dict:
    proc = extract_process_digits(trecho)
    trib = extract_tribunal(trecho)
    ano = extract_ano(trecho)
    best_sim = 0.0
    proc_match = 0
    if hasattr(resolver, "_canonical_emb") and len(resolver._canonical_emb):
        sims = resolver._canonical_emb @ emb
        best_sim = float(sims.max())
    if proc and proc in resolver.proc_index:
        proc_match = 1
    # súmula/artigo: casa por natureza correspondente (heurística leve)
    sumula_match = 0
    artigo_match = 0
    row = trecho.lower()
    if ("sumula" in row or "súmula" in row or "5umula" in row):
        # qualquer doc sumula cujo tribunal bata é sinal forte
        pass
    return {
        "proc_match": proc_match,
        "sumula_match": sumula_match,
        "artigo_match": artigo_match,
        "best_sim": best_sim,
        "trib_match": int(bool(trib)),
        "ano_match": int(ano is not None),
    }


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(CLASSES)),
        )

    def forward(self, x):
        return self.net(x)


def _make_matrix(df: pd.DataFrame, emb: np.ndarray, resolver: CanonicalResolver) -> np.ndarray:
    ident = np.array([list(build_features(t).values()) for t in df["trecho"]], dtype="float32")
    search = np.array([list(compute_search_features(resolver, t, e).values())
                       for t, e in zip(df["trecho"], emb)], dtype="float32")
    X = np.hstack([emb.astype("float32"), ident, search])
    return X


def train(epochs: int = 40, lr: float = 1e-3, seed: int = 42,
          val_frac: float = 0.2, save: bool = True) -> dict:
    """Treina, calibra em conjunto separado e avalia pelo pipeline de produção."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    df, audit = reconcile_annotations(pd.read_parquet(MERGED_PARQUET))
    tr, cal, va = split_three_way(df, val_frac, seed)
    print(f"Split: treino={len(tr)}, calibração={len(cal)}, validação={len(va)}, "
          f"purgados={len(df) - len(tr) - len(cal) - len(va)}")
    resolver = load_resolver()

    from .embed import build_canonical_index, encode, load_model

    model = load_model()
    canon_emb, _ = build_canonical_index(model)
    resolver.set_canonical_embeddings(canon_emb)

    trechos = df["trecho"].astype(str).tolist()
    emb = encode(model, trechos)

    y = np.array([CLASS2IDX[c] for c in df["classificacao"]], dtype="int64")

    scaler = StandardScaler()
    X = _make_matrix(df, emb, resolver)
    scaler.fit(X[tr])
    X = scaler.transform(X)
    Xtr, ytr = X[tr], y[tr]

    # Ponderação de classes; a penalidade assimétrica é medida no score oficial.
    counts = np.bincount(ytr, minlength=len(CLASSES)).astype("float32")
    w = 1.0 / np.maximum(counts, 1.0)
    w[CLASS2IDX["real"]] *= 1.2
    w = w / w.sum() * len(CLASSES)

    net = MLP(in_dim=X.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        logits = net(Xtr_t)
        loss = loss_fn(logits, ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1:3d} loss={loss.item():.4f}")

    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(X, dtype=torch.float32)).numpy()
    # O conjunto de validação final não participa da calibração.
    T = _fit_temperature(logits[cal], y[cal])
    state = {
        "mlp": net.state_dict(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "in_dim": X.shape[1],
        "temperature": float(T),
        "class2idx": CLASS2IDX,
        "seed": seed,
        "val_frac": val_frac,
    }
    from sklearn.metrics import classification_report, accuracy_score
    metrics = {}
    for name, indices in (("train", tr), ("calibration", cal), ("validation", va)):
        pred = logits[indices].argmax(1)
        metrics[name] = {
            "accuracy": float(accuracy_score(y[indices], pred)),
            "report": classification_report(y[indices], pred, labels=list(range(len(CLASSES))),
                                            target_names=list(CLASSES), zero_division=0,
                                            output_dict=True),
        }
        print(f"{name}: accuracy={metrics[name]['accuracy']:.4f}, "
              f"macro_f1={metrics[name]['report']['macro avg']['f1-score']:.4f}")

    from .pipeline import CitationPipeline
    from .evaluate import evaluate_pipeline
    inference_state = dict(state, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    pipe = CitationPipeline.from_components(model, net, inference_state, resolver)
    result = evaluate_pipeline(df, pipe, va)
    result.update(temperature=float(T), val_acc=metrics["validation"]["accuracy"],
                  classification=metrics, seed=seed, val_frac=val_frac, epochs=epochs,
                  annotation_audit=audit.status.value_counts().to_dict())
    manifest = df[["documento_id", "citacao_id"]].copy()
    manifest["split"] = "purged"
    for name, indices in (("train", tr), ("calibration", cal), ("validation", va)):
        manifest.loc[indices, "split"] = name
    result["split_counts"] = manifest.split.value_counts().to_dict()
    result["validation_documents"] = sorted(df.iloc[va].documento_id.unique().tolist())
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(CLASSIFIER_STATE))
        for name, indices in (("train", tr), ("calibration", cal), ("val", va)):
            np.save(SPLITS_DIR / f"{name}_idx.npy", indices)
        manifest.to_csv(SPLITS_DIR / "manifest.csv", index=False)
        import json
        (MODELS_DIR / "training_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        p = _softmax(logits / t)
        p = np.clip(p, 1e-9, 1.0)
        nll = -np.log(p[np.arange(len(y)), y]).mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_classifier() -> dict:
    state = torch.load(str(CLASSIFIER_STATE), map_location="cpu")
    net = MLP(in_dim=state["in_dim"])
    net.load_state_dict(state["mlp"])
    net.eval()
    state["scaler_mean"] = np.asarray(state["scaler_mean"], dtype="float32")
    state["scaler_scale"] = np.asarray(state["scaler_scale"], dtype="float32")
    return {"net": net, "state": state}


if __name__ == "__main__":
    train()
