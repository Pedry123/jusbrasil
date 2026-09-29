# -*- coding: utf-8 -*-
"""Fase 1 — Merge dos datasets (goldenset oficial + dataset.xlsx da turma).

Gera:
  - data/merged_citacoes.parquet  (citações rotuladas unificadas)
  - data/canonical_base.parquet   (base canônica extraída do sqlite)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .data_quality import reconcile_annotations

from .config import (
    CANONICAL_DB,
    CANONICAL_PARQUET,
    DATASET_XLSX,
    GOLDENSET_CSV,
    MERGED_PARQUET,
    TXT_DIR,
)

COLS = [
    "nivel",
    "documento_id",
    "citacao_id",
    "inicio",
    "fim",
    "trecho",
    "tipo",
    "classificacao",
    "id_canonico",
    "autor",
    "origem",
]


def _txt_stems() -> set[str]:
    return {p.stem for p in TXT_DIR.glob("*.txt")}


def _resolve_txt(doc_id: str, stems: set[str]) -> str | None:
    """Mapeia documento_id -> nome do arquivo .txt (há nomes divergentes na turma)."""
    if doc_id in stems:
        return doc_id
    # paulo_n1_019 -> gen_n1_019paulo
    if doc_id.startswith("paulo_"):
        alt = "gen_" + doc_id[len("paulo_"):] + "paulo"
        if alt in stems:
            return alt
    return None


def _load_texts(stems: set[str]) -> dict[str, str]:
    """Carrega o texto completo de cada .txt disponível, por documento_id."""
    out: dict[str, str] = {}
    for p in TXT_DIR.glob("*.txt"):
        out[p.stem] = p.read_text(encoding="utf-8")
    return out


def _norm_id_canonico(v) -> str:
    """id_canonico pode vir como float (ex.: 5665364632.0), float científico
    (ex.: 2.566535283E9) ou str/int."""
    if v is None:
        return ""
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        v = int(v)
    s = str(v).strip()
    if s in ("", "-", "nan", "None", "NaN"):
        return ""
    if "E" in s.upper():
        try:
            return str(int(float(s)))
        except ValueError:
            return ""
    if "." in s or "," in s:
        try:
            return str(int(float(s.replace(",", "."))))
        except ValueError:
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits or s


def load_canonical() -> pd.DataFrame:
    """Base canônica: documento_id, id (canonico), tribunal, ano, relator,
    natureza, tipo, texto, texto_len."""
    conn = sqlite3.connect(CANONICAL_DB)
    df = pd.read_sql_query(
        "SELECT documento_id, id, tribunal, ano, relator, natureza, tipo, texto, texto_len "
        "FROM documentos ORDER BY rowid",
        conn,
    )
    conn.close()
    df["id"] = df["id"].astype(str)
    return df


def load_goldenset() -> pd.DataFrame:
    df = pd.read_csv(GOLDENSET_CSV, encoding="utf-8-sig")
    df = df.rename(columns=str.strip)
    df["origem"] = "oficial"
    df["autor"] = "goldenset"
    df["nivel"] = df["nivel"].astype(int)
    df["inicio"] = df["inicio"].astype(int)
    df["fim"] = df["fim"].astype(int)
    df["id_canonico"] = df["id_canonico"].apply(_norm_id_canonico)
    df["classificacao"] = df["classificacao"].str.strip().str.lower()
    df["tipo"] = df["tipo"].str.strip().str.lower()
    return df[COLS]


def load_turma() -> pd.DataFrame:
    df = pd.read_excel(DATASET_XLSX)
    df = df.rename(columns=str.strip)
    df["origem"] = "extra"
    df["nivel"] = df["nivel"].astype(int)
    df["inicio"] = df["inicio"].astype(int)
    df["fim"] = df["fim"].astype(int)
    df["id_canonico"] = df["id_canonico"].apply(_norm_id_canonico)
    df["classificacao"] = df["classificacao"].str.strip().str.lower()
    df["tipo"] = df["tipo"].str.strip().str.lower()
    df["autor"] = df["autor"].fillna("").astype(str)
    return df[COLS]


def _validate(df: pd.DataFrame, canonical_ids: set[str]) -> pd.DataFrame:
    """Exclui documentos com span, classe/tipo ou ID canônico inválidos.

    Preserva apenas documentos com gabarito completo para avaliação.
    """
    manter = []
    problemas: dict[str, int] = {}
    def _erro(tipo: str):
        problemas[tipo] = problemas.get(tipo, 0) + 1
    for i, r in df.iterrows():
        doc = r["documento_id"]
        txt = r.get("_texto")
        if txt is None or (isinstance(txt, float) and pd.isna(txt)):
            _erro(f"texto não encontrado ({doc})")
            continue
        if not (0 <= r["inicio"] < r["fim"] <= len(txt)):
            _erro(f"span inválido ({doc})")
            continue
        if r["classificacao"] not in ("real", "inventada", "incompleta"):
            _erro("classe inválida")
            continue
        if r["tipo"] not in ("jurisprudencia", "lei"):
            _erro("tipo inválido")
            continue
        if r["classificacao"] == "real":
            if not r["id_canonico"]:
                _erro("real sem id_canonico")
                continue
            if r["id_canonico"] not in canonical_ids:
                _erro("id_canonico fora da base")
                continue
        manter.append(i)
    if problemas:
        print("  [validação] descartadas:")
        for k, v in problemas.items():
            print(f"    - {v:4d} {k}")
    invalid_docs = set(df.loc[~df.index.isin(manter), "documento_id"])
    return df.loc[~df.documento_id.isin(invalid_docs)].reset_index(drop=True)


def main() -> None:
    DATA_DIR_M = MERGED_PARQUET.parent
    DATA_DIR_M.mkdir(parents=True, exist_ok=True)

    canonical = load_canonical()
    canonical.to_parquet(CANONICAL_PARQUET, index=False)
    print(f"canonical_base.parquet: {len(canonical)} registros")

    canonical_ids = set(canonical["id"])

    gold = load_goldenset()
    turma = load_turma()
    merged = pd.concat([gold, turma], ignore_index=True)

    stems = _txt_stems()
    textos = _load_texts(stems)

    # junta o texto completo para conferência de span
    merged["_txt_stem"] = merged["documento_id"].apply(lambda d: _resolve_txt(d, stems))
    merged["_texto"] = merged["_txt_stem"].map(textos)

    merged, audit = reconcile_annotations(merged)
    validated = _validate(merged, canonical_ids)
    invalid_docs = set(merged.documento_id) - set(validated.documento_id)
    if invalid_docs:
        audit = pd.concat([audit, pd.DataFrame([
            {"documento_id": doc, "status": "rotulo_ou_id_invalido", "documento_excluido": True}
            for doc in sorted(invalid_docs)
        ])], ignore_index=True)
        audit.loc[audit.documento_id.isin(invalid_docs), "documento_excluido"] = True
    merged = validated
    audit.to_csv(DATA_DIR_M / "annotation_audit.csv", index=False)
    print("Auditoria:", audit["status"].value_counts().to_dict())
    print("Documentos excluídos:", audit.loc[audit.documento_excluido, "documento_id"].nunique())

    merged["texto_len"] = merged["_texto"].apply(lambda t: len(t) if t else 0)
    merged = merged.drop(columns=["_txt_stem"])

    merged.to_parquet(MERGED_PARQUET, index=False)
    print(f"merged_citacoes.parquet: {len(merged)} citações")
    print(merged["classificacao"].value_counts().to_dict())
    print(merged["origem"].value_counts().to_dict())


if __name__ == "__main__":
    main()
