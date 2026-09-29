# -*- coding: utf-8 -*-
"""Fase 6 — CLI de inferência: .txt -> JSON (contrato) e submission.csv.

Uso:
  python -m src.infer <pasta_com_txt> [--out-json pasta_json] [--submission submission.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import CitationPipeline


def _encode_cell(citacoes: list[dict]) -> str:
    partes = []
    for c in citacoes:
        classe = c["classificacao"]
        resol = c.get("resolucao") or {}
        idc = str(resol.get("id_canonico", "") or "").strip() or "-"
        conf = c.get("confianca")
        conf_s = "-" if conf is None else f"{float(conf):.4f}"
        partes.append(f"{int(c['inicio'])},{int(c['fim'])},{classe},{idc},{conf_s}")
    return "|".join(partes) if partes else "-"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("txt_dir", help="pasta com arquivos .txt")
    ap.add_argument("--out-json", default=None, help="pasta de saída dos JSONs")
    ap.add_argument("--submission", default="submission.csv")
    ap.add_argument("--sample", default=None,
                    help="sample_submission.csv com a lista/ordem oficial dos documentos")
    args = ap.parse_args(argv)

    txt_dir = Path(args.txt_dir)
    if args.sample:
        import pandas as pd
        ids = [str(x).strip() for x in pd.read_csv(args.sample)["documento_id"]]
        arquivos = []
        for sid in ids:
            p = txt_dir / f"{sid}.txt"
            if not p.exists():
                sys.exit(f".txt ausente para {sid}")
            arquivos.append(p)
    else:
        arquivos = sorted(txt_dir.glob("*.txt"))
    if not arquivos:
        sys.exit(f"nenhum .txt em {txt_dir}")

    print("carregando pipeline (modelo de embedding + classificador + índice)...")
    pipe = CitationPipeline()

    out_json = Path(args.out_json) if args.out_json else None
    if out_json:
        out_json.mkdir(parents=True, exist_ok=True)

    linhas = []
    for arq in arquivos:
        texto = arq.read_text(encoding="utf-8")
        doc = pipe.process_document(texto, documento_id=arq.stem)
        if out_json:
            (out_json / f"{arq.stem}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        linhas.append((arq.stem, _encode_cell(doc["citacoes"])))
        print(f"  {arq.stem}: {len(doc['citacoes'])} citações")

    import csv
    with Path(args.submission).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["documento_id", "citacoes"])
        w.writerows(linhas)
    print(f"submission: {args.submission} ({len(linhas)} documentos)")


if __name__ == "__main__":
    main()
