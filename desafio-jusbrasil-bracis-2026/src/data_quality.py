"""Conferência conservadora das anotações, sem alterar as fontes originais."""
from __future__ import annotations

import re
import pandas as pd

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def annotation_text(text: str) -> str:
    return (str(text).replace("\\n", "\n").replace("\\r", "\r")
            .replace("\\t", "\t").translate(_QUOTES))


def text_key(text: str) -> str:
    return " ".join(annotation_text(text).split()).casefold()


def reconcile_annotations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aceita equivalência de formatação ou realocação única e verificável.

    Casos ambíguos não são corrigidos por proximidade. Exclui o documento
    inteiro se restarem anotações inválidas, para não avaliar gabarito parcial.
    """
    out = df.copy()
    audit = []
    invalid_docs = set()
    for col in ("trecho", "inicio", "fim"):
        if col + "_original" not in out:
            out[col + "_original"] = out[col]
    for idx, row in out.iterrows():
        text = row.get("_texto")
        start, end = int(row.inicio), int(row.fim)
        snippet = str(row.trecho)
        status = "ok"
        if not isinstance(text, str):
            status = "texto_ausente"
        else:
            actual = text[start:end] if 0 <= start < end <= len(text) else None
            normalized = annotation_text(snippet)
            if actual == snippet:
                pass
            elif actual is not None and (
                text_key(normalized) == text_key(actual)
                or text_key(normalized) == text_key(re.sub(r"\s*\n\s*", " / ", actual))
            ):
                status = "formatacao"
            else:
                # Traduzir aspas preserva comprimento e, portanto, offsets.
                tokens = normalized.split()
                pattern = r"\s+".join(re.escape(t) for t in tokens)
                matches = list(re.finditer(pattern, text.translate(_QUOTES))) if tokens else []
                if len(matches) == 1:
                    start, end = matches[0].span()
                    status = "offset_corrigido"
                else:
                    status = "ambiguo" if matches else "trecho_nao_localizado"
            if status in {"ok", "formatacao", "offset_corrigido"}:
                out.at[idx, "inicio"] = start
                out.at[idx, "fim"] = end
                out.at[idx, "trecho"] = text[start:end]
        if status not in {"ok", "formatacao", "offset_corrigido"}:
            invalid_docs.add(row.documento_id)
        audit.append(dict(documento_id=row.documento_id, citacao_id=row.get("citacao_id", idx),
                          status=status, inicio_original=int(row.inicio), fim_original=int(row.fim),
                          inicio=start, fim=end, trecho_original=snippet,
                          trecho=out.at[idx, "trecho"]))

    # A métrica oficial pressupõe anotações disjuntas.
    duplicate = out.duplicated(["documento_id", "inicio", "fim", "classificacao", "id_canonico"])
    out = out.loc[~duplicate].copy()
    for doc, group in out.groupby("documento_id"):
        group = group.sort_values(["inicio", "fim"])
        if any(group.inicio.to_numpy()[1:] < group.fim.cummax().to_numpy()[:-1]):
            invalid_docs.add(doc)
            audit.append(dict(documento_id=doc, status="anotacoes_sobrepostas"))
    report = pd.DataFrame(audit)
    report["documento_excluido"] = report.documento_id.isin(invalid_docs)
    return out.loc[~out.documento_id.isin(invalid_docs)].reset_index(drop=True), report
