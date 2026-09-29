# -*- coding: utf-8 -*-
"""Interface Streamlit: colar/uploadar um parecer .txt e classificar citações."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CLASSIFIER_STATE, FAISS_INDEX
from src.pipeline import CitationPipeline


@st.cache_resource
def get_pipeline() -> CitationPipeline:
    if not Path(CLASSIFIER_STATE).exists() or not Path(FAISS_INDEX).exists():
        st.error(
            "Modelo ainda não treinado. Rode primeiro: `python -m src.train` "
            "e gere o índice FAISS."
        )
        st.stop()
    with st.spinner("Carregando pipeline (embedding + classificador + índice)..."):
        return CitationPipeline()


def encode_cell(citacoes: list[dict]) -> str:
    partes = []
    for c in citacoes:
        classe = c["classificacao"]
        resol = c.get("resolucao") or {}
        idc = str(resol.get("id_canonico", "") or "").strip() or "-"
        conf = c.get("confianca")
        conf_s = "-" if conf is None else f"{float(conf):.4f}"
        partes.append(f"{int(c['inicio'])},{int(c['fim'])},{classe},{idc},{conf_s}")
    return "|".join(partes) if partes else "-"


st.set_page_config(page_title="Caça-Alucinações · Verificador de Citações", layout="wide")
st.title("Verificador de Citações Jurídicas")
st.caption("BRACIS 2026 × Jusbrasil — classifica citações em real / inventada / incompleta")

pipe = get_pipeline()

col1, col2 = st.columns(2)
with col1:
    upload = st.file_uploader("Enviar parecer (.txt)", type=["txt"])
with col2:
    doc_id = st.text_input("documento_id (opcional)", value="")

texto = ""
if upload is not None:
    texto = upload.getvalue().decode("utf-8")
    if not doc_id:
        doc_id = Path(upload.name).stem

pasted = st.text_area("Ou cole o texto do parecer aqui:", height=260)

if texto == "" and pasted.strip():
    texto = pasted

if st.button("Classificar citações", type="primary") and texto:
    with st.spinner("Processando..."):
        citacoes = pipe.process(texto)

    st.subheader(f"Resultado — {len(citacoes)} citação(ões)")

    if citacoes:
        import pandas as pd
        df = pd.DataFrame([
            {
                "inicio": c["inicio"],
                "fim": c["fim"],
                "trecho": c["trecho"],
                "tipo": c["tipo"],
                "classe": c["classificacao"],
                "id_canonico": (c["resolucao"] or {}).get("id_canonico", ""),
                "confianca": c["confianca"],
            }
            for c in citacoes
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        doc = {"schema_version": "1.2", "documento_id": doc_id or "documento", "citacoes": citacoes}
        js = json.dumps(doc, ensure_ascii=False, indent=2)
        st.download_button("Baixar JSON", js, file_name=f"{doc_id or 'doc'}.json",
                           mime="application/json")
        st.download_button("Baixar submission.csv",
                           f"documento_id,citacoes\n{doc_id or 'doc'},{encode_cell(citacoes)}\n",
                           file_name="submission.csv", mime="text/csv")
    else:
        st.info("Nenhuma citação detectada.")
