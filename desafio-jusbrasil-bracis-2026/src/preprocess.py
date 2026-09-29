# -*- coding: utf-8 -*-
"""Fase 2 — Pré-processamento: extração de candidatos, normalização de
identificadores e feature engineering.

O detector de candidatos é o elo mais frágil da inferência (a métrica alinha
spans por IoU em codepoints); aqui extraímos spans âncora (número de processo,
súmula, artigo, referência vaga) com janela de contexto controlada.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- normalização

_NORM_MAP = str.maketrans({
    "ª": "a", "º": "o", "°": "o", "ª": "a", "¹": "1", "²": "2", "³": "3",
    "\xa0": " ", "—": "-", "–": "-",
})


def norm_text(s: str) -> str:
    return s.translate(_NORM_MAP)


def norm_digits(s: str) -> str:
    """Apenas dígitos (para casar número de processo independente de pontuação)."""
    return re.sub(r"\D", "", s)


def norm_identifier(s: str) -> str:
    """Normalização 'forte' para casar variantes de superfície de um identificador."""
    s = norm_text(s)
    s = s.replace("ú", "u").replace("Ú", "U")
    s = re.sub(r"[.\-–—\s]", "", s)
    s = s.lower()
    s = s.replace("nº", "").replace("no", "").replace("n", "")
    return s


_TRIBUNAL_RE = re.compile(
    r"\b(stf|stj|stm|tse|tst|trf|tj[a-z]{0,2}|tribunal)\b", re.IGNORECASE
)

_PROC_NUM_CNJ = re.compile(r"\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}")
_PROC_NUM_SIMPLES = re.compile(r"\d{1,3}\.\d{3}\.\d{3}(?:[-/]\d{1,3})?")
_SUMULA_RE = re.compile(r"[s5]ú?mula", re.IGNORECASE)
_ARTIGO_RE = re.compile(r"\b(art|artigo|arts)\b\.?", re.IGNORECASE)


def tem_processo(trecho: str) -> bool:
    return bool(_PROC_NUM_CNJ.search(trecho) or _PROC_NUM_SIMPLES.search(trecho))


def tem_sumula(trecho: str) -> bool:
    return bool(_SUMULA_RE.search(trecho))


def tem_artigo(trecho: str) -> bool:
    return bool(_ARTIGO_RE.search(trecho))


def extract_process_digits(trecho: str) -> str:
    """Devolve a maior sequência de dígitos do trecho (ignorando separadores),
    como assinatura do número de processo/identificador."""
    m = _PROC_NUM_CNJ.search(trecho)
    if m:
        return norm_digits(m.group(0))
    m = _PROC_NUM_SIMPLES.search(trecho)
    if m:
        return norm_digits(m.group(0))
    # fallback: sequência de dígitos relevante (>= 4)
    digitos = re.findall(r"\d[\d.\-/]{3,}", trecho)
    if digitos:
        return norm_digits(max(digitos, key=len))
    return ""


def extract_tribunal(trecho: str) -> str:
    m = _TRIBUNAL_RE.search(trecho)
    return m.group(1).lower() if m else ""


def extract_ano(trecho: str) -> int | None:
    m = re.search(r"\b(19|20)\d{2}\b", trecho)
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- features

def build_features(trecho: str) -> dict:
    """Vetor de features de identificadores para uma citação (sem embedding)."""
    trecho = norm_text(trecho)
    proc = extract_process_digits(trecho)
    return {
        "tem_processo": int(tem_processo(trecho)),
        "tem_sumula": int(tem_sumula(trecho)),
        "tem_artigo": int(tem_artigo(trecho)),
        "tem_tribunal": int(bool(extract_tribunal(trecho))),
        "tem_ano": int(extract_ano(trecho) is not None),
        "len_proc": len(proc),
        "n_palavras": len(trecho.split()),
        "tem_n": int(bool(re.search(r"\bn[º°o]\b|\bno\s*\d", trecho, re.IGNORECASE))),
    }


FEATURE_KEYS = [
    "tem_processo",
    "tem_sumula",
    "tem_artigo",
    "tem_tribunal",
    "tem_ano",
    "len_proc",
    "n_palavras",
    "tem_n",
]

# --------------------------------------------------------------------------- extração de candidatos (inferência)

_CLASS_TOKEN = (
    r"EREsp|EAg|CP|AgReg|REspEl|RO|ROT|AIRR|RRAg|EDCiv|EIN|Infringentes|Nulidade|Extraordinário|Cível|TST|AgARR|H|C|AgInt|AgRg|AgR|AgREsp|EDcl|EDv|ED|AREspEl|AREsp|ARE|RESP|REsp|REspe|RE|Rcl|Recl|RHC|RMS|APL|RSE|"
    r"AI|ARR|RR|AR|HC|MS|Rp|Ag|Rec|Agravo|Embargos|Reclamação|Apelação|Recurso|Habeas|"
    r"Corpus|Mandado|Segurança|Suspensão|Liminar|Sentença|Processo|Especial|Interno|"
    r"Eleitoral|Regimental|Criminal|Ordinário|Declaração|Divergência|Terceiro|Primeira|Segunda|Terceira|Turma|Instrumento|Esp|Int|Reg|E"
)

# cadeia de classes processuais terminando no número:
# 'Embargos de Declaração no Recurso em Mandado de Segurança nº ...'
_CLASS_RE = re.compile(r"\b(?:" + _CLASS_TOKEN + r")\b", re.IGNORECASE)

# cauda de uma cadeia: 'Mandado de ' / 'Liminar e de ' terminando no início da classe
_CHAIN_TAIL = re.compile(
    r"\b(?:" + _CLASS_TOKEN + r")\b\s*(?:(?:de|do|da|dos|das|no|na|nos|nas|em|e|ou)\b\s*)*$",
    re.IGNORECASE,
)

_VAGUE_RE = re.compile(
    r"\b(?:julgados?|precedentes?|ac[óo]rd[ãa]os?|decis[ãa]o|decis[õo]es|"
    r"orienta[çc][ãa]o|orienta[çc][õo]es|entendimentos?|jurisprud[êe]ncias?|"
    r"teses?|enunciados?)\b",
    re.IGNORECASE,
)

# referência a órgão julgador (abreviatura de tribunal ou órgão colegiado)
_CORTE_RE = re.compile(
    r"\b(?:stf|stj|stm|tse|tst|trf|tj[a-z]{0,2}|tribunal|tribunais|supremos?|c[ôo]rtes?|"
    r"turmas?|se[çc][ãa]o|se[çc][õo]es|plen[áa]rios?|c[âa]maras?)\b",
    re.IGNORECASE,
)

# comentário metalinguístico após vírgula ('..., sem número ...') => fim do span.
# Aqui usamos a lógica INVERSA: a vírgula continua a citação apenas se introduz
# ano ('de 2023'), relatoria ('da relatoria', 'Rel.') ou diploma ('do STJ');
# qualquer outro conteúdo após a vírgula é tratado como comentário/oração relativa.
_VAGUE_CONTINUA = re.compile(
    r"\s*(?:d[oa]s?\b|dos\b|de\s+(?:19|20)\d{2}|rel\b|relat[ao]ri[ao]\b|julgad[oa]\s+em\b)",
    re.IGNORECASE,
)

# 'o acórdão recorrido' / 'a decisão impugnada' = prosa, não é citação
_PROSE_APOS_VAGUE = re.compile(
    r"\s*\b(?:recorrid[oa]s?|impugnad[oa]s?|vergastad[oa]s?|objurgad[oa]s?|"
    r"guerread[oa]s?|atacad[oa]s?|embargad[oa]s?|agravad[oa]s?|hostilizad[oa]s?)\b",
    re.IGNORECASE,
)

# número de processo (com separadores) e UF opcional logo em seguida
_NUM = re.compile(r"(?<!\w)[\dlI](?:[\dGgOoIlSB]|(?:[.\-/]\s*)+(?=[\dGgOoIlSB])|[ \t\n]+(?=\d{1,4}\b))*")
_UF = re.compile(r"\s*(?:[-/]\s*|\(\s*)?[A-Z]{2}\b\)?")

# súmula / artigo com continuação (nome do diploma)
_SUMULA_TEMA = re.compile(
    r"(?:[s5][úu]?m(?:ula|\.)\s+(?:vinculante\s+)?\d+(?:\.\d{3})*|tem[aã]s?\.?\s+\d+(?:\.\d{3})*)",
    re.IGNORECASE,
)
_ARTIGO = re.compile(r"\b(?:art|artigo|arts)\.?\s+\d+(?:\.\d{3})*[º°o]?", re.IGNORECASE)

# o que pode seguir uma vírgula *dentro* da citação de artigo:
# inciso (I/IV/XXIX), §, letra ('g') ou introdução do diploma (do/da/dos/das)
_ARTIGO_CONTINUA = re.compile(
    r"\s*(?:[ivxlcdm]{1,6}\.?(?!\w)|§\s*\d+[º°o]?-?\w?|[\"']\w[\"']|do\b|da\b|dos\b|das\b)",
    re.IGNORECASE,
)

# 'do STJ' / 'da repercussão geral' após 'Súmula N' / 'Tema N'
_DO_DA_RE = re.compile(r"\s*d[oa]s?\b", re.IGNORECASE)

# ', Rel. Min. NOME' após um ano em citação incompleta ('Rcl de 2021, Rel. Min. X')
_REL_APOS = re.compile(r"\s*,?\s*rel\.?\s*min\.?", re.IGNORECASE)


def _span_to_codepoints(text: str, m_start: int, m_end: int) -> tuple[int, int]:
    """Regex do Python já retorna offsets em codepoints."""
    return m_start, m_end


def _back_to_class(text: str, num_start: int) -> int:
    """Aceita apenas uma cadeia processual contígua ao número."""
    lo = max(0, num_start - 200)
    window = text[lo:num_start]
    allowed = re.compile(
        r"(?:(?:" + _CLASS_TOKEN + r"|de|do|da|dos|das|no|na|nos|nas|em|e|ou|n|o|stf|stj|tst|stm|tse)\b|[\s,.°ºª-])+",
        re.IGNORECASE,
    )
    for m in _CLASS_RE.finditer(window):
        tail = window[m.start():]
        if not re.search(r"\n\s*\n", tail) and allowed.fullmatch(tail):
            return lo + m.start()
    return num_start


def _extend_orgao(text: str, end: int) -> int:
    """Estende 'Súmula N'/'Tema N' até 'do STJ'/'da repercussão geral'."""
    m = _DO_DA_RE.match(text, end)
    if not m:
        return end
    i = m.end()

    def _word(p: int) -> int:
        while p < len(text) and text[p].isalpha():
            p += 1
        return p

    def _skip(p: int) -> int:
        while p < len(text) and text[p] in " \t\n":
            p += 1
        return p

    i = _skip(i)
    w1 = i
    i = _word(i)
    if i == w1:
        return end
    if text[w1:i].isupper() and i - w1 <= 5:
        return i  # abreviatura de tribunal (STJ/TST/STF...)
    # segunda palavra ('repercussão geral', 'Federal')
    i = _skip(i)
    i = _word(i)
    return i


def _extend_relator(text: str, end: int) -> int:
    """Estende 'Rcl de 2021' até ', Rel. Min. NOME' (citação incompleta sem nº)."""
    m = _REL_APOS.match(text, end)
    if not m:
        return end
    i = m.end()
    while i < len(text) and text[i] not in ",.\n":
        i += 1
    return i


def _vague_end(text: str, s: int) -> int:
    """Fim do span de uma referência vaga: até o ponto final, quebra de parágrafo
    ('\n\n'), ou a vírgula que introduz comentário metalinguístico
    ('..., sem número ...'). Cruza quebras de linha simples (wrap de linha)."""
    n = len(text)
    e = s + 1
    while e < n:
        ch = text[e]
        if ch == ".":
            e += 1
            break
        if ch == "\n":
            if e + 1 < n and text[e + 1] == "\n":
                break
            e += 1
            continue
        if ch == ",":
            if not _VAGUE_CONTINUA.match(text, e + 1):
                break
        e += 1
    return e


def _extend_artigo(text: str, start: int, end: int) -> int:
    """Estende 'art. N' até o fim do diploma, atravessando incisos
    ('art. 373, I, do CPC') e parando na vírgula que introduz comentário
    ('..., sob pena de nulidade')."""
    n = len(text)
    i = end
    while i < n:
        ch = text[i]
        if ch == ".":
            if i > 0 and i + 1 < n and text[i - 1].isdigit() and text[i + 1].isdigit():
                i += 1
                continue  # separador de milhar no número da lei
            break
        if ch == "\n":
            if i + 1 < n and text[i + 1] == "\n":
                break
            i += 1
            continue
        if ch in ",;":
            if _ARTIGO_CONTINUA.match(text, i + 1):
                i += 1
                continue
            break
        i += 1
    return i


def extract_candidates(text: str) -> list[dict]:
    """Retorna candidatos: {inicio, fim, trecho}. Offsets em codepoints."""
    original = text
    text = norm_text(text).replace("\\n", "  ")  # preserva comprimento e offsets
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    # 1) citações de processo: número + UF, expandindo para trás até a classe
    for m in _NUM.finditer(text):
        s, e = m.start(), m.end()
        # trim separadores finais
        while e > s and text[e - 1] in "-./":
            e -= 1
        if e - s < 4:
            continue
        uf = _UF.match(text, e)
        if uf:
            e = uf.end()
        start = _back_to_class(text, s)
        if start == s:
            continue  # número isolado, folha, valor ou protocolo não basta
        line_start = text.rfind("\n", 0, start) + 1
        if (start < 600 and not text[line_start:start].strip()
                and re.match(r"processo\b", text[start:s], re.IGNORECASE)):
            continue  # processo do próprio documento no cabeçalho
        # ano solto ('2024') sem classe processual antes não é citação (é data)
        if re.fullmatch(r"(?:19|20)\d{2}", m.group(0)):
            if start == s and not uf:
                continue
            # 'Rcl de 2021' (citação incompleta sem nº) + ', Rel. Min. NOME'
            e = _extend_relator(text, e)
        spans.append((start, e))

    # 2) súmula / tema
    for m in _SUMULA_TEMA.finditer(text):
        start = m.start()
        end = _extend_orgao(text, m.end())
        spans.append((start, end))

    # 3) artigo (estende até o diploma)
    for m in _ARTIGO.finditer(text):
        start = m.start()
        end = _extend_artigo(text, start, m.end())
        spans.append((start, end))

    # 4) referências vagas (precedente/julgado/jurisprudência + corte/tribunal/ano)
    for m in _VAGUE_RE.finditer(text):
        s = m.start()
        if _PROSE_APOS_VAGUE.match(text, m.end()):
            continue
        e = _vague_end(text, s)
        if e - s > 200:
            continue
        # órgão julgador ou ano precisa estar DENTRO do span (não em frase seguinte)
        span = text[s:e]
        if not (_CORTE_RE.search(span) or re.search(r"\b(19|20)\d{2}\b", span)):
            continue
        if re.search(r"\b[ée]\s+firme\b", span, re.IGNORECASE):
            continue  # afirmação genérica, sem individualizar precedente
        spans.append((s, e))

    # converte para codepoints e deduplica
    out = []
    for s, e in spans:
        if e <= s:
            continue
        cs, ce = _span_to_codepoints(text, s, e)
        if cs >= ce or (cs, ce) in seen:
            continue
        seen.add((cs, ce))
        out.append({"inicio": cs, "fim": ce, "trecho": original[cs:ce]})
    out.sort(key=lambda x: (x["inicio"], x["fim"]))
    return _non_overlapping(out)


def _iou(a, b) -> float:
    inter = max(0, min(a["fim"], b["fim"]) - max(a["inicio"], b["inicio"]))
    if inter == 0:
        return 0.0
    union = (a["fim"] - a["inicio"]) + (b["fim"] - b["inicio"]) - inter
    return inter / union


def _non_overlapping(spans: list[dict]) -> list[dict]:
    """Remove candidatos aninhados/sobrepostos mantendo o mais longo."""
    kept: list[dict] = []
    for sp in spans:
        replaced = False
        for i, k in enumerate(kept):
            if _iou(sp, k) >= 0.5:
                if (sp["fim"] - sp["inicio"]) > (k["fim"] - k["inicio"]):
                    kept[i] = sp
                replaced = True
                break
        if not replaced:
            kept.append(sp)
    kept.sort(key=lambda x: (x["inicio"], x["fim"]))
    return kept
