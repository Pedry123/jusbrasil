import unittest
from itertools import combinations

import numpy as np
import pandas as pd

from src.data_quality import reconcile_annotations, text_key
from src.preprocess import extract_candidates
from src.validation import split_three_way


def annotation(text, snippet, start, end, doc="a", **kwargs):
    return dict(_texto=text, trecho=snippet, inicio=start, fim=end,
                documento_id=doc, citacao_id="c1", classificacao="real", id_canonico="1", **kwargs)


class AnnotationQualityTests(unittest.TestCase):
    def test_formatting_preserves_offsets_and_original(self):
        text = "art. 1º, ‘g’, da Lei\n64/1990"
        df, audit = reconcile_annotations(pd.DataFrame([
            annotation(text, "art. 1º, 'g', da Lei\\n64/1990", 0, len(text))]))
        self.assertEqual(df.iloc[0].trecho, text)
        self.assertEqual(df.iloc[0].fim, len(text))
        self.assertIn("\\n", df.iloc[0].trecho_original)
        self.assertEqual(audit.iloc[0].status, "formatacao")

    def test_slash_line_marker_is_not_confused_with_process_separator(self):
        text = "HC\n188.532/SP"
        df, audit = reconcile_annotations(pd.DataFrame([
            annotation(text, "HC / 188.532/SP", 0, len(text))]))
        self.assertEqual(df.iloc[0].trecho, text)
        self.assertEqual(audit.iloc[0].status, "formatacao")

    def test_unique_relocation_repairs_off_by_one(self):
        text = "Veja a Súmula 55 do STM."
        snippet = "Súmula 55 do STM"
        start = text.index(snippet)
        df, audit = reconcile_annotations(pd.DataFrame([
            annotation(text, snippet, start + 1, start + 1 + len(snippet))]))
        self.assertEqual(df.iloc[0].inicio, start)
        self.assertEqual(audit.iloc[0].status, "offset_corrigido")

    def test_ambiguous_relocation_excludes_whole_document(self):
        text = "HC 1234 e HC 1234. Súmula 55."
        df, audit = reconcile_annotations(pd.DataFrame([
            annotation(text, "HC 1234", 1, 8),
            annotation(text, "Súmula 55", 19, 28),
        ]))
        self.assertTrue(df.empty)
        self.assertTrue(audit.documento_excluido.all())
        self.assertIn("ambiguo", audit.status.tolist())

    def test_overlapping_annotations_are_not_evaluated_as_partial_gold(self):
        df, audit = reconcile_annotations(pd.DataFrame([
            annotation("HC 1234/SP", "HC 1234/SP", 0, 10),
            annotation("HC 1234/SP", "1234", 3, 7),
        ]))
        self.assertTrue(df.empty)
        self.assertIn("anotacoes_sobrepostas", audit.status.tolist())


class ExtractionTests(unittest.TestCase):
    def test_metadata_and_distant_prose_are_not_citations(self):
        text = ("Processo nº 1292746-27.2020.7.13.1173\n"
                "OAB/SP 123456. Folhas 762/872. Valor: R$ 123.456.\n\n"
                "Julgue-se o recurso.\n\nCuritiba, 3 de janeiro de 2025.")
        self.assertEqual(extract_candidates(text), [])

    def test_process_reference_remains_in_body(self):
        text = "Conforme o Processo nº 1292746-27.2020.7.13.1173, aplica-se a regra."
        self.assertEqual(extract_candidates(text)[0]["trecho"], "Processo nº 1292746-27.2020.7.13.1173")

    def test_thousands_in_article_and_theme(self):
        text = "Aplica-se o art. 1.134 da Lei nº 13.105/2015. Veja o Tema 2.680 da repercussão geral."
        spans = [c["trecho"] for c in extract_candidates(text)]
        self.assertEqual(spans, ["art. 1.134 da Lei nº 13.105/2015", "Tema 2.680 da repercussão geral"])

    def test_ocr_and_abbreviations_preserve_original_offsets(self):
        snippets = ["Rcl 88.178/RS", "REsp n. l.741.784/PR", "TST-Ag-AIRR-373-80.2021.5.05.0341",
                    "Reclamação n. 44-\n.921 (PE)", "AgRg no H.C. Nº 891369 (RS)"]
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                text = "⚖ Veja " + snippet + ", conforme decidido."
                candidates = extract_candidates(text)
                self.assertEqual(len(candidates), 1)
                candidate = candidates[0]
                self.assertEqual(candidate["trecho"], snippet)
                self.assertEqual(text[candidate["inicio"]:candidate["fim"]], snippet)


class SplitTests(unittest.TestCase):
    def test_documents_and_equivalent_excerpts_do_not_leak(self):
        rows = []
        for i in range(30):
            for j, label in enumerate(("real", "inventada", "incompleta")):
                rows.append(dict(documento_id=f"doc{i}", _texto=f"Documento completo {i}",
                                 trecho=f"Citação {i} classe {j}", classificacao=label))
            rows.append(dict(documento_id=f"doc{i}", _texto=f"Documento completo {i}",
                             trecho="Citação  comum" if i % 2 else "Citação\ncomum",
                             classificacao="real"))
        # Documento duplicado com outro ID.
        for r in rows[:4]:
            rows.append(dict(r, documento_id="copy"))
        df = pd.DataFrame(rows)
        parts = split_three_way(df)
        for a, b in combinations(parts, 2):
            self.assertFalse(set(df.iloc[a].documento_id) & set(df.iloc[b].documento_id))
            self.assertFalse(set(df.iloc[a].trecho.map(text_key)) & set(df.iloc[b].trecho.map(text_key)))
            self.assertFalse(set(df.iloc[a]._texto.map(text_key)) & set(df.iloc[b]._texto.map(text_key)))
        val = df.iloc[parts[2]]
        for doc, group in val.groupby("documento_id"):
            self.assertEqual(len(group), int((df.documento_id == doc).sum()))
        for a, b in zip(parts, split_three_way(df)):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
