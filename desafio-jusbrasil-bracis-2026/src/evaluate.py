"""Avaliação sem salvar classificador: python -m src.evaluate --epochs 40."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from .config import DATA_DIR


def evaluate_pipeline(df, pipe, indices) -> dict:
    """Score oficial e diagnóstico das etapas nos documentos held-out completos."""
    from kaggle_metric import avaliar, _casar, _contida
    counts = Counter()
    confusion = Counter()
    solutions, submissions = [], []
    for doc, group in df.iloc[indices].groupby("documento_id", sort=True):
        # Uma validação parcial transformaria anotações omitidas em falsos positivos.
        if len(group) != int((df.documento_id == doc).sum()):
            raise ValueError(f"Documento de validação parcial: {doc}")
        predictions = pipe.process(group["_texto"].iloc[0])
        golds = group.to_dict("records")
        pairs, missing, extra = _casar(golds, predictions)
        counts.update(gold=len(golds), predicted=len(predictions), matched=len(pairs), missed=len(missing))
        counts["spurious"] += sum(not any(_contida(predictions[i], golds[j]) for j, _ in pairs)
                                  for i in extra)
        counts["ignored_components"] += sum(any(_contida(predictions[i], golds[j]) for j, _ in pairs)
                                            for i in extra)
        for gi, pi in pairs:
            gold, pred = golds[gi], predictions[pi]
            gc, pc = gold["classificacao"], pred["classificacao"]
            confusion[f"{gc}->{pc}"] += 1
            counts["class_correct"] += int(gc == pc)
            if gc == pc == "real":
                counts["real_pairs"] += 1
                counts["correct_id"] += int(str((pred.get("resolucao") or {}).get("id_canonico", ""))
                                            in str(gold["id_canonico"]).split(":"))
        gold_cell = "|".join(
            f"{r['inicio']},{r['fim']},{r['classificacao']},"
            f"{r['id_canonico'] if r['classificacao'] == 'real' else '-'}" for r in golds)
        from .infer import _encode_cell
        solutions.append((doc, int(group.nivel.iloc[0]), gold_cell))
        submissions.append((doc, _encode_cell(predictions)))
    solution = pd.DataFrame(solutions, columns=["documento_id", "nivel", "citacoes"])
    submission = pd.DataFrame(submissions, columns=["documento_id", "citacoes"])
    result = avaliar(solution, submission)
    result["extraction"] = {
        **{key: counts[key] for key in ("gold", "predicted", "matched", "missed", "spurious", "ignored_components")},
        "precision": counts["matched"] / max(1, counts["matched"] + counts["spurious"]),
        "recall": counts["matched"] / max(1, counts["gold"]),
    }
    result["matched_classification"] = {
        "accuracy": counts["class_correct"] / max(1, counts["matched"]),
        "confusion": dict(confusion),
    }
    result["resolution"] = {"real_pairs": counts["real_pairs"], "correct_id": counts["correct_id"],
                            "accuracy": counts["correct_id"] / max(1, counts["real_pairs"])}
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--output", type=Path, default=DATA_DIR / "evaluation.json")
    args = ap.parse_args()
    from .train import train
    result = train(epochs=args.epochs, seed=args.seed, val_frac=args.test_size, save=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"score_final: {result['score_final']:.4f}")
    for level, metrics in result["niveis"].items():
        print(f"  nível {level}: score={metrics['score']:.4f}, F1={metrics['f1_por_classe']}")
    print("Extração:", result["extraction"])
    print("Resolução:", result["resolution"])
    print(f"Relatório: {args.output}")


if __name__ == "__main__":
    main()
