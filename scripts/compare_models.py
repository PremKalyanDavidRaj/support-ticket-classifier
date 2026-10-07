"""Compare three configurations using validation data only."""

import json
import time
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]


def main():
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    validation = pd.read_csv(ROOT / "data/processed/validation.csv")

    configurations = [
        {"name": "original", "min_df": 2, "C": 1.0},
        {"name": "keep_rare_phrases", "min_df": 1, "C": 1.0},
        {"name": "less_regularization", "min_df": 1, "C": 4.0},
    ]

    results = []
    predictions = pd.DataFrame({
        "text": validation["text"],
        "actual_category": validation["category"],
    })

    for config in configurations:
        print(f"Training: {config['name']}...", flush=True)

        model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=config["min_df"],
                sublinear_tf=True,
                max_features=50000,
            )),
            ("classifier", LogisticRegression(
                C=config["C"],
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ])

        started = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train["text"], train["category"])

        training_seconds = time.perf_counter() - started
        predicted = model.predict(validation["text"])
        predictions[config["name"]] = predicted

        results.append({
            **config,
            "accuracy": float(
                accuracy_score(validation["category"], predicted)
            ),
            "macro_f1": float(
                f1_score(
                    validation["category"],
                    predicted,
                    average="macro",
                    zero_division=0,
                )
            ),
            "training_seconds": round(training_seconds, 2),
            "vocabulary_size": len(
                model.named_steps["tfidf"].vocabulary_
            ),
        })

    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    (reports / "model_comparison.json").write_text(
        json.dumps({
            "evaluation_split": "validation",
            "results": results,
            "note": (
                "Exploratory model selection on validation data. "
                "Original saved model unchanged; test set untouched."
            ),
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions.to_csv(
        reports / "comparison_predictions.csv", index=False
    )

    print("\nValidation comparison:")
    print(
        pd.DataFrame(results)
        .sort_values("macro_f1", ascending=False)
        .round(4)
        .to_string(index=False)
    )
    print("\nYour saved model has NOT been replaced.")


if __name__ == "__main__":
    main()
