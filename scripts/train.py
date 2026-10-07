"""Train a text classifier and evaluate on validation data only."""

import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"


def scores(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
    }


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    train_path = DATA / "train.csv"
    validation_path = DATA / "validation.csv"

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Baseline: always predict the most frequent training category.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train), 1)), train["category"])
    baseline_predictions = baseline.predict(
        np.zeros((len(validation), 1))
    )

    # Save text processing and the classifier as one inference pipeline.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=50000,
        )),
        ("classifier", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])

    print("Training TF-IDF + logistic regression...", flush=True)
    started = time.perf_counter()

    # Stop rather than silently save an unconverged model.
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train["text"], train["category"])

    training_seconds = time.perf_counter() - started
    predictions = model.predict(validation["text"])

    metrics = {
        "evaluation_split": "validation",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "baseline": scores(validation["category"], baseline_predictions),
        "tfidf_logistic_regression": scores(
            validation["category"], predictions
        ),
        "training_seconds": round(training_seconds, 2),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "training_sha256": file_hash(train_path),
        "validation_sha256": file_hash(validation_path),
        "configuration": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "sublinear_tf": True,
            "max_features": 50000,
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        },
    }

    model_path = MODELS / "ticket_classifier.joblib"
    joblib.dump(model, model_path)
    metrics["model_sha256"] = file_hash(model_path)

    (REPORTS / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    per_class = classification_report(
        validation["category"],
        predictions,
        output_dict=True,
        zero_division=0,
    )
    (REPORTS / "validation_classification_report.json").write_text(
        json.dumps(per_class, indent=2) + "\n",
        encoding="utf-8",
    )

    # Save mistakes for later qualitative error analysis.
    results = validation.copy()
    results["predicted_category"] = predictions
    errors = results.loc[
        results["category"] != results["predicted_category"]
    ]
    errors.to_csv(REPORTS / "validation_errors.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print("\nModel saved to models/ticket_classifier.joblib")
    print("Validation mistakes saved to reports/validation_errors.csv")
    print("The original test split has not been evaluated.")


if __name__ == "__main__":
    main()
