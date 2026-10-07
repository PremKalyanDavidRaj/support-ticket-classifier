"""Evaluate the frozen model on original and filtered test data."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(frame):
    return (
        frame["text"]
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def metrics(frame, predictions, labels):
    if frame.empty:
        raise ValueError("Evaluation subset is empty.")

    return {
        "rows": len(frame),
        "categories_present": int(frame["category"].nunique()),
        "accuracy": float(
            accuracy_score(frame["category"], predictions)
        ),
        "macro_f1": float(
            f1_score(
                frame["category"],
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():
    model_path = ROOT / "models/ticket_classifier.joblib"
    train_path = ROOT / "data/processed/train.csv"
    validation_path = ROOT / "data/processed/validation.csv"
    test_path = ROOT / "data/raw/test.csv"

    validation_report = json.loads(
        (ROOT / "reports/validation_metrics.json").read_text()
    )

    # Verify this is the model and data used in the recorded experiment.
    for path, key in [
        (model_path, "model_sha256"),
        (train_path, "training_sha256"),
        (validation_path, "validation_sha256"),
    ]:
        if digest(path) != validation_report[key]:
            raise ValueError(f"Hash mismatch for {path.name}")

    model = joblib.load(model_path)
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    test = pd.read_csv(test_path)

    development_keys = set(normalize(train)) | set(normalize(validation))
    test_keys = normalize(test)
    overlap = test_keys.isin(development_keys)

    filtered = test.loc[~overlap].copy()
    filtered["_key"] = test_keys.loc[~overlap]
    before_deduplication = len(filtered)
    filtered = filtered.drop_duplicates("_key")

    original_predictions = model.predict(test["text"])
    filtered_predictions = model.predict(filtered["text"])

    report = {
        "model_sha256": digest(model_path),
        "test_sha256": digest(test_path),
        "selected_configuration": validation_report["configuration"],
        "original_test": metrics(
            test, original_predictions, model.classes_
        ),
        "filtered_test": metrics(
            filtered, filtered_predictions, model.classes_
        ),
        "development_overlap_rows_removed": int(overlap.sum()),
        "additional_duplicate_rows_removed": (
            before_deduplication - len(filtered)
        ),
        "note": (
            "Model selected using validation data only. "
            "Filtered subset removes normalized exact-text overlap "
            "and keeps the first occurrence of repeated test text. "
            "Paraphrase overlap is not assessed."
        ),
    }

    output = ROOT / "reports/test_metrics.json"
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
