"""Prepare reproducible training and validation splits."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main():
    source = ROOT / "data" / "raw" / "train.csv"
    frame = pd.read_csv(source)
    original_rows = len(frame)

    # Normalize only for duplicate detection; retain original model inputs.
    frame["_key"] = (
        frame["text"]
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove ambiguous examples with identical text but different labels.
    label_counts = frame.groupby("_key")["category"].nunique()
    conflicting_keys = label_counts[label_counts > 1].index
    conflict_mask = frame["_key"].isin(conflicting_keys)
    conflicting_rows = int(conflict_mask.sum())
    frame = frame.loc[~conflict_mask].copy()

    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset="_key")
    duplicate_rows = before_dedup - len(frame)

    if frame["category"].nunique() != 77:
        raise ValueError("Cleaning removed an entire category; inspect the data.")

    train, validation = train_test_split(
        frame,
        test_size=0.20,
        random_state=42,
        stratify=frame["category"],
    )

    overlap = set(train["_key"]) & set(validation["_key"])
    if overlap:
        raise ValueError("Duplicate text leaked across the split.")

    expected_labels = set(frame["category"])
    if set(train["category"]) != expected_labels:
        raise ValueError("Training split is missing categories.")
    if set(validation["category"]) != expected_labels:
        raise ValueError("Validation split is missing categories.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    columns = ["text", "category"]
    train[columns].to_csv(OUTPUT / "train.csv", index=False)
    validation[columns].to_csv(OUTPUT / "validation.csv", index=False)

    report = {
        "source_rows": original_rows,
        "conflicting_rows_removed": conflicting_rows,
        "duplicate_rows_removed": duplicate_rows,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "categories": len(expected_labels),
        "normalized_text_overlap": len(overlap),
        "random_seed": 42,
        "test_split": "Untouched; reserved for final evaluation",
    }

    (REPORTS / "split_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
