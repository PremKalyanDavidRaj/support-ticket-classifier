"""Check normalized text overlap without evaluating test predictions."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def normalized_text(frame):
    return (
        frame["text"]
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def main():
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    validation = pd.read_csv(ROOT / "data/processed/validation.csv")
    test = pd.read_csv(ROOT / "data/raw/test.csv")

    train_keys = set(normalized_text(train))
    validation_keys = set(normalized_text(validation))
    test_keys = normalized_text(test)

    overlaps_train = test_keys.isin(train_keys)
    overlaps_validation = test_keys.isin(validation_keys)
    overlaps_development = overlaps_train | overlaps_validation

    report = {
        "test_rows": len(test),
        "test_rows_matching_training": int(overlaps_train.sum()),
        "test_rows_matching_validation": int(overlaps_validation.sum()),
        "test_rows_matching_either": int(overlaps_development.sum()),
        "duplicate_test_rows_beyond_first": int(
            test_keys.duplicated().sum()
        ),
        "note": (
            "Normalized exact-text audit only; does not detect "
            "paraphrases. No predictions made or files modified."
        ),
    }

    output = ROOT / "reports/test_overlap_audit.json"
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
