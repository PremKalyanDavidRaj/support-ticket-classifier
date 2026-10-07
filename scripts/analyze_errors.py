"""Summarize validation mistakes without using the test set."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main():
    errors = pd.read_csv(REPORTS / "validation_errors.csv")
    report = json.loads(
        (REPORTS / "validation_classification_report.json")
        .read_text(encoding="utf-8")
    )

    confused = (
        errors.groupby(["category", "predicted_category"])
        .size()
        .reset_index(name="error_count")
        .sort_values(
            ["error_count", "category", "predicted_category"],
            ascending=[False, True, True],
        )
    )

    per_class = pd.DataFrame([
        {
            "category": category,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1-score"],
            "support": int(values["support"]),
        }
        for category, values in report.items()
        if isinstance(values, dict)
        and category not in {"macro avg", "weighted avg", "micro avg"}
    ]).sort_values(["f1", "category"])

    confused.to_csv(REPORTS / "confused_categories.csv", index=False)
    per_class.to_csv(REPORTS / "per_category_metrics.csv", index=False)

    print(f"\nTotal validation mistakes: {len(errors)}")

    print("\nTop 10 confusion pairs:")
    print(confused.head(10).to_string(index=False))

    print("\n10 categories with lowest F1:")
    print(per_class.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
