# Support Ticket Classifier

A machine learning API that classifies banking customer-support
messages into 77 intent categories using TF-IDF and logistic regression.

## Results

The selected model uses word unigrams and bigrams, `min_df=1`,
and logistic regression with `C=4.0`.

| Evaluation set | Rows | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Validation | 2,000 | 87.45% | 0.8762 |
| Original test | 3,080 | 87.73% | 0.8777 |
| Filtered test | 3,072 | 87.70% | 0.8774 |

Training used 7,999 examples. Three configurations were compared
on validation data, improving validation accuracy from 84.55%
to 87.45%. The selected model was frozen before test evaluation.

The filtered test subset excludes seven examples matching normalized
development text and one additional repeated test text. All 77
categories remain represented. Paraphrase overlap was not assessed.

Source-data and model hashes are recorded in the reports.
See `reports/model_comparison.json` and `reports/test_metrics.json`.

## Features

- Reproducible dataset download with SHA-256 source hashes
- Normalized-text duplicate detection before splitting
- Stratified training and validation splits
- TF-IDF and classifier saved together in one pipeline
- Baseline comparison and per-category evaluation
- Validation error examples for analysis
- Command-line predictions with the top three categories
- FastAPI prediction and health endpoints
- Human-review response when no vocabulary features are recognized
- Seven automated API tests
- GitHub Actions test workflow

## Technology

Python, scikit-learn, pandas, joblib, FastAPI, pytest,
and GitHub Actions.

## Setup

Run these commands from the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

The initial model was trained using Python 3.13.1 and
scikit-learn 1.9.1.

## Download data and train

```bash
python scripts/download_data.py
python scripts/prepare_data.py
python scripts/train.py
```

Training creates:

- `models/ticket_classifier.joblib`
- `reports/validation_metrics.json`
- `reports/validation_classification_report.json`
- `reports/validation_errors.csv`

Raw datasets and model artifacts are excluded from Git.
Run the commands above to recreate them.

## Command-line prediction

```bash
python -m src.predict "I lost my bank card. How can I block it?"
```

The response includes the predicted intent and the three
highest-scoring categories.

## Run the API

Train the model first, then run:

```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8002
```

Interactive API documentation:
http://127.0.0.1:8002/docs

Health endpoint:
http://127.0.0.1:8002/health

Example request:

```bash
curl -sS -X POST http://127.0.0.1:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I lost my bank card. How can I block it?"}'
```

There is no homepage at `/`; use `/docs` to try predictions.

## Tests

```bash
python -m pytest -q
```

The seven API tests use a small test model. They check API behavior,
input validation, and unknown-vocabulary handling without requiring
the downloaded dataset or trained model.

They do not measure the full classifier's accuracy.

## Dataset and attribution

This project uses PolyAI's BANKING77 dataset under the
Creative Commons Attribution 4.0 International license.

Reference: Casanueva et al. (2020),
"Efficient Intent Detection with Dual Sentence Encoders."

See [DATASET.md](DATASET.md) for source links, licensing,
and preprocessing details.

## Limitations

- Trained for banking queries, not general IT support.
- Model probabilities have not been calibrated.
- No automatic-routing confidence threshold has been selected.
- Unknown-word handling is not general out-of-domain detection.
- Normalized exact-text overlap is audited; paraphrase overlap
  and generalization to other datasets remain unassessed.
- Routing-team and priority labels are not present in the dataset.
- The local API has no authentication or rate limiting.
- Test results apply to BANKING77, not unseen production traffic.

## Next steps

- Analyze validation mistakes and commonly confused categories.
- Compare additional model configurations.
- Evaluate confidence-based human-review thresholds.
- Evaluate on a separate dataset before claiming broader generalization.
- Add Docker packaging and deployment monitoring.
