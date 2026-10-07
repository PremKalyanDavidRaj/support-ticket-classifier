# Support Ticket Classifier

A machine learning API that classifies banking customer-support
messages into 77 intent categories using TF-IDF and logistic regression.

## Results

Measured on a 2,000-example validation split:

| Metric | Majority-class baseline | TF-IDF + Logistic Regression |
| --- | ---: | ---: |
| Accuracy | 1.90% | 84.55% |
| Macro-F1 | 0.0005 | 0.8426 |

The model was trained on 7,999 examples after removing four duplicate
training texts and creating a stratified training/validation split.

These are validation results, not final test results.
The original test split has not been evaluated.

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
- Training/validation normalized-text overlap is checked;
  training/test overlap has not yet been audited.
- Routing-team and priority labels are not present in the dataset.
- The local API has no authentication or rate limiting.
- Final held-out test evaluation remains pending.

## Next steps

- Analyze validation mistakes and commonly confused categories.
- Compare additional model configurations.
- Evaluate confidence-based human-review thresholds.
- Audit test overlap and perform final held-out evaluation.
- Add Docker packaging and deployment monitoring.
