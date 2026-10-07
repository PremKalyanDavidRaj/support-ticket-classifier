"""Predict banking-support intents using the locally trained model."""

import argparse
import json
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ticket_classifier.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python scripts/train.py"
        )
    # Load only the trusted model you trained locally.
    return joblib.load(MODEL_PATH)


def predict_ticket(model, text):
    text = text.strip()

    if not text:
        raise ValueError("Please provide a non-empty support message.")
    if len(text) > 5000:
        raise ValueError("Message must be 5,000 characters or fewer.")

    # Avoid returning a category based only on class priors when
    # none of the input words are represented in the vocabulary.
    features = model.named_steps["tfidf"].transform([text])
    if features.nnz == 0:
        return {
            "predicted_intent": None,
            "top_predictions": [],
            "needs_human_review": True,
            "reason": "No known vocabulary features found.",
        }

    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    indices = probabilities.argsort()[::-1][:3]

    return {
        "predicted_intent": str(classes[indices[0]]),
        "top_predictions": [
            {
                "intent": str(classes[index]),
                "model_probability": round(
                    float(probabilities[index]), 4
                ),
            }
            for index in indices
        ],
        "note": (
            "Model probabilities are not calibrated guarantees. "
            "Automatic routing thresholds have not been selected."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Support message to classify")
    args = parser.parse_args()

    try:
        result = predict_ticket(load_model(), args.text)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
