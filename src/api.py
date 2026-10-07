"""HTTP API for the trained support-ticket classifier."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.predict import load_model, predict_ticket


@asynccontextmanager
async def lifespan(app):
    # Load once at startup, rather than for every request.
    app.state.model = load_model()
    yield
    del app.state.model


app = FastAPI(
    title="Support Ticket Classifier",
    description="Classifies banking-support messages into intent categories.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_loaded": hasattr(request.app.state, "model"),
    }


@app.post("/predict")
def predict(body: PredictionRequest, request: Request):
    try:
        return predict_ticket(request.app.state.model, body.text)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
