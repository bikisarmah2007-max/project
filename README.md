import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("reviewsense.api")

class ReviewInput(BaseModel):
    review_text: str = Field(..., min_length=1)

    @field_validator("review_text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("review_text must not be empty or whitespace")
        return v.strip()

class BatchInput(BaseModel):
    reviews: List[ReviewInput] = Field(..., min_length=1)

    @field_validator("reviews")
    @classmethod
    def no_empty_list(cls, v: List[ReviewInput]) -> List[ReviewInput]:
        if not v:
            raise ValueError("reviews list must not be empty")
        return v

class PredictionResult(BaseModel):
    sentiment: str
    aspects: List[str] = Field(default_factory=list)

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    count: int

def _stub_predict(review_text: str) -> Dict[str, Any]:
    text = (review_text or "").lower()
    positive_words = {"good", "great", "love", "excellent", "fast", "amazing", "best", "nice"}
    negative_words = {"bad", "worst", "hate", "slow", "broken", "expensive", "poor", "terrible"}

    pos = sum(1 for w in positive_words if w in text)
    neg = sum(1 for w in negative_words if w in text)

    if pos > neg:
        sentiment = "positive"
    elif neg > pos:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    aspects: List[str] = []
    if any(k in text for k in ("quality", "material", "build", "durable", "broken")):
        aspects.append("Quality")
    if any(k in text for k in ("shipping", "delivery", "arrived", "courier", "late")):
        aspects.append("Shipping")
    if any(k in text for k in ("price", "cost", "expensive", "cheap", "value", "money")):
        aspects.append("Price")

    return {"sentiment": sentiment, "aspects": aspects}

def _load_predictor():
    try:
        from backend.predictor import predict as real_predict
        return real_predict, True
    except Exception:
        pass
        
    try:
        from predictor import predict as real_predict
        return real_predict, True
    except Exception:
        pass

    return _stub_predict, False

PREDICT_FN, PREDICTOR_IS_REAL = _load_predictor()

def run_prediction(review_text: str) -> PredictionResult:
    try:
        raw = PREDICT_FN(review_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    if isinstance(raw, dict):
        sentiment = raw.get("sentiment", "neutral")
        aspects = raw.get("aspects", [])
    elif isinstance(raw, str):
        sentiment, aspects = raw, []
    elif isinstance(raw, (list, tuple)) and raw:
        sentiment, aspects = raw[0], (raw[1] if len(raw) > 1 else [])
    else:
        sentiment, aspects = "neutral", []

    sentiment = str(sentiment).lower().strip() or "neutral"

    if aspects is None:
        aspects = []
    if not isinstance(aspects, list):
        aspects = [str(aspects)]
    aspects = [str(a) for a in aspects if str(a).strip()]

    return PredictionResult(sentiment=sentiment, aspects=aspects)

app = FastAPI(title="ReviewSense API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "predictor_available": PREDICTOR_IS_REAL}

@app.post("/predict", response_model=PredictionResult)
def predict_single(payload: ReviewInput):
    return run_prediction(payload.review_text)

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchInput):
    if not payload.reviews:
        raise HTTPException(status_code=422, detail="reviews list is empty")
    results = [run_prediction(r.review_text) for r in payload.reviews]
    return BatchPredictionResponse(results=results, count=len(results))

def predict_with_fallback(reviews: List[str], api_url: Optional[str] = None) -> List[Dict[str, Any]]:
    import requests
    base = api_url or "http://127.0.0.1:8000"
    payload = {"reviews": [{"review_text": r} for r in reviews]}
    try:
        resp = requests.post(f"{base}/predict_batch", json=payload, timeout=5.0)
        resp.raise_for_status()
        return resp.json()["results"]
    except Exception:
        return [run_prediction(r).model_dump() for r in reviews]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
