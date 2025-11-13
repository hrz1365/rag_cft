from deepeval.models import GeminiModel
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
import os


def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    model = GeminiModel(model_name="gemini-2.0-flash", api_key=api_key, temperature=0)
    return model


def get_metrics(model=None):
    if model is None:
        model = get_gemini_model()

    metrics = [
        AnswerRelevancyMetric(model=model),
        FaithfulnessMetric(model=model),
        ContextualPrecisionMetric(model=model),
        ContextualRecallMetric(model=model),
    ]
    return metrics
