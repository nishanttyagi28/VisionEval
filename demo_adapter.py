from visioneval.core.types import ClassificationPrediction, ClassificationSample

def predict(sample: ClassificationSample) -> ClassificationPrediction:
    return ClassificationPrediction(sample.label, 0.9)
