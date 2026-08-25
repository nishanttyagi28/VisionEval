"""Lazy CPU-first Torchvision and ONNX Runtime classification adapters."""

from dataclasses import dataclass, field
from pathlib import Path

from visioneval.core.types import ClassificationPrediction, ClassificationSample


@dataclass
class TorchvisionAdapter:
    """Load a torchvision classifier only on its first prediction."""
    architecture: str
    labels: list[str]
    weights: str = "DEFAULT"
    device: str | None = None
    _model: object | None = field(default=None, init=False, repr=False)
    _transform: object | None = field(default=None, init=False, repr=False)

    def __call__(self, sample: ClassificationSample) -> ClassificationPrediction:
        if not sample.image_path:
            raise ValueError("TorchvisionAdapter requires sample.image_path")
        if self._model is None:
            import torch
            from torchvision import models
            resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            weights_enum = models.get_model_weights(self.architecture)
            weights = getattr(weights_enum, self.weights)
            self._model = models.get_model(self.architecture, weights=weights).eval().to(resolved_device)
            self._transform = (weights.transforms(), torch, resolved_device)
        transform, torch, resolved_device = self._transform
        from PIL import Image
        with Image.open(sample.image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0).to(resolved_device)
        with torch.inference_mode():
            probabilities = self._model(tensor).softmax(dim=1)[0]
        index = int(probabilities.argmax().item())
        return ClassificationPrediction(self.labels[index], float(probabilities[index].item()))


@dataclass
class ONNXRuntimeAdapter:
    """Load an ONNX classifier only on its first prediction."""
    model_path: str
    labels: list[str]
    input_size: int = 224
    _session: object | None = field(default=None, init=False, repr=False)

    def __call__(self, sample: ClassificationSample) -> ClassificationPrediction:
        if not sample.image_path:
            raise ValueError("ONNXRuntimeAdapter requires sample.image_path")
        import numpy as np
        from PIL import Image
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        with Image.open(sample.image_path) as image:
            pixels = np.asarray(image.convert("RGB").resize((self.input_size, self.input_size)), dtype=np.float32) / 255.0
        tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
        logits = self._session.run(None, {self._session.get_inputs()[0].name: tensor})[0][0]
        probabilities = np.exp(logits - np.max(logits)); probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        return ClassificationPrediction(self.labels[index], float(probabilities[index]))