"""Alignment backends: hashed mock (default) and optional HuggingFace CLIP/BLIP."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

from PIL import Image


def _fingerprint(image: Image.Image, text: str) -> str:
    payload = image.convert("RGB").resize((16, 16)).tobytes() + text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MockAlignmentBackend:
    """Deterministic cosine-similarity stand-in. No downloads, no GPU.

    Scores are derived from a SHA-256 digest unless an explicit mapping is
    provided. Optional ``boost_terms`` raise the score when those substrings
    appear in the caption, which keeps demo/radar charts readable without
    pretending to run CLIP.
    """

    def __init__(
        self,
        scores: Mapping[str, float] | None = None,
        *,
        boost_terms: tuple[str, ...] = (),
        default: float | None = None,
    ) -> None:
        self._scores = dict(scores or {})
        self._boost_terms = tuple(term.lower() for term in boost_terms)
        self._default = default

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        key = _fingerprint(image, text)
        if key in self._scores:
            return float(self._scores[key])
        if text in self._scores:
            return float(self._scores[text])
        if self._default is not None:
            base = float(self._default)
        else:
            digest = int(_fingerprint(image, text)[:8], 16)
            base = (digest / 0xFFFFFFFF) * 0.4 + 0.3  # stay in ~[0.3, 0.7]
        lowered = text.lower()
        hits = sum(1 for term in self._boost_terms if term in lowered)
        if self._boost_terms:
            base = min(1.0, base + 0.15 * hits / max(len(self._boost_terms), 1))
        return float(base)


class ConstantAlignmentBackend:
    """Fixed similarity, used to unit-test metric formulas in isolation."""

    def __init__(self, similarity: float) -> None:
        self.similarity = similarity

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        return self.similarity


class HuggingFaceCLIPBackend:
    """Real CLIP embeddings via ``transformers``. Requires the ``hf`` extra."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP HuggingFace backend requires the 'hf' extra: pip install -e '.[hf]'"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id).to(device).eval()
        self._device = device
        self._torch = torch

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        self._load()
        torch = self._torch
        inputs = self._processor(text=[text], images=image.convert("RGB"), return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            similarity = (image_embeds * text_embeds).sum(dim=-1)
        return float(similarity[0].item())


class HuggingFaceBLIPBackend:
    """BLIP image-text matching via ``transformers``. Requires the ``hf`` extra."""

    def __init__(self, model_id: str = "Salesforce/blip-itm-base-coco", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import BlipForImageTextRetrieval, BlipProcessor
        except ImportError as exc:
            raise ImportError(
                "BLIP HuggingFace backend requires the 'hf' extra: pip install -e '.[hf]'"
            ) from exc
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = BlipProcessor.from_pretrained(self.model_id)
        self._model = BlipForImageTextRetrieval.from_pretrained(self.model_id).to(device).eval()
        self._device = device
        self._torch = torch

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        self._load()
        torch = self._torch
        inputs = self._processor(images=image.convert("RGB"), text=text, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs, use_itm_head=True)
            # ITM head: logits over {no-match, match}; take match probability.
            logits = outputs.itm_score
            prob = torch.softmax(logits, dim=-1)[0, 1]
        return float(prob.item())


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
