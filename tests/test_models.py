"""Unified VLM interface: fakes, factory, and optional extra import guards."""

from unittest.mock import MagicMock, patch

import pytest

from visioneval.models.base import GenerationResult, VisionLanguageModel
from visioneval.models.factory import build_model, describe_available_backends
from visioneval.models.fake import FakeVLM
from visioneval.models.hf import HuggingFaceVLM
from visioneval.models.api import OpenAICompatibleVLM, image_to_data_url


def test_fake_vlm_satisfies_protocol_and_returns_profile_fields(red_square) -> None:
    model = FakeVLM(
        name="unit-fake",
        responses={"red_square": "A red square sits in the center."},
        object_map={"red_square": ["square", "red square"]},
        latency_ms=0.0,
    )
    assert isinstance(model, VisionLanguageModel)
    result = model.generate(red_square, "Describe the image.", sample_id="red_square")
    assert isinstance(result, GenerationResult)
    assert result.text == "A red square sits in the center."
    assert result.ttft_ms >= 0.0
    assert result.total_ms >= result.ttft_ms
    assert result.extra["backend"] == "fake"


def test_fake_vlm_answers_pope_from_object_map(red_square) -> None:
    model = FakeVLM(object_map={"red_square": ["square"]})
    yes = model.generate(red_square, "Is there a square in the image?", sample_id="red_square")
    no = model.generate(red_square, "Is there a cat in the image?", sample_id="red_square")
    assert yes.text.startswith("Yes")
    assert no.text.startswith("No")


def test_factory_builds_fake_and_rejects_unknown_kind() -> None:
    model = build_model({"kind": "fake", "name": "x", "default_response": "hi"})
    assert isinstance(model, FakeVLM)
    assert model.name == "x"
    with pytest.raises(ValueError, match="unknown model kind"):
        build_model({"kind": "not-a-backend", "name": "x"})


def test_huggingface_adapter_requires_hf_extra() -> None:
    adapter = HuggingFaceVLM(model_id="Qwen/Qwen2-VL-2B-Instruct", kind="qwen2_vl")
    with patch.dict("sys.modules", {"torch": None, "transformers": None}):
        with patch("visioneval.models.hf.HuggingFaceVLM._require_transformers", side_effect=ImportError("missing")):
            with pytest.raises(ImportError, match="missing"):
                adapter._load()


def test_huggingface_adapter_import_error_message(red_square) -> None:
    adapter = HuggingFaceVLM(model_id="llava-hf/llava-1.5-7b-hf", kind="llava")

    def _boom() -> None:
        raise ImportError(
            "HuggingFace VLM adapters require the 'hf' extra: pip install -e '.[hf]'"
        )

    adapter._require_transformers = lambda: (_boom() or (None, None))  # type: ignore[method-assign]
    with pytest.raises(ImportError, match=r"pip install -e '\.\[hf\]'"):
        adapter.generate(red_square, "hi")


def test_api_adapter_uses_streaming_client(red_square) -> None:
    adapter = OpenAICompatibleVLM(model="gpt-4o-mini", name="api-demo")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="A red "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="square."))]

    client = MagicMock()
    client.chat.completions.create.return_value = [chunk1, chunk2]
    adapter._client = lambda: client  # type: ignore[method-assign]
    result = adapter.generate(red_square, "Describe the image.")
    assert result.text == "A red square."
    assert result.extra["backend"] == "api"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["model"] == "gpt-4o-mini"


def test_image_to_data_url_is_png(red_square) -> None:
    url = image_to_data_url(red_square)
    assert url.startswith("data:image/png;base64,")


def test_describe_available_backends_always_includes_fake() -> None:
    available = describe_available_backends()
    assert available["fake"] is True
    assert set(available) >= {"fake", "hf", "api"}
