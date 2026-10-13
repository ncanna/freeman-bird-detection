"""Offline checks for the question bank, answer parsing, and mock eval. No API, no weights."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from system_one_detector.datasets.frames import FrameExample
from system_one_detector.datasets.questions import (
    build_questions,
    cell_of_box,
    count_level,
    labels_for_frame,
)
from system_one_detector.engine import evaluate_examples
from system_one_detector.models.jev_omni import omni_call, omni_to_answer
from system_one_detector.models.mock import MockBackend
from system_one_detector.models.registry import get_backend
from system_one_detector.util.metrics import aggregate, score_pair
from system_one_detector.util.schema import Answer, parse_systemone_response, validate_questions


def test_labels() -> None:
    assert count_level(0) == 0
    assert count_level(1) == 1
    assert count_level(3) == 2
    assert count_level(4) == 3
    assert cell_of_box([0, 0, 10, 10], 300, 300) == "tl"
    assert cell_of_box([100, 100, 20, 20], 300, 300) == "mc"
    assert cell_of_box([250, 250, 10, 10], 300, 300) == "br"
    labels = labels_for_frame(
        ["bird_present", "bird_count", "largest_bird_cell", "primary_category"],
        [{"bbox": [250, 250, 10, 10], "category_id": 3}],
        width=300,
        height=300,
        category_names={3: "cardinal"},
    )
    assert labels["bird_present"]["noul"] == 1.0
    assert labels["bird_count"]["level"] == 1
    assert labels["largest_bird_cell"]["choice"] == "br"
    assert labels["primary_category"]["choice"] == "cardinal"
    empty = labels_for_frame(
        ["bird_present", "largest_bird_cell"],
        [],
        width=100,
        height=100,
    )
    assert empty["bird_present"]["noul"] == 0.0
    assert empty["largest_bird_cell"]["choice"] == "none"
    missing_size = labels_for_frame(["largest_bird_cell"], [{"bbox": [1, 1, 2, 2]}], width=None, height=None)
    assert "largest_bird_cell" not in missing_size


def test_questions_and_jev_shape() -> None:
    questions = build_questions()
    validate_questions(questions)
    assert set(questions) == {
        "bird_present",
        "bird_count",
        "largest_bird_cell",
        "bird_behavior",
    }
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "bird_present": {"type": "noul", "noul": 0.95},
            "largest_bird_cell": {
                "type": "choice",
                "choice": "mc",
                "probabilities": {"mc": 0.8, "none": 0.2},
                "confidence": 0.7,
            },
        },
        "usage": {"input_tokens": 10, "output_tokens": 4},
    }
    parsed = parse_systemone_response(payload)
    assert parsed.answers["bird_present"].noul == 0.95
    assert parsed.answers["largest_bird_cell"].choice == "mc"


def test_jev_omni_translation() -> None:
    questions = build_questions()
    text, options = omni_call(questions["bird_present"])
    assert options == ["No", "Yes"]
    assert "bird" in text.lower()
    answer = omni_to_answer(
        questions["bird_present"],
        {"prediction": "Yes", "probabilities": {"No": 0.2, "Yes": 0.8}, "confidence": 0.6},
    )
    assert answer.type == "noul"
    assert abs(answer.noul - 0.8) < 1e-6
    _text, count_options = omni_call(questions["bird_count"])
    score = omni_to_answer(
        questions["bird_count"],
        {"prediction": count_options[1], "probabilities": [0.1, 0.7, 0.1, 0.1], "confidence": 0.5},
    )
    assert score.type == "score"
    assert abs(score.score - (0 * 0.1 + 1 * 0.7 + 2 * 0.1 + 3 * 0.1)) < 1e-6
    assert score.probabilities["1"] == 0.7


def test_clef_envelope_and_djev_receipt() -> None:
    from system_one_detector.models.http import unwrap_result

    # Cloudflare wraps the System One body in {"result": {...}, "success": true}.
    wrapped = {
        "result": {
            "model": "clef",
            "answers": {"bird_present": {"type": "noul", "noul": 0.75}},
            "usage": {"input_tokens": 5},
        },
        "success": True,
        "errors": [],
    }
    parsed = parse_systemone_response(unwrap_result(wrapped))
    assert parsed.answers["bird_present"].noul == 0.75

    # Djev 202 receipt: a receipt is not an answer. The terminal envelope nests
    # the body under "response" and only succeeded means inference worked.
    receipt = {"id": "abc", "status": "queued", "request_sha256": "deadbeef", "retry_after": 3}
    assert "answers" not in receipt
    envelope = {
        "id": "abc",
        "status": "succeeded",
        "request_sha256": "deadbeef",
        "response": {
            "model": "djev-0.1",
            "answers": {"bird_present": {"type": "noul", "noul": 0.9}},
        },
    }
    assert envelope["status"] == "succeeded"
    parsed = parse_systemone_response(unwrap_result(envelope))
    assert parsed.answers["bird_present"].noul == 0.9


def test_media_guards() -> None:
    from system_one_detector.models.media import clef_images, djev_images

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "frame.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        uri = clef_images([str(png)])[0]
        assert uri.startswith("data:image/png;base64,")
        assert djev_images(str(png))[0].startswith("data:image/png;base64,")
        mov = Path(tmp) / "clip.mov"
        mov.write_bytes(b"ftyp")
        for fn, arg in ((clef_images, [str(mov)]), (djev_images, str(mov))):
            try:
                fn(arg)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} should reject .mov")


def test_input_guards_precede_credentials() -> None:
    """A schema error should surface even when no API key is set."""
    from system_one_detector.models.clef import ClefBackend
    from system_one_detector.models.djev import DjevBackend
    from system_one_detector.models.jevimage import JevImageBackend

    for env in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_AUTH_TOKEN", "DJEV_API_KEY", "JEVIMAGE_API_KEY"):
        os.environ.pop(env, None)
    questions = build_questions(["bird_present"])
    guards = [
        (ClefBackend(), "video", "images only"),
        (ClefBackend(), "audio", "images only"),
        (DjevBackend(), "video", "takes images"),
        (DjevBackend(), "audio", "takes images"),
        (JevImageBackend(), "audio", "image-only"),
    ]
    for backend, modality, fragment in guards:
        try:
            backend.decide({}, questions, media_path="frame.png", modality=modality)
        except ValueError as exc:
            assert fragment in str(exc), f"{backend.name}/{modality}: {exc}"
        else:
            raise AssertionError(f"{backend.name} should reject modality={modality!r}")
    # Missing media is a ValueError even without a key.
    try:
        ClefBackend().decide({}, questions, modality="image")
    except ValueError:
        pass
    else:
        raise AssertionError("clef should require media_path for image modality")
    # Djev caps questions per request.
    try:
        DjevBackend().decide({}, {f"q{i}": {"type": "noul", "instructions": "a"} for i in range(33)})
    except ValueError:
        pass
    else:
        raise AssertionError("djev should reject more than 32 questions")


def test_clef_and_openai_payloads() -> None:
    from system_one_detector.models.clef import ClefBackend
    from system_one_detector.models.openai_decisions import OpenAiDecisionsBackend

    try:
        ClefBackend(model="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("clef should reject an unknown model")

    questions = build_questions(["bird_present"])
    clef = ClefBackend(model="clef-flash")
    assert clef.sees_pixels is True

    openai = OpenAiDecisionsBackend()
    payload = openai._build_payload({"frame_name": "a.png"}, questions)
    assert "context" in payload and "questions" in payload

    # A bare selection is the shape public reporting describes.
    bare = {"choice": "mc", "confidence": 0.7}
    answers = openai._read_answers(bare, build_questions(["largest_bird_cell"]))
    assert answers["largest_bird_cell"].choice == "mc"
    assert abs(sum(answers["largest_bird_cell"].probabilities.values()) - 1.0) < 1e-9


def test_omni_jev_bias_reading() -> None:
    from system_one_detector.models.omni_jev import read_noul_bias

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "release_manifest.json"
        manifest.write_text(json.dumps({"biases": {"noul": 0.0531085661}}))
        bias = read_noul_bias(tmp, None)
        assert bias is not None and abs(bias - 0.0531085661) < 1e-9
    assert read_noul_bias(None, 0.25) == 0.25


def test_registry_and_config() -> None:
    from system_one_detector.models import list_backends
    from system_one_detector.util.config import KNOWN_BACKENDS

    registered = list_backends()
    assert registered == sorted(KNOWN_BACKENDS), (registered, sorted(KNOWN_BACKENDS))
    text_only = {"jev", "laya", "mock", "systemone_http"}
    for name in registered:
        cls = get_backend(name)
        assert isinstance(cls.sees_pixels, bool)
        if name not in text_only:
            assert cls.sees_pixels is True, f"{name} should see pixels"


def test_metrics_and_mock_engine() -> None:
    questions = build_questions(["bird_present"])
    hit = score_pair(Answer(type="noul", noul=0.9), {"type": "noul", "noul": 1.0})
    assert hit["correct"] == 1.0
    example = FrameExample(
        frame_id="IMG_0001_frame_000000",
        file_name="IMG_0001_frame_000000.png",
        image_path=None,
        video_stem="IMG_0001",
        split="test",
        state={"frame_name": "IMG_0001_frame_000000.png"},
        questions=questions,
        labels={"bird_present": {"type": "noul", "noul": 1.0}},
        modality="image",
        media_path=None,
    )
    rows, metrics = evaluate_examples([example], MockBackend())
    assert rows[0]["error"] is None
    assert rows[0]["answers"]["bird_present"]["noul"] == 0.5
    assert metrics["n_scored"] == 1
    assert metrics["per_question"]["bird_present"]["accuracy"] == 1.0
    summary = aggregate(
        [{"answers": {"bird_present": {"type": "noul", "noul": 0.1}}, "labels": {"bird_present": {"type": "noul", "noul": 1.0}}}]
    )
    assert summary["per_question"]["bird_present"]["accuracy"] == 0.0


def main() -> None:
    test_labels()
    test_questions_and_jev_shape()
    test_jev_omni_translation()
    test_clef_envelope_and_djev_receipt()
    test_media_guards()
    test_clef_and_openai_payloads()
    test_input_guards_precede_credentials()
    test_omni_jev_bias_reading()
    test_registry_and_config()
    test_metrics_and_mock_engine()
    print("ok")


if __name__ == "__main__":
    main()
