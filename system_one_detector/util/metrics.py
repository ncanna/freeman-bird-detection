"""Score typed answers against COCO-derived labels.

Metrics follow the typed-decision reports used for these models: argmax
accuracy, Brier score, a simple equal-width ECE, and MAE for score questions.
"""

from __future__ import annotations

from typing import Any

from system_one_detector.util.schema import Answer


def _argmax(probabilities: dict[str, float]) -> str | None:
    if not probabilities:
        return None
    return max(probabilities, key=probabilities.get)


def score_pair(
    answer: Answer,
    label: dict[str, Any],
    *,
    noul_threshold: float = 0.5,
) -> dict[str, float]:
    """Score one answer against one gold label. Keys depend on question type."""
    qtype = label["type"]
    if answer.type != qtype:
        raise ValueError(f"answer type {answer.type!r} does not match label type {qtype!r}")
    if qtype == "noul":
        gold = float(label["noul"])
        pred = float(answer.noul if answer.noul is not None else 0.0)
        return {
            "correct": float((pred >= noul_threshold) == (gold >= noul_threshold)),
            "brier": (pred - gold) ** 2,
            "confidence": pred if gold >= noul_threshold else 1.0 - pred,
        }
    if qtype == "choice":
        gold = str(label["choice"])
        pred = answer.choice
        probs = answer.probabilities or {}
        if pred is None:
            pred = _argmax(probs)
        brier = 0.0
        keys = set(probs) | {gold}
        for key in keys:
            target = 1.0 if key == gold else 0.0
            brier += (float(probs.get(key, 0.0)) - target) ** 2
        return {
            "correct": float(pred == gold),
            "brier": brier,
            "confidence": float(answer.confidence if answer.confidence is not None else probs.get(pred or "", 0.0)),
        }
    if qtype == "score":
        gold_level = int(label["level"])
        probs = {str(k): float(v) for k, v in (answer.probabilities or {}).items()}
        pred_level = _argmax(probs)
        predicted = int(pred_level) if pred_level is not None and pred_level.lstrip("-").isdigit() else None
        if predicted is None and answer.score is not None:
            predicted = int(round(answer.score))
        score = float(answer.score if answer.score is not None else gold_level)
        return {
            "correct": float(predicted == gold_level),
            "mae": abs(score - gold_level),
            "confidence": float(answer.confidence if answer.confidence is not None else 0.0),
        }
    raise ValueError(f"unsupported label type {qtype!r}")


def _ece(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    """Expected calibration error. Each pair is (confidence, correct)."""
    if not pairs:
        return None
    totals = [0.0] * bins
    correct = [0.0] * bins
    conf = [0.0] * bins
    for confidence, hit in pairs:
        index = min(bins - 1, max(0, int(confidence * bins)))
        if confidence == 1.0:
            index = bins - 1
        totals[index] += 1
        correct[index] += hit
        conf[index] += confidence
    n = float(len(pairs))
    error = 0.0
    for index in range(bins):
        if totals[index] == 0:
            continue
        accuracy = correct[index] / totals[index]
        mean_conf = conf[index] / totals[index]
        error += (totals[index] / n) * abs(accuracy - mean_conf)
    return error


def aggregate(
    rows: list[dict[str, Any]],
    *,
    noul_threshold: float = 0.5,
) -> dict[str, Any]:
    """Aggregate per-frame decision rows into per-question metrics.

    Each row has ``answers`` and ``labels`` maps. Rows with ``error`` set are
    counted but not scored.
    """
    per_question: dict[str, dict[str, Any]] = {}
    scored = 0
    errors = 0
    for row in rows:
        if row.get("error"):
            errors += 1
            continue
        scored += 1
        answers = row.get("answers") or {}
        labels = row.get("labels") or {}
        for qid, label in labels.items():
            raw_answer = answers.get(qid)
            if raw_answer is None:
                continue
            answer = raw_answer if isinstance(raw_answer, Answer) else Answer(**{
                key: raw_answer[key]
                for key in ("type", "noul", "choice", "score", "probabilities", "confidence", "legend")
                if key in raw_answer
            })
            try:
                stats = score_pair(answer, label, noul_threshold=noul_threshold)
            except (ValueError, KeyError, TypeError) as exc:
                bucket = per_question.setdefault(qid, {"type": label.get("type"), "skipped": 0, "pairs": []})
                bucket["skipped"] = bucket.get("skipped", 0) + 1
                bucket.setdefault("skip_reason", str(exc))
                continue
            bucket = per_question.setdefault(qid, {"type": label["type"], "pairs": [], "skipped": 0})
            bucket["pairs"].append(stats)

    summary: dict[str, Any] = {}
    for qid, bucket in per_question.items():
        pairs = bucket["pairs"]
        n = len(pairs)
        entry: dict[str, Any] = {
            "type": bucket["type"],
            "n": n,
            "skipped": bucket.get("skipped", 0),
        }
        if n:
            entry["accuracy"] = sum(item["correct"] for item in pairs) / n
            if "brier" in pairs[0]:
                entry["brier"] = sum(item["brier"] for item in pairs) / n
            if "mae" in pairs[0]:
                entry["mae"] = sum(item["mae"] for item in pairs) / n
            ece = _ece([(item["confidence"], item["correct"]) for item in pairs])
            if ece is not None:
                entry["ece"] = ece
        if bucket.get("skip_reason"):
            entry["skip_reason"] = bucket["skip_reason"]
        summary[qid] = entry

    return {
        "n_frames": len(rows),
        "n_scored": scored,
        "n_errors": errors,
        "noul_threshold": noul_threshold,
        "per_question": summary,
    }
