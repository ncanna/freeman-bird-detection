"""POST JSON to a System One HTTP endpoint. Stdlib only."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


def resolve_endpoint(base_url: str | None, endpoint: str | None) -> str:
    if endpoint:
        return endpoint
    if not base_url:
        raise ValueError("set base_url or endpoint")
    root = base_url.rstrip("/")
    if root.endswith("/v1/systemone"):
        return root
    return root + "/v1/systemone"


def build_headers(
    api_key: str | None,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    retries: int = 2,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """POST a System One body. Retries 429 and 529, which the TypeSafe docs call out."""
    raw, status, response_headers = post_json_raw(
        url,
        payload,
        api_key=api_key,
        timeout_s=timeout_s,
        retries=retries,
        headers=headers,
        accepted_status=(200,),
    )
    if status != 200:
        raise RuntimeError(f"HTTP {status} from {url}: {json.dumps(raw)[:500]}")
    return raw


def post_json_raw(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    retries: int = 2,
    headers: dict[str, str] | None = None,
    accepted_status: tuple[int, ...] = (200,),
) -> tuple[dict[str, Any], int, dict[str, str]]:
    """POST and return ``(body, status, headers)``. Retries 429, 502, 503, 504, 529."""
    body = json.dumps(payload).encode("utf-8")
    request_headers = build_headers(api_key, headers)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8")
                status = response.status
                response_headers = dict(response.headers.items())
            parsed = json.loads(raw) if raw.strip() else {}
            if not isinstance(parsed, dict):
                raise ValueError(f"expected a JSON object from {url}")
            return parsed, status, response_headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code in (429, 502, 503, 504, 529) and attempt < retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
                last_error = exc
                continue
            if exc.code in accepted_status:
                try:
                    parsed = json.loads(detail or "{}")
                except json.JSONDecodeError:
                    parsed = {"error": detail}
                return (parsed if isinstance(parsed, dict) else {"error": detail}), exc.code, dict(exc.headers.items())
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc
    raise RuntimeError(f"request to {url} failed after retries") from last_error


def get_json(
    url: str,
    *,
    api_key: str | None = None,
    timeout_s: float = 120.0,
    headers: dict[str, str] | None = None,
    accepted_status: tuple[int, ...] = (200,),
) -> tuple[dict[str, Any], int]:
    request = urllib.request.Request(url, headers=build_headers(api_key, headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code not in accepted_status:
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
        try:
            return json.loads(detail or "{}"), exc.code
        except json.JSONDecodeError:
            return {"error": detail}, exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc
    parsed = json.loads(raw) if raw.strip() else {}
    return (parsed if isinstance(parsed, dict) else {"error": str(parsed)}), status


def api_key_from_env(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    return value or None


def unwrap_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Unwrap provider envelopes such as Cloudflare's ``{"result": {...}}``."""
    if "answers" in payload:
        return payload
    inner = payload.get("result")
    if isinstance(inner, dict) and "answers" in inner:
        return inner
    for key in ("response", "data"):
        inner = payload.get(key)
        if isinstance(inner, dict) and "answers" in inner:
            return inner
    return payload
