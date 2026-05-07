"""Fail fast when an OpenRouter API key has no spend left."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def _annotation(level: str, title: str, message: str) -> None:
    escaped_message = message.replace("\r", " ").replace("\n", " ")
    print(f"::{level} title={title}::{escaped_message}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_key_info(api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        OPENROUTER_KEY_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def main() -> int:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        _annotation(
            "error",
            "OpenRouter key missing",
            "OPENROUTER_API_KEY is required for the configured forecasting models.",
        )
        return 1

    min_remaining = _as_float(
        os.getenv("OPENROUTER_MIN_CREDITS_REMAINING", "0.05")
    )
    if min_remaining is None:
        min_remaining = 0.05

    try:
        key_info = _fetch_key_info(api_key)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:600]
        _annotation(
            "error",
            "OpenRouter key check failed",
            f"OpenRouter returned HTTP {error.code} from {OPENROUTER_KEY_URL}: {detail}",
        )
        return 1
    except Exception as error:
        _annotation(
            "error",
            "OpenRouter key check failed",
            f"Could not check OpenRouter key budget: {error!r}",
        )
        return 1

    data = key_info.get("data", {})
    limit_remaining = _as_float(data.get("limit_remaining"))
    limit = _as_float(data.get("limit"))
    usage_monthly = _as_float(data.get("usage_monthly"))
    reset = data.get("limit_reset") or "not reported"

    if limit_remaining is None:
        print("OpenRouter key budget OK: no configured credit limit reported.")
        return 0

    if limit_remaining <= min_remaining:
        _annotation(
            "error",
            "OpenRouter key limit exhausted",
            "OpenRouter reports "
            f"${limit_remaining:.4f} remaining, at/below the workflow threshold "
            f"of ${min_remaining:.4f}. Limit reset: {reset}. "
            "Increase the key limit/add credits or reduce model spend before forecasting.",
        )
        return 1

    limit_text = "unlimited" if limit is None else f"${limit:.4f}"
    usage_text = "unknown" if usage_monthly is None else f"${usage_monthly:.4f}"
    print(
        "OpenRouter key budget OK: "
        f"${limit_remaining:.4f} remaining, limit {limit_text}, "
        f"monthly usage {usage_text}, reset {reset}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
