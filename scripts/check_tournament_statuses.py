"""Diagnostic: report the live status of questions in a Metaculus tournament.

Answers the two questions that block the open-vs-upcoming fix decision:
  1. Are recent questions currently `open` or `upcoming`?
  2. For upcoming ones, when do they open, and is forecasting actually permitted?

Usage (PowerShell):
    $env:METACULUS_TOKEN = "<your token>"      # or put it in .env
    python scripts/check_tournament_statuses.py
    python scripts/check_tournament_statuses.py --tournament summer-futureeval-2026
    python scripts/check_tournament_statuses.py --url https://www.metaculus.com/questions/12345/...

Read-only: it never submits a forecast.
"""

from __future__ import annotations

import argparse
import os
import sys

import dotenv
import requests

dotenv.load_dotenv()

API_BASE = "https://www.metaculus.com/api"
DEFAULT_TOURNAMENT = "summer-futureeval-2026"


def _auth_headers() -> dict[str, str]:
    token = os.getenv("METACULUS_TOKEN", "").strip()
    if not token:
        sys.exit(
            "METACULUS_TOKEN is not set. Set it in the environment or .env "
            "(Account -> API token on Metaculus)."
        )
    return {"Authorization": f"Token {token}", "Accept": "application/json"}


def _get(path: str, params: dict | None = None) -> dict:
    response = requests.get(
        f"{API_BASE}{path}", headers=_auth_headers(), params=params, timeout=30
    )
    response.raise_for_status()
    return response.json()


def _describe_post(post: dict) -> str:
    question = post.get("question") or {}
    # Group/multiple-question posts nest sub-questions instead of a single one.
    sub = post.get("group_of_questions") or {}
    q_status = question.get("status") or sub.get("status") or "?"
    open_time = question.get("open_time") or post.get("open_time") or ""
    close_time = question.get("scheduled_close_time") or post.get("scheduled_close_time") or ""
    nr_forecasters = question.get("nr_forecasters")
    # `my_forecasts` is present (non-null) when the authenticated user has predicted.
    mine = question.get("my_forecasts")
    have_predicted = bool(mine and (mine.get("latest") if isinstance(mine, dict) else mine))
    forecastable = q_status == "open"
    return (
        f"  post {post.get('id')!s:>8}  post_status={post.get('status','?'):<9}"
        f"  q_status={q_status:<9}  forecastable={str(forecastable):<5}"
        f"  predicted_by_me={str(have_predicted):<5}"
        f"  opens={open_time or '-'}  closes={close_time or '-'}\n"
        f"            title: {post.get('title','')[:90]}"
    )


def check_tournament(slug: str) -> None:
    print(f"\n=== Tournament: {slug} ===")
    for status_group in (["open"], ["upcoming"], ["closed", "resolved"]):
        params = {
            "tournaments": slug,
            "limit": 50,
            "order_by": "-created_at",
        }
        # The API accepts repeated `statuses` params.
        url = f"{API_BASE}/posts/"
        resp = requests.get(
            url,
            headers=_auth_headers(),
            params=[("tournaments", slug), ("limit", 50), ("order_by", "-created_at")]
            + [("statuses", s) for s in status_group],
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        print(f"\n[{'/'.join(status_group)}]  count={data.get('count')}  shown={len(results)}")
        for post in results:
            print(_describe_post(post))


def check_url(url: str) -> None:
    # Extract the numeric post id from a question URL.
    parts = [p for p in url.split("/") if p.isdigit()]
    if not parts:
        sys.exit(f"Could not find a numeric id in URL: {url}")
    post_id = parts[0]
    post = _get(f"/posts/{post_id}/")
    print(f"\n=== {url} ===")
    print(_describe_post(post))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT)
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="Specific question URL(s) to inspect instead of the whole tournament.",
    )
    args = parser.parse_args()

    if args.url:
        for url in args.url:
            check_url(url)
    else:
        check_tournament(args.tournament)


if __name__ == "__main__":
    main()
