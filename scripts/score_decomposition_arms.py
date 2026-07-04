"""Score the decomposition A/B experiment: Brier per arm on resolved binary questions.

Reads experiment_logs/*/forecast_reports.jsonl records (which carry a
"decomposition_arm" field when DECOMPOSITION_MODE is on/random), fetches
resolutions from Metaculus for the questions that have resolved, and compares
mean Brier score between the "on" and "off" arms.

Logs from GitHub Actions runs live in workflow artifacts; download them first,
e.g.:  gh run download <run-id> --dir experiment_logs_artifacts
then:  python scripts/score_decomposition_arms.py --log-dir experiment_logs_artifacts

Notes:
- Binary questions only (Brier). MC/numeric records are counted but not scored.
- "on_failed" records (arm was on but decomposition errored) are reported
  separately; a high on_failed rate biases the comparison and should be fixed
  before trusting results.
- Expect this to be underpowered for a ~6% relative Brier effect until each arm
  has hundreds of resolved questions. Treat early readouts as directional only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
dotenv.load_dotenv(REPO_ROOT / ".env")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from forecasting_tools import MetaculusClient  # noqa: E402


def _load_latest_records(log_dir: Path) -> dict[tuple[int, str], dict]:
    """Latest record per question, keyed by (post_id, question_text).

    Run directories are named with a UTC timestamp prefix, so sorted file order
    is chronological and later records overwrite earlier ones (the bot may
    reforecast a question; the latest forecast is the one that counts).
    """
    latest: dict[tuple[int, str], dict] = {}
    for jsonl_path in sorted(log_dir.glob("*/forecast_reports.jsonl")):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            report = record.get("report", {})
            if report.get("status") != "ok":
                continue
            arm = report.get("decomposition_arm")
            if arm not in ("on", "off", "on_failed"):
                continue
            question = report.get("question", {})
            post_id = question.get("id_of_post")
            question_text = question.get("question_text") or ""
            if not post_id:
                continue
            latest[(post_id, question_text)] = {
                "arm": arm,
                "report_type": report.get("report_type"),
                "prediction": report.get("prediction"),
                "question_text": question_text,
                "post_id": post_id,
                "run_id": record.get("run_id"),
            }
    return latest


def _resolve_outcomes(
    records: list[dict], client: MetaculusClient
) -> tuple[list[dict], int]:
    """Attach outcome (1.0/0.0) to records whose question resolved YES/NO."""
    resolved: list[dict] = []
    pending = 0
    questions_by_post: dict[int, list] = {}
    for record in records:
        post_id = record["post_id"]
        if post_id not in questions_by_post:
            try:
                try:
                    result = client.get_question_by_post_id(post_id)
                except ValueError as error:
                    if "got 0" not in str(error):
                        raise
                    result = client.get_question_by_post_id(
                        post_id, group_question_mode="unpack_subquestions"
                    )
                questions_by_post[post_id] = (
                    result if isinstance(result, list) else [result]
                )
            except Exception as error:
                print(f"  warning: could not fetch post {post_id}: {error!r}")
                questions_by_post[post_id] = []
            time.sleep(0.3)
        candidates = questions_by_post[post_id]
        question = None
        if len(candidates) == 1:
            question = candidates[0]
        else:
            for candidate in candidates:
                if candidate.question_text == record["question_text"]:
                    question = candidate
                    break
        if question is None:
            continue
        state = getattr(question, "state", None)
        state_value = getattr(state, "value", state)
        if state_value != "resolved":
            pending += 1
            continue
        resolution = (question.resolution_string or "").strip().lower()
        if resolution not in ("yes", "no"):
            continue  # annulled/ambiguous: unscoreable
        record["outcome"] = 1.0 if resolution == "yes" else 0.0
        resolved.append(record)
    return resolved, pending


def _mean_brier(records: list[dict]) -> float:
    return sum((r["prediction"] - r["outcome"]) ** 2 for r in records) / len(records)


def _bootstrap_diff_ci(
    on_records: list[dict], off_records: list[dict], iterations: int = 4000
) -> tuple[float, float]:
    rng = random.Random(42)
    diffs = []
    for _ in range(iterations):
        on_sample = rng.choices(on_records, k=len(on_records))
        off_sample = rng.choices(off_records, k=len(off_records))
        diffs.append(_mean_brier(on_sample) - _mean_brier(off_sample))
    diffs.sort()
    return diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=str, default="experiment_logs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    latest = _load_latest_records(log_dir)
    all_records = list(latest.values())
    if not all_records:
        print("No records with a decomposition_arm field found.")
        print("(Arms are only logged when DECOMPOSITION_MODE is 'on' or 'random'.)")
        sys.exit(0)

    on_failed = [r for r in all_records if r["arm"] == "on_failed"]
    binary = [
        r
        for r in all_records
        if r["arm"] in ("on", "off")
        and r["report_type"] == "BinaryReport"
        and isinstance(r["prediction"], (int, float))
    ]
    non_binary = [
        r
        for r in all_records
        if r["arm"] in ("on", "off") and r["report_type"] != "BinaryReport"
    ]

    print(f"Questions with an arm assignment: {len(all_records)}")
    print(f"  binary (scoreable):     {len(binary)}")
    print(f"  non-binary (unscored):  {len(non_binary)}")
    print(f"  on_failed (excluded):   {len(on_failed)}")
    if on_failed:
        rate = len(on_failed) / max(
            1, len(on_failed) + len([r for r in binary if r["arm"] == "on"])
        )
        print(f"  WARNING: decomposition failure rate in 'on' arm ~{rate:.0%}; "
              "fix failures before trusting the comparison.")

    print("\nFetching resolutions from Metaculus...")
    client = MetaculusClient()
    resolved, pending = _resolve_outcomes(binary, client)
    on_arm = [r for r in resolved if r["arm"] == "on"]
    off_arm = [r for r in resolved if r["arm"] == "off"]

    print(f"\nResolved YES/NO binary questions: {len(resolved)} "
          f"(pending/unresolved: {pending})")
    for label, arm_records in (("decomposition ON", on_arm), ("decomposition OFF", off_arm)):
        if arm_records:
            print(f"  {label}: n={len(arm_records)}, "
                  f"mean Brier={_mean_brier(arm_records):.4f}")
        else:
            print(f"  {label}: n=0")

    if len(on_arm) >= 5 and len(off_arm) >= 5:
        low, high = _bootstrap_diff_ci(on_arm, off_arm)
        diff = _mean_brier(on_arm) - _mean_brier(off_arm)
        print(f"\nBrier difference (ON - OFF): {diff:+.4f}  "
              f"[95% bootstrap CI: {low:+.4f}, {high:+.4f}]")
        print("Negative = decomposition arm scored better (lower Brier).")
        if low <= 0 <= high:
            print("CI includes zero: no significant difference yet - keep collecting.")
    else:
        print("\nToo few resolved questions per arm for a comparison (need >= 5 each).")


if __name__ == "__main__":
    main()
