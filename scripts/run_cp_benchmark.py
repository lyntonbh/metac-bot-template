"""Score the bot against the Metaculus Community Prediction for fast, no-wait feedback.

STATUS (2026-07-04): currently non-functional. Community-prediction data appears
unreachable via the bot API token, for ANY question -- including "Human Extinction by
2100" (1,712 forecasters, 9,008 forecasts, revealed since 2017). A full sequential
sweep of all 1,252 currently-open binary questions found zero with a retrievable
community_prediction_exists=True. This isn't a filter/sampling bug: this script uses
the exact ApiFilter combination from Metaculus's own official bot-template reference
example (community_benchmark.py, March 2025), so the query mechanics match the
documented approach. The leading hypothesis is that Metaculus now withholds Community
Prediction data from bot/API tokens specifically -- consistent with their documented
policy of hiding Pro forecaster data from bots in AI tournaments to prevent
crowd-copying instead of independent reasoning -- rather than a client-side bug.
Left in the repo as-is (not actively being pursued); see memory entry
cp-benchmark-blocked-by-bot-token for the investigation.

Unlike a live tournament forecast, this never publishes anything and doesn't wait for
question resolution — it compares the bot's forecast to the current Community
Prediction on a batch of open questions that already have an established crowd
forecast, so you get a signal the same day you make a harness/prompt change. It
measures agreement-with-the-crowd, not truth, so treat divergence as "investigate
why," not "you're wrong." (This premise depends on CP data being readable at all --
see STATUS above.)

Sourcing mechanics (for whenever CP access is resolved): forecasting_tools' built-in
Benchmarker/get_benchmark_questions() samples only 3 *random* pages of open questions
before locally filtering for community_prediction_exists=True, which undercounts a
sparse survivor pool. This script instead walks every page sequentially so a real
(non-zero) survivor pool would still be found in full.

Usage:
    poetry run python scripts/run_cp_benchmark.py --num-questions 50
"""

from __future__ import annotations

import argparse
import asyncio

import dotenv

dotenv.load_dotenv()

from forecasting_tools import ApiFilter, MetaculusClient
from forecasting_tools.cp_benchmarking.benchmarker import Benchmarker

from main import SpringTemplateBot2026, TournamentPublicCommentMetaculusClient


def _build_benchmark_bot() -> SpringTemplateBot2026:
    SpringTemplateBot2026.apply_runtime_config_from_env()
    bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=True,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )
    bot.metaculus_client = TournamentPublicCommentMetaculusClient()
    return bot


async def _find_cp_benchmark_questions(num_questions: int, num_forecasters_gte: int):
    """Sequentially sweep ALL open binary questions with a real Community
    Prediction, instead of forecasting_tools' default random-3-page sample
    (which finds ~nothing against this sparse a pool -- see module docstring)."""
    client = MetaculusClient()
    api_filter = ApiFilter(
        allowed_statuses=["open"],
        allowed_types=["binary"],
        community_prediction_exists=True,
        includes_bots_in_aggregates=False,
        num_forecasters_gte=num_forecasters_gte,
        # No open_time_gt bound: a long-running still-open question with an
        # established crowd is exactly what we want, regardless of age.
    )
    questions = await client.get_questions_matching_filter(
        api_filter,
        num_questions=num_questions,
        randomly_sample=False,  # sequential sweep of every page, not a random sample
        error_if_question_target_missed=False,  # OK to return fewer than requested
    )
    return questions


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-questions",
        type=int,
        default=50,
        help="Max questions to use. The forecasting-tools Benchmarker docstring "
        "suggests 100-200 for a decent read and 500+ as ideal; this is a ceiling, "
        "not a guarantee -- see module docstring on current pool sparsity.",
    )
    parser.add_argument(
        "--num-forecasters-gte",
        type=int,
        default=1,
        help="Minimum forecaster count for a question to count as having an "
        "established Community Prediction. Empirically NOT the bottleneck "
        "(community_prediction_exists is), so the default is deliberately low.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cp_benchmarks",
        help="Directory to write the benchmark JSONL report to.",
    )
    args = parser.parse_args()

    print(f"Sweeping the open-question pool for up to {args.num_questions} "
          "questions with an established Community Prediction (this walks every "
          "matching page, so it may take a minute)...")
    questions = await _find_cp_benchmark_questions(
        args.num_questions, args.num_forecasters_gte
    )
    print(f"Found {len(questions)} usable question(s).")
    if not questions:
        print("No questions found with a Community Prediction right now -- this "
              "reflects thin engagement on the current open-question pool, not a "
              "tool failure. Try again later, or lower --num-forecasters-gte "
              "further (already at the floor of 1 by default).")
        return

    bot = _build_benchmark_bot()
    benchmarker = Benchmarker(
        forecast_bots=[bot],
        questions_to_use=questions,
        file_path_to_save_reports=args.output_dir,
        concurrent_question_batch_size=10,
    )
    benchmarks = await benchmarker.run_benchmark()

    for benchmark in benchmarks:
        print(f"\n=== {benchmark.name} ===")
        print(f"Questions attempted: {benchmark.num_input_questions}")
        print(f"Failed forecasts: {benchmark.num_failed_forecasts}")
        if benchmark.forecast_reports:
            print(
                "Average expected baseline score (vs. Community Prediction): "
                f"{benchmark.average_expected_baseline_score:.3f}"
            )
        else:
            print("No successful forecasts to score.")
        if benchmark.total_cost is not None:
            print(f"Total cost: ${benchmark.total_cost:.4f}")
        if benchmark.time_taken_in_minutes is not None:
            print(f"Time taken: {benchmark.time_taken_in_minutes:.1f} min")
        if benchmark.failed_report_errors:
            print(f"First failure: {benchmark.failed_report_errors[0][:300]}")


if __name__ == "__main__":
    asyncio.run(main())
