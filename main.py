import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import dotenv
from typing import Any, Literal, TypeVar

dotenv.load_dotenv()

import requests

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceReport,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericReport,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class EvidenceItem:
    source: str
    provider: str
    title: str
    url: str
    retrieved_at: str
    summary: str
    value: str = ""
    unit: str = ""
    date: str = ""
    probability: float | None = None
    directness: str = "weak"
    caveats: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class ForecasterRoleSpec:
    name: str
    llm: GeneralLlm
    instructions: str


@dataclass(frozen=True)
class ExperimentVariant:
    id: int
    name: str
    description: str
    env_overrides: dict[str, str]


@dataclass
class RefreshScoutDecision:
    question_id: int | None
    post_id: int | None
    page_url: str | None
    question_title: str
    question_type: str
    previous_forecast: str
    previous_forecast_timestamp: str | None
    should_reforecast: bool
    recommended_action: str
    confidence_in_movement: str
    movement_reason: str
    fresh_evidence_summary: str
    estimated_new_forecast: float | None = None
    estimated_new_probabilities: dict[str, float] | None = None
    material_distribution_shift: bool = False
    fresh_high_signal_evidence: bool = False
    prior_reasoning_stale: bool = False
    raw_response: str = ""
    parse_error: str | None = None
    gate_triggered: bool = False
    gate_reasons: list[str] = field(default_factory=list)


EXPERIMENT_VARIANTS: tuple[ExperimentVariant, ...] = (
    ExperimentVariant(
        id=0,
        name="role_ensemble_control",
        description="Current role-specialized cheap ensemble with default research and escalation.",
        env_overrides={},
    ),
    ExperimentVariant(
        id=1,
        name="market_data_heavy",
        description="Lean harder on direct market/official data and require larger cheap-model disagreement before escalation.",
        env_overrides={
            "ENABLE_DIRECT_STRUCTURED_RESEARCH": "true",
            "ENABLE_DIRECT_MARKET_APIS": "true",
            "ENABLE_MARKET_PRIOR_RESEARCH": "true",
            "ENABLE_OFFICIAL_DATA_RESEARCH": "true",
            "DIRECT_EVIDENCE_MAX_ITEMS": "30",
            "ENABLE_DEEP_RESEARCH_ON_DISAGREEMENT": "false",
            "BINARY_ESCALATION_SPREAD": "0.25",
            "MULTIPLE_CHOICE_ESCALATION_SPREAD": "0.28",
            "NUMERIC_ESCALATION_RANGE_FRACTION": "0.15",
        },
    ),
    ExperimentVariant(
        id=2,
        name="deep_research_sensitive",
        description="Escalate and deep-research sooner when cheap roles disagree.",
        env_overrides={
            "ENABLE_DEEP_RESEARCH_ON_DISAGREEMENT": "true",
            "ENABLE_DEEP_RESEARCH_ON_HIGH_VALUE": "true",
            "BINARY_ESCALATION_SPREAD": "0.15",
            "MULTIPLE_CHOICE_ESCALATION_SPREAD": "0.18",
            "NUMERIC_ESCALATION_RANGE_FRACTION": "0.09",
            "NUMERIC_ESCALATION_RELATIVE_FRACTION": "0.35",
        },
    ),
    ExperimentVariant(
        id=3,
        name="same_model_deepseek_roles",
        description="Use DeepSeek for all cheap roles; duplicate-model role passes are bundled before aggregation.",
        env_overrides={
            "BASE_RATE_FORECASTER_MODEL": "openrouter/deepseek/deepseek-v4-pro",
            "INSIDE_VIEW_FORECASTER_MODEL": "openrouter/deepseek/deepseek-v4-pro",
            "MARKET_DATA_FORECASTER_MODEL": "openrouter/deepseek/deepseek-v4-pro",
        },
    ),
    ExperimentVariant(
        id=4,
        name="same_model_mistral_roles",
        description="Use Mistral Large for all cheap roles as a lower-cost Western/European control.",
        env_overrides={
            "BASE_RATE_FORECASTER_MODEL": "openrouter/mistralai/mistral-large-2512",
            "INSIDE_VIEW_FORECASTER_MODEL": "openrouter/mistralai/mistral-large-2512",
            "MARKET_DATA_FORECASTER_MODEL": "openrouter/mistralai/mistral-large-2512",
        },
    ),
    ExperimentVariant(
        id=5,
        name="thin_research_control",
        description="Disable direct structured, market-prior, official-data, and deep-research passes to measure research-layer value.",
        env_overrides={
            "ENABLE_DIRECT_STRUCTURED_RESEARCH": "false",
            "ENABLE_MARKET_PRIOR_RESEARCH": "false",
            "ENABLE_OFFICIAL_DATA_RESEARCH": "false",
            "ENABLE_DEEP_RESEARCH_ON_DISAGREEMENT": "false",
            "ENABLE_DEEP_RESEARCH_ON_HIGH_VALUE": "false",
        },
    ),
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %s", name, raw_value, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value.strip())
    except ValueError:
        logger.warning("Invalid float for %s=%r; using %s", name, raw_value, default)
        return default


def _env_is_set(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def _required_key_for_model(model: str) -> str | None:
    if model.startswith("openrouter/"):
        return "OPENROUTER_API_KEY"
    if model.startswith("openai/"):
        return "OPENAI_API_KEY"
    if model.startswith("perplexity/"):
        return "PERPLEXITY_API_KEY"
    if model.startswith("anthropic/"):
        return "ANTHROPIC_API_KEY"
    if model.startswith("exa/"):
        return "EXA_API_KEY"
    return None


def _log_startup_key_status() -> None:
    if not _env_is_set("METACULUS_TOKEN"):
        logger.warning(
            "METACULUS_TOKEN is not set. The bot will not be able to publish forecasts to Metaculus."
        )

    configured_models = {
        "DEFAULT_FORECASTER_MODEL": os.getenv(
            "DEFAULT_FORECASTER_MODEL", "openrouter/mistralai/mistral-large-2512"
        ),
        "SUMMARIZER_MODEL": os.getenv("SUMMARIZER_MODEL", "openrouter/openai/gpt-5-nano"),
        "PARSER_MODEL": os.getenv("PARSER_MODEL", "openrouter/openai/gpt-5-nano"),
        "ESCALATION_FORECASTER_MODEL": os.getenv(
            "ESCALATION_FORECASTER_MODEL", "openrouter/openai/gpt-5.5"
        ),
        "SCOUT_MODEL": os.getenv("SCOUT_MODEL", "openrouter/z-ai/glm-5.1"),
    }
    missing_model_keys = sorted(
        {
            required_key
            for model in configured_models.values()
            if (required_key := _required_key_for_model(model))
            and not _env_is_set(required_key)
        }
    )
    if missing_model_keys:
        logger.warning(
            "Missing model API key(s): %s. Configured LLM calls may fail.",
            ", ".join(missing_model_keys),
        )

    researcher = os.getenv("RESEARCHER_MODEL", "random").strip()
    fallback = os.getenv("RESEARCHER_FALLBACK_MODEL", "exa").strip()
    if researcher.startswith("asknews/") and (
        not _env_is_set("ASKNEWS_CLIENT_ID") or not _env_is_set("ASKNEWS_SECRET")
    ):
        logger.warning(
            "AskNews is configured but ASKNEWS_CLIENT_ID/ASKNEWS_SECRET are not set. "
            "Configured research will fall back to %s.",
            fallback or "no fallback",
        )
    if researcher in {"exa", "exa/auto"} or fallback in {"exa", "exa/auto"}:
        if not _env_is_set("EXA_API_KEY"):
            logger.warning("Exa researcher is configured but EXA_API_KEY is not set.")
    if fallback in {"perplexity", "sonar", "perplexity/auto"} and not (
        _env_is_set("PERPLEXITY_API_KEY") or _env_is_set("OPENROUTER_API_KEY")
    ):
        logger.warning(
            "Perplexity fallback is configured but neither PERPLEXITY_API_KEY nor OPENROUTER_API_KEY is set."
        )


def _select_experiment_variant(
    mode: str, requested_variant_id: int | None, seed: int | None
) -> tuple[ExperimentVariant | None, int | None]:
    if requested_variant_id is not None and mode == "off":
        mode = "variant"
    if mode == "off":
        return None, seed
    variants_by_id = {variant.id: variant for variant in EXPERIMENT_VARIANTS}
    if mode == "variant":
        if requested_variant_id is None:
            raise ValueError("--experiment-variant is required when --experiment-mode=variant")
        if requested_variant_id not in variants_by_id:
            raise ValueError(
                f"Unknown experiment variant {requested_variant_id}. "
                f"Available variants: {sorted(variants_by_id)}"
            )
        return variants_by_id[requested_variant_id], seed

    if seed is None:
        seed = int(time.time_ns() % (2**31 - 1))
    rng = random.Random(seed)
    return rng.choice(EXPERIMENT_VARIANTS), seed


def _apply_experiment_variant(variant: ExperimentVariant | None) -> None:
    if variant is None:
        return
    for env_name, env_value in variant.env_overrides.items():
        os.environ[env_name] = env_value
    logger.info(
        "Selected experiment variant %s (%s): %s",
        variant.id,
        variant.name,
        variant.description,
    )


def _json_safe(value: Any, max_text_chars: int = 25000) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value[:max_text_chars]
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": repr(value)}
    if dataclasses := getattr(value, "__dataclass_fields__", None):
        try:
            return _json_safe(asdict(value), max_text_chars=max_text_chars)
        except Exception:
            return str(dataclasses)[:max_text_chars]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(), max_text_chars=max_text_chars)
        except Exception:
            pass
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, max_text_chars=max_text_chars)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item, max_text_chars=max_text_chars)
            for item in value
        ]
    return str(value)[:max_text_chars]


def _question_log_record(question: Any) -> dict[str, Any]:
    if question is None:
        return {}
    field_names = [
        "id",
        "id_of_post",
        "question_text",
        "page_url",
        "background_info",
        "resolution_criteria",
        "fine_print",
        "close_time",
        "scheduled_resolution_time",
        "conditional_type",
    ]
    record = {
        field_name: _json_safe(getattr(question, field_name, None), max_text_chars=6000)
        for field_name in field_names
    }
    record["question_object_type"] = type(question).__name__
    return record


def _report_log_record(report: Any) -> dict[str, Any]:
    if isinstance(report, BaseException):
        return {
            "status": "exception",
            "report_type": type(report).__name__,
            "exception": _json_safe(report),
        }
    question = getattr(report, "question", None)
    return {
        "status": "ok",
        "report_type": type(report).__name__,
        "question": _question_log_record(question),
        "prediction": _json_safe(getattr(report, "prediction", None)),
        "explanation": _json_safe(getattr(report, "explanation", "")),
        "research": _json_safe(getattr(report, "research", "")),
    }


def _summarize_forecast_reports(
    forecast_reports: list[Any], *, publish_reports: bool
) -> dict[str, Any]:
    ok_reports = [
        report for report in forecast_reports if not isinstance(report, BaseException)
    ]
    exception_reports = [
        report for report in forecast_reports if isinstance(report, BaseException)
    ]
    question_records = [
        _question_log_record(getattr(report, "question", None))
        for report in ok_reports
    ]
    question_ids = [
        record.get("id") or record.get("id_of_post")
        for record in question_records
        if record.get("id") or record.get("id_of_post")
    ]
    if ok_reports:
        status = "forecasted"
        message = (
            f"Forecasted {len(ok_reports)} question(s)"
            f"{' and attempted to publish them' if publish_reports else ' in practice mode'}."
        )
    elif exception_reports:
        status = "errors"
        message = f"No successful forecasts; {len(exception_reports)} exception(s) returned."
    else:
        status = "no_eligible_questions"
        message = (
            "No forecast reports were returned. This usually means the target had no "
            "eligible open questions, or all open questions were skipped because the bot "
            "had already forecasted them."
        )
    return {
        "status": status,
        "message": message,
        "reports_returned": len(forecast_reports),
        "successful_forecasts": len(ok_reports),
        "exceptions": len(exception_reports),
        "publish_reports_to_metaculus": publish_reports,
        "question_ids": question_ids,
        "questions": question_records,
    }


def _run_summary_markdown(
    *,
    run_id: str,
    mode: str,
    target_tournament_ids: list[str | int],
    selected_variant: ExperimentVariant | None,
    experiment_seed: int | None,
    summary: dict[str, Any],
) -> str:
    target_text = ", ".join(str(target_id) for target_id in target_tournament_ids)
    variant_text = (
        f"{selected_variant.id} ({selected_variant.name})"
        if selected_variant
        else "off"
    )
    question_ids = summary.get("question_ids") or []
    question_text = (
        ", ".join(str(question_id) for question_id in question_ids)
        if question_ids
        else "none"
    )
    return clean_indents(
        f"""
        # Forecast Run Status

        **Status:** {summary["status"]}

        {summary["message"]}

        - Run ID: `{run_id}`
        - Mode: `{mode}`
        - Target: `{target_text}`
        - Publish to Metaculus: `{summary["publish_reports_to_metaculus"]}`
        - Reports returned: `{summary["reports_returned"]}`
        - Successful forecasts: `{summary["successful_forecasts"]}`
        - Exceptions: `{summary["exceptions"]}`
        - Question IDs: `{question_text}`
        - Experiment variant: `{variant_text}`
        - Experiment seed: `{experiment_seed}`
        """
    ).strip() + "\n"


def _write_experiment_logs(
    forecast_reports: list[Any],
    *,
    run_id: str,
    log_dir: Path,
    mode: str,
    target_tournament_ids: list[str | int],
    selected_variant: ExperimentVariant | None,
    experiment_seed: int | None,
    publish_reports: bool,
) -> None:
    if not _env_bool("ENABLE_EXPERIMENT_LOGGING", True):
        return
    run_dir = log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = _summarize_forecast_reports(
        forecast_reports, publish_reports=publish_reports
    )
    variant_payload = (
        {
            "id": selected_variant.id,
            "name": selected_variant.name,
            "description": selected_variant.description,
            "env_overrides": selected_variant.env_overrides,
        }
        if selected_variant
        else None
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "target_tournament_ids": target_tournament_ids,
        "publish_reports_to_metaculus": publish_reports,
        "experiment_seed": experiment_seed,
        "selected_variant": variant_payload,
        "runtime_models": {
            name: os.getenv(name)
            for name in (
                "BASE_RATE_FORECASTER_MODEL",
                "INSIDE_VIEW_FORECASTER_MODEL",
                "MARKET_DATA_FORECASTER_MODEL",
                "ESCALATION_FORECASTER_MODEL",
                "SCOUT_MODEL",
                "SUMMARIZER_MODEL",
                "PARSER_MODEL",
            )
        },
        "runtime_research_controls": {
            name: os.getenv(name)
            for name in (
                "ENABLE_DIRECT_STRUCTURED_RESEARCH",
                "ENABLE_MARKET_PRIOR_RESEARCH",
                "ENABLE_OFFICIAL_DATA_RESEARCH",
                "ENABLE_DEEP_RESEARCH_ON_DISAGREEMENT",
                "ENABLE_DEEP_RESEARCH_ON_HIGH_VALUE",
                "RESEARCHER_MODEL",
                "RESEARCHER_FALLBACK_MODEL",
                "RESEARCHER_RANDOM_MODELS",
                "RESEARCHER_RANDOM_SEED",
                "SCOUT_RESEARCHER_MODEL",
                "PERPLEXITY_RESEARCHER_MODEL",
                "OPENROUTER_PERPLEXITY_RESEARCHER_MODEL",
                "EXA_RESEARCHER_MODEL",
            )
        },
        "forecast_summary": summary,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (run_dir / "run_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "run_summary.md").write_text(
        _run_summary_markdown(
            run_id=run_id,
            mode=mode,
            target_tournament_ids=target_tournament_ids,
            selected_variant=selected_variant,
            experiment_seed=experiment_seed,
            summary=summary,
        ),
        encoding="utf-8",
    )
    jsonl_path = run_dir / "forecast_reports.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for report in forecast_reports:
            record = {
                "run_id": run_id,
                "variant": variant_payload,
                "report": _report_log_record(report),
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    logger.info("Experiment logs written to %s", run_dir)
    logger.info("Forecast run status: %s - %s", summary["status"], summary["message"])


def _get_open_tournament_questions(
    client: MetaculusClient, tournament_id: str | int
) -> list[MetaculusQuestion]:
    method = getattr(client, "get_all_open_questions_from_tournament", None)
    if method is None:
        raise AttributeError(
            "MetaculusClient does not expose get_all_open_questions_from_tournament; "
            "omit --max-questions to use ForecastBot.forecast_on_tournament directly."
        )
    try:
        return list(method(tournament_id=tournament_id))
    except TypeError:
        return list(method(tournament_id))


def _select_question_batch(
    questions: list[MetaculusQuestion],
    *,
    max_questions: int | None,
    shuffle_seed: int | None,
) -> list[MetaculusQuestion]:
    if max_questions is None or max_questions <= 0 or len(questions) <= max_questions:
        return questions
    selected_questions = list(questions)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(selected_questions)
    return selected_questions[:max_questions]


def _filter_previously_forecasted_questions(
    questions: list[MetaculusQuestion],
) -> list[MetaculusQuestion]:
    return [
        question
        for question in questions
        if not bool(getattr(question, "already_forecasted", False))
    ]


def _split_csv_args(values: list[str] | None) -> list[str]:
    items: list[str] = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                items.append(item)
    return items


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        unique_items.append(item)
    return unique_items


def _looks_like_credit_exhaustion(error: BaseException) -> bool:
    error_text = repr(error).lower()
    return any(
        phrase in error_text
        for phrase in (
            "insufficient credits",
            "requires more credits",
            "payment required",
            "quota",
        )
    )


def _forecast_questions_resiliently(
    bot: ForecastBot, questions: list[MetaculusQuestion]
) -> list[Any]:
    forecast_reports: list[Any] = []
    for index, question in enumerate(questions, start=1):
        logger.info(
            "Forecasting selected question %s/%s: %s",
            index,
            len(questions),
            getattr(question, "page_url", ""),
        )
        try:
            forecast_reports.extend(
                asyncio.run(bot.forecast_questions([question], return_exceptions=True))
            )
        except BaseException as error:
            forecast_reports.append(error)
            logger.error(
                "Question-level forecasting failed for %s: %r",
                getattr(question, "page_url", ""),
                error,
            )
            if _looks_like_credit_exhaustion(error):
                logger.error(
                    "Stopping batch early because the model provider appears to be out of credits/quota."
                )
                break
    return forecast_reports


class SpringTemplateBot2026(ForecastBot):
    """
    This is the template bot for Spring 2026 Metaculus AI Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    Main changes since Fall:
    - Additional prompting has been added to numeric questions to emphasize putting pecentile values in the correct order.
    - Support for conditional and date questions has been added
    - Note: Spring AIB will not use date/conditional questions, so these are only for forecasting on the main site as you wish.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-5-nano",
            "researcher": "random",
            "parser": "openai/gpt-5-nano",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name"
    if research_strategy == "asknews/news-summaries":
        ...
    elif research_strategy == "random":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2
    _min_successful_ensemble_predictions = 1
    _binary_escalation_spread = 0.20
    _multiple_choice_escalation_spread = 0.22
    _numeric_escalation_range_fraction = 0.12
    _numeric_escalation_relative_fraction = 0.50
    _research_cache_dir = Path("research_cache")
    _direct_evidence_max_items_per_provider = 5
    _stop_words = {
        "about",
        "above",
        "after",
        "again",
        "against",
        "between",
        "before",
        "below",
        "could",
        "does",
        "during",
        "from",
        "have",
        "into",
        "more",
        "most",
        "over",
        "question",
        "resolve",
        "resolution",
        "than",
        "that",
        "their",
        "there",
        "this",
        "through",
        "will",
        "with",
        "would",
    }
    _default_high_value_deep_research_keywords = (
        "frontier ai,agi,artificial intelligence,ai benchmark,llm,model release,"
        "election,poll,president,congress,parliament,inflation,gdp,cpi,unemployment,"
        "interest rate,central bank,fed,ecb,stock,market cap,earnings,crypto,"
        "war,ceasefire,nuclear,sanction,treaty,geopolitical"
    )
    _official_data_topic_keywords: dict[str, tuple[str, ...]] = {
        "economic": (
            "inflation",
            "cpi",
            "gdp",
            "unemployment",
            "jobs",
            "payroll",
            "interest rate",
            "fed",
            "federal reserve",
            "ecb",
            "central bank",
            "recession",
            "trade deficit",
            "exports",
            "imports",
            "debt",
            "deficit",
        ),
        "finance": (
            "stock",
            "share price",
            "market cap",
            "earnings",
            "revenue",
            "profit",
            "bankruptcy",
            "ipo",
            "merger",
            "acquisition",
            "sec",
            "filing",
            "crypto",
            "bitcoin",
            "ethereum",
        ),
        "ai": (
            "ai",
            "artificial intelligence",
            "frontier model",
            "llm",
            "benchmark",
            "swe-bench",
            "mmlu",
            "chatbot arena",
            "compute",
            "gpu",
            "h100",
            "h200",
            "b200",
            "model release",
        ),
        "polling": (
            "election",
            "poll",
            "vote share",
            "approval",
            "president",
            "congress",
            "senate",
            "house",
            "parliament",
            "referendum",
            "primary",
        ),
        "geopolitical": (
            "war",
            "ceasefire",
            "invasion",
            "missile",
            "nuclear",
            "sanction",
            "treaty",
            "united nations",
            "nato",
            "iaea",
            "who",
            "conflict",
            "military",
            "geopolitical",
        ),
    }
    _official_data_source_hints: dict[str, str] = {
        "economic": (
            "FRED, BLS, BEA, Census, Treasury, Federal Reserve, ECB, Eurostat, "
            "OECD, World Bank, IMF, national statistical offices, and central bank releases"
        ),
        "finance": (
            "SEC EDGAR, company investor relations, exchange filings, central bank data, "
            "Yahoo Finance/Nasdaq only as secondary market-data references"
        ),
        "ai": (
            "official lab announcements, model cards, technical reports, arXiv papers, "
            "Epoch AI, Artificial Analysis, LMSYS/Chatbot Arena, SWE-bench, METR, MLPerf, "
            "Papers With Code, and benchmark source pages"
        ),
        "polling": (
            "official election authorities, FEC/electoral commission data, polling averages, "
            "pollster releases, FiveThirtyEight/Nate Silver-style aggregators, and reputable election trackers"
        ),
        "geopolitical": (
            "UN, NATO, IAEA, WHO, IMF/World Bank, government ministries, official statements, "
            "treaty texts, sanctions lists, ACLED/ISW-style conflict trackers, and major wire services"
        ),
    }
    _market_providers = {"kalshi", "polymarket", "manifold"}
    _country_alias_terms: dict[str, tuple[str, ...]] = {
        "argentina": ("argentina", "argentine", "argentinian"),
        "australia": ("australia", "australian"),
        "brazil": ("brazil", "brazilian"),
        "canada": ("canada", "canadian"),
        "chile": ("chile", "chilean"),
        "china": ("china", "chinese"),
        "colombia": ("colombia", "colombian"),
        "france": ("france", "french"),
        "germany": ("germany", "german"),
        "india": ("india", "indian"),
        "indonesia": ("indonesia", "indonesian"),
        "iran": ("iran", "iranian"),
        "israel": ("israel", "israeli"),
        "italy": ("italy", "italian"),
        "japan": ("japan", "japanese"),
        "mexico": ("mexico", "mexican"),
        "russia": ("russia", "russian"),
        "south korea": ("south korea", "korea", "korean"),
        "taiwan": ("taiwan", "taiwanese"),
        "turkey": ("turkey", "turkish"),
        "ukraine": ("ukraine", "ukrainian"),
        "united kingdom": ("united kingdom", "uk", "britain", "british"),
        "united states": (
            "united states",
            "u.s.",
            "us",
            "usa",
            "american",
        ),
        "venezuela": ("venezuela", "venezuelan"),
    }
    _election_market_terms = {
        "ballot",
        "candidate",
        "election",
        "electoral",
        "nominee",
        "parliament",
        "president",
        "presidential",
        "primary",
        "runoff",
        "vote",
        "winner",
    }
    _fred_series_by_keyword: dict[str, tuple[str, str]] = {
        "inflation": ("CPIAUCSL", "Consumer Price Index for All Urban Consumers"),
        "cpi": ("CPIAUCSL", "Consumer Price Index for All Urban Consumers"),
        "core cpi": ("CPILFESL", "Core CPI"),
        "unemployment": ("UNRATE", "Unemployment Rate"),
        "jobless": ("UNRATE", "Unemployment Rate"),
        "payroll": ("PAYEMS", "All Employees, Total Nonfarm"),
        "employment": ("PAYEMS", "All Employees, Total Nonfarm"),
        "gdp": ("GDP", "Gross Domestic Product"),
        "real gdp": ("GDPC1", "Real Gross Domestic Product"),
        "recession": ("USREC", "NBER based Recession Indicators"),
        "fed funds": ("FEDFUNDS", "Effective Federal Funds Rate"),
        "interest rate": ("FEDFUNDS", "Effective Federal Funds Rate"),
        "mortgage": ("MORTGAGE30US", "30-Year Fixed Rate Mortgage Average"),
        "treasury": ("DGS10", "10-Year Treasury Constant Maturity Rate"),
        "oil": ("DCOILWTICO", "WTI Crude Oil Price"),
        "gas": ("GASREGW", "US Regular Conventional Gas Price"),
    }
    _ai_benchmark_registry: tuple[tuple[str, str, str], ...] = (
        (
            "LMArena Chatbot Arena",
            "https://lmarena.ai/leaderboard/",
            "Crowd-sourced pairwise model preference leaderboard.",
        ),
        (
            "Artificial Analysis",
            "https://artificialanalysis.ai/",
            "Model capability, speed, and price comparisons.",
        ),
        (
            "SWE-bench",
            "https://www.swebench.com/",
            "Software engineering benchmark leaderboard.",
        ),
        (
            "METR",
            "https://metr.org/",
            "AI capability evaluations and task-completion research.",
        ),
        (
            "MLPerf",
            "https://mlcommons.org/benchmarks/",
            "MLCommons training/inference benchmark suites.",
        ),
        (
            "Epoch AI",
            "https://epoch.ai/",
            "AI trends, compute, benchmark, and model release data.",
        ),
    )

    @classmethod
    def _base_forecaster_specs(cls) -> list[ForecasterRoleSpec]:
        """
        Cheap, diverse first-pass ensemble with distinct reasoning roles.
        GPT-5.5 is intentionally excluded and only used when this group disagrees.
        """
        return [
            ForecasterRoleSpec(
                name="Base-rate / outside-view forecaster",
                llm=GeneralLlm(
                    model=os.getenv(
                        "BASE_RATE_FORECASTER_MODEL",
                        "openrouter/deepseek/deepseek-v4-pro",
                    ),
                    temperature=0.25,
                    timeout=240,
                    allowed_tries=2,
                    max_tokens=_env_int("FORECASTER_MAX_TOKENS", 4096),
                ),
                instructions=clean_indents(
                    """
                    Your primary role is the outside view.
                    - Start from the best reference class, not the most vivid recent evidence.
                    - Look for historical analogues, prior frequencies, base rates, and the status quo/no-change outcome.
                    - State the base-rate prior before applying question-specific updates.
                    - Be conservative about trend extrapolation unless the reference class supports it.
                    - Still produce a complete forecast in the exact final format requested.
                    """
                ),
            ),
            ForecasterRoleSpec(
                name="Inside-view / current-evidence forecaster",
                llm=GeneralLlm(
                    model=os.getenv(
                        "INSIDE_VIEW_FORECASTER_MODEL",
                        "openrouter/z-ai/glm-5.1",
                    ),
                    temperature=0.25,
                    timeout=180,
                    allowed_tries=2,
                    max_tokens=_env_int("FORECASTER_MAX_TOKENS", 4096),
                ),
                instructions=clean_indents(
                    """
                    Your primary role is the inside view.
                    - Focus on mechanisms, causal pathways, incentives, constraints, and latest evidence.
                    - Identify recent developments, trend breaks, catalysts, and near-term watchpoints.
                    - Explain which facts would move the forecast materially up/down or earlier/later.
                    - Check whether current evidence is strong enough to overcome the base rate.
                    - Still produce a complete forecast in the exact final format requested.
                    """
                ),
            ),
            ForecasterRoleSpec(
                name="Market / structured-data interpreter",
                llm=GeneralLlm(
                    model=os.getenv(
                        "MARKET_DATA_FORECASTER_MODEL",
                        "openrouter/mistralai/mistral-large-2512",
                    ),
                    temperature=0.25,
                    timeout=180,
                    allowed_tries=2,
                    max_tokens=_env_int("FORECASTER_MAX_TOKENS", 4096),
                ),
                instructions=clean_indents(
                    """
                    Your primary role is interpreting markets and structured data.
                    - Map Kalshi, Polymarket, Manifold, FRED, SEC, polling, AI benchmark, and other direct evidence to the exact resolution criteria.
                    - Distinguish direct evidence from merely similar markets/data.
                    - Convert market prices or official values into forecast-relevant priors only when the mapping is defensible.
                    - Flag caveats: liquidity, stale data, different resolution dates, different thresholds, revisions, and selection bias.
                    - If no direct market/data evidence exists, say so and make a normal forecast from the remaining evidence.
                    - Still produce a complete forecast in the exact final format requested.
                    """
                ),
            ),
        ]

    @classmethod
    def _base_forecaster_llms(cls) -> list[GeneralLlm]:
        return [spec.llm for spec in cls._base_forecaster_specs()]

    @classmethod
    def _escalation_forecaster_spec(cls) -> ForecasterRoleSpec:
        return ForecasterRoleSpec(
            name="Stacker / skeptic adjudicator",
            llm=GeneralLlm(
                model=os.getenv(
                    "ESCALATION_FORECASTER_MODEL", "openrouter/openai/gpt-5.5"
                ),
                temperature=0.2,
                timeout=180,
                allowed_tries=2,
                max_tokens=_env_int("ESCALATION_MAX_TOKENS", 4096),
            ),
            instructions=clean_indents(
                """
                Your role is the final adjudicator.
                - Treat the cheaper model outputs as role-specific evidence, not independent votes.
                - Red-team the resolution criteria, stale research, bad analogies, and market/data mismatches.
                - Decide which role found the most resolution-relevant evidence and which assumptions should be discounted.
                - Produce the final forecast only after weighing the outside view, inside view, and market/data view.
                - Still use the exact final answer format requested.
                """
            ),
        )

    @classmethod
    def _escalation_forecaster_llm(cls) -> GeneralLlm:
        return cls._escalation_forecaster_spec().llm

    @staticmethod
    def _role_prompt(prompt: str, role_spec: ForecasterRoleSpec) -> str:
        return clean_indents(
            f"""
            You are acting as the {role_spec.name}.

            Role-specific instructions:
            {role_spec.instructions}

            ---

            {prompt}
            """
        )

    @classmethod
    def _forecasting_checklist(cls) -> str:
        return clean_indents(
            """
            Forecasting discipline:
            - Restate the exact resolution criteria and check whether the question is already effectively resolved.
            - Separate the outside view/base rate from the inside view/current evidence.
            - Identify the most important cruxes and what would change your mind.
            - Watch for bait-and-switch errors: answer the question as written, not a nearby easier question.
            - Calibrate uncertainty: avoid both false precision and lazy 50/50 hedging.
            - If markets, expert forecasts, polls, official data, or historical reference classes are relevant, weigh them explicitly.
            - Include a section headed exactly "Market prior reconciliation" before your final answer. State whether direct Kalshi, Polymarket, Manifold, or other market odds were found; list any implied probability, liquidity/directness caveat, and why your forecast agrees with or deviates from the market. If only weak or irrelevant markets were found, say they were not used.
            """
        )

    async def _gather_ensemble_predictions(
        self,
        tasks: list[asyncio.Task[ReasonedPrediction[T]]],
        question: MetaculusQuestion,
    ) -> list[ReasonedPrediction[T]]:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        predictions: list[ReasonedPrediction[T]] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                errors.append(repr(result))
            else:
                predictions.append(result)

        if errors:
            logger.warning(
                "Ensemble members failed for URL %s: %s", question.page_url, errors
            )
        if len(predictions) < self._min_successful_ensemble_predictions:
            raise RuntimeError(
                f"All ensemble forecasters failed for {question.page_url}. Errors: {errors}"
            )
        return predictions

    @classmethod
    def _market_reconciliation_from_predictions(
        cls, predictions: list[ReasonedPrediction[T]]
    ) -> str:
        sections: list[str] = []
        section_pattern = re.compile(
            r"(?:^|\n)#{0,3}\s*Market prior reconciliation\s*:?\s*\n(?P<body>.*?)(?=\n#{1,3}\s|\Z)",
            flags=re.IGNORECASE | re.DOTALL,
        )
        ordered_predictions = sorted(
            predictions,
            key=lambda prediction: (
                0
                if "Role: Market / structured-data interpreter"
                in prediction.reasoning
                else 1
                if "Role: Stacker / skeptic adjudicator" in prediction.reasoning
                else 2
            ),
        )
        for prediction in ordered_predictions:
            reasoning = prediction.reasoning
            if "market" not in reasoning.lower():
                continue
            for match in section_pattern.finditer(reasoning):
                body = match.group("body").strip()
                if body:
                    sections.append(cls._truncate_for_prompt(body, 1800))
                    break
        if sections:
            return "\n\n".join(sections[:2])
        return (
            "No separate market-prior reconciliation was extracted from the role "
            "outputs. See the market/data role below for any market evidence that "
            "was found or rejected."
        )

    @classmethod
    def _combine_model_reasoning(cls, predictions: list[ReasonedPrediction[T]]) -> str:
        model_rationales = "\n\n".join(
            f"## Role output {i + 1}\n{prediction.reasoning}"
            for i, prediction in enumerate(predictions)
        )
        market_reconciliation = cls._market_reconciliation_from_predictions(
            predictions
        )
        return clean_indents(
            f"""
            Ensemble forecast synthesized from {len(predictions)} successful role-specific forecasts.

            ## Market prior reconciliation
            {market_reconciliation}

            {model_rationales}
            """
        )

    @classmethod
    def _parse_numeric_percentiles_from_text(cls, text: str) -> list[Percentile] | None:
        percentile_values: dict[float, float] = {}
        percentile_line_pattern = re.compile(
            r"""
            (?:
                percentile\s+
                (?P<pct_after_label>\d+(?:\.\d+)?)
                |
                (?P<pct_before_label>\d+(?:\.\d+)?)(?:st|nd|rd|th)?\s+percentile
            )
            \s*(?:[:=\-]|is|at|around|of)?\s*
            (?P<value>.*)
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        for line in text.splitlines():
            match = percentile_line_pattern.search(line)
            if not match:
                continue
            raw_percentile = match.group("pct_after_label") or match.group(
                "pct_before_label"
            )
            raw_value = match.group("value")
            try:
                percentile = float(raw_percentile)
            except (TypeError, ValueError):
                continue
            percentile = percentile / 100 if percentile > 1 else percentile
            value_match = re.search(
                r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|[-+]?\.\d+",
                raw_value.replace(",", ""),
            )
            if not value_match:
                continue
            value = float(value_match.group(0))
            percentile_values[percentile] = value

        if len(percentile_values) < 3 or 0.5 not in percentile_values:
            return None
        return [
            Percentile(percentile=percentile, value=value)
            for percentile, value in sorted(percentile_values.items())
        ]

    @staticmethod
    def _model_key_from_reasoning(
        prediction: ReasonedPrediction[T], fallback: str
    ) -> str:
        for line in prediction.reasoning.splitlines():
            if line.startswith("Model: "):
                return line.removeprefix("Model: ").strip()
        return fallback

    @classmethod
    def _group_predictions_by_model(
        cls, predictions: list[ReasonedPrediction[T]]
    ) -> dict[str, list[ReasonedPrediction[T]]]:
        groups: dict[str, list[ReasonedPrediction[T]]] = {}
        for index, prediction in enumerate(predictions):
            model_key = cls._model_key_from_reasoning(
                prediction, fallback=f"unknown-model-{index}"
            )
            groups.setdefault(model_key, []).append(prediction)
        return groups

    @staticmethod
    def _same_model_role_reasoning(
        model_key: str, predictions: list[ReasonedPrediction[T]]
    ) -> str:
        role_outputs = "\n\n".join(
            f"### Same-model role pass {i + 1}\nPrediction: {prediction.prediction_value}\n\n{prediction.reasoning}"
            for i, prediction in enumerate(predictions)
        )
        return clean_indents(
            f"""
            Same-model role bundle.
            Model: {model_key}

            These role passes use the same underlying model, so they are treated as lenses from one model family rather than independent votes.

            {role_outputs}
            """
        )

    @classmethod
    def _collapse_same_model_binary_predictions(
        cls, predictions: list[ReasonedPrediction[float]]
    ) -> list[ReasonedPrediction[float]]:
        collapsed: list[ReasonedPrediction[float]] = []
        for model_key, model_predictions in cls._group_predictions_by_model(
            predictions
        ).items():
            if len(model_predictions) == 1:
                collapsed.append(model_predictions[0])
                continue
            collapsed.append(
                ReasonedPrediction(
                    prediction_value=cls._logit_mean_probability(
                        [
                            prediction.prediction_value
                            for prediction in model_predictions
                        ]
                    ),
                    reasoning=cls._same_model_role_reasoning(
                        model_key, model_predictions
                    ),
                )
            )
        return collapsed

    async def _collapse_same_model_multiple_choice_predictions(
        self,
        predictions: list[ReasonedPrediction[PredictedOptionList]],
        question: MultipleChoiceQuestion,
    ) -> list[ReasonedPrediction[PredictedOptionList]]:
        collapsed: list[ReasonedPrediction[PredictedOptionList]] = []
        for model_key, model_predictions in self._group_predictions_by_model(
            predictions
        ).items():
            if len(model_predictions) == 1:
                collapsed.append(model_predictions[0])
                continue
            prediction_value = await MultipleChoiceReport.aggregate_predictions(
                [
                    prediction.prediction_value
                    for prediction in model_predictions
                ],
                question,
            )
            collapsed.append(
                ReasonedPrediction(
                    prediction_value=prediction_value,
                    reasoning=self._same_model_role_reasoning(
                        model_key, model_predictions
                    ),
                )
            )
        return collapsed

    async def _collapse_same_model_numeric_predictions(
        self,
        predictions: list[ReasonedPrediction[NumericDistribution]],
        question: NumericQuestion | DateQuestion,
    ) -> list[ReasonedPrediction[NumericDistribution]]:
        collapsed: list[ReasonedPrediction[NumericDistribution]] = []
        for model_key, model_predictions in self._group_predictions_by_model(
            predictions
        ).items():
            if len(model_predictions) == 1:
                collapsed.append(model_predictions[0])
                continue
            prediction_value = await NumericReport.aggregate_predictions(
                [
                    prediction.prediction_value
                    for prediction in model_predictions
                ],
                question,
            )
            collapsed.append(
                ReasonedPrediction(
                    prediction_value=prediction_value,
                    reasoning=self._same_model_role_reasoning(
                        model_key, model_predictions
                    ),
                )
            )
        return collapsed

    @staticmethod
    def _truncate_for_prompt(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "\n...[truncated]"

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw_value = os.getenv(name)
        if raw_value is None:
            return default
        try:
            return float(raw_value)
        except ValueError:
            logger.warning("Ignoring invalid float env var %s=%r", name, raw_value)
            return default

    @classmethod
    def apply_runtime_config_from_env(cls) -> None:
        cls._binary_escalation_spread = cls._env_float(
            "BINARY_ESCALATION_SPREAD", cls._binary_escalation_spread
        )
        cls._multiple_choice_escalation_spread = cls._env_float(
            "MULTIPLE_CHOICE_ESCALATION_SPREAD",
            cls._multiple_choice_escalation_spread,
        )
        cls._numeric_escalation_range_fraction = cls._env_float(
            "NUMERIC_ESCALATION_RANGE_FRACTION",
            cls._numeric_escalation_range_fraction,
        )
        cls._numeric_escalation_relative_fraction = cls._env_float(
            "NUMERIC_ESCALATION_RELATIVE_FRACTION",
            cls._numeric_escalation_relative_fraction,
        )

    @staticmethod
    def _question_text_blob(question: MetaculusQuestion) -> str:
        parts = [
            getattr(question, "question_text", ""),
            getattr(question, "background_info", ""),
            getattr(question, "resolution_criteria", ""),
            getattr(question, "fine_print", ""),
        ]
        options = getattr(question, "options", None)
        if options:
            parts.append(" ".join(str(option) for option in options))
        return "\n".join(str(part) for part in parts if part).lower()

    @staticmethod
    def _normalise_relevance_text(text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", str(text))
        ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
        return ascii_text.lower()

    @classmethod
    def _text_tokens(cls, text: str) -> set[str]:
        return {
            word
            for word in re.findall(
                r"[a-z][a-z0-9\-]{2,}", cls._normalise_relevance_text(text)
            )
            if len(word) > 2
        }

    @classmethod
    def _question_option_terms(cls, question: MetaculusQuestion) -> list[set[str]]:
        option_terms: list[set[str]] = []
        for option in getattr(question, "options", None) or []:
            option_text = str(option).strip()
            if option_text.lower() in {
                "another",
                "another candidate",
                "none",
                "none of the above",
                "other",
                "other candidate",
            }:
                continue
            terms = {
                term
                for term in cls._text_tokens(option_text)
                if term not in cls._stop_words and len(term) > 2
            }
            if terms:
                option_terms.append(terms)
        return option_terms

    @classmethod
    def _country_alias_groups_for_question(
        cls, question: MetaculusQuestion
    ) -> list[set[str]]:
        question_blob = cls._normalise_relevance_text(cls._question_text_blob(question))
        matched_groups: list[set[str]] = []
        for aliases in cls._country_alias_terms.values():
            if any(
                re.search(rf"\b{re.escape(alias.lower())}\b", question_blob)
                for alias in aliases
            ):
                alias_terms = set()
                for alias in aliases:
                    alias_terms.update(cls._text_tokens(alias))
                matched_groups.append(alias_terms)
        return matched_groups

    @classmethod
    def _country_names_for_question(cls, question: MetaculusQuestion) -> list[str]:
        question_blob = cls._normalise_relevance_text(cls._question_text_blob(question))
        return [
            country
            for country, aliases in cls._country_alias_terms.items()
            if any(
                re.search(rf"\b{re.escape(alias.lower())}\b", question_blob)
                for alias in aliases
            )
        ]

    @classmethod
    def _is_election_question(cls, question: MetaculusQuestion) -> bool:
        return bool(cls._text_tokens(cls._question_text_blob(question)) & cls._election_market_terms)

    @classmethod
    def _official_data_topics(cls, question: MetaculusQuestion) -> list[str]:
        question_blob = cls._question_text_blob(question)
        return [
            topic
            for topic, keywords in cls._official_data_topic_keywords.items()
            if any(keyword in question_blob for keyword in keywords)
        ]

    @classmethod
    def _should_run_high_value_deep_research(
        cls, question: MetaculusQuestion
    ) -> bool:
        if cls._env_flag("DEEP_RESEARCH_ON_ALL_QUESTIONS", False):
            return True
        if not cls._env_flag("ENABLE_DEEP_RESEARCH_ON_HIGH_VALUE", False):
            return False
        question_blob = cls._question_text_blob(question)
        configured_keywords = os.getenv(
            "HIGH_VALUE_RESEARCH_KEYWORDS",
            cls._default_high_value_deep_research_keywords,
        )
        keywords = [
            keyword.strip().lower()
            for keyword in configured_keywords.split(",")
            if keyword.strip()
        ]
        return any(keyword in question_blob for keyword in keywords)

    @classmethod
    def _deep_research_llm(cls) -> GeneralLlm | None:
        override_model = os.getenv("DEEP_RESEARCH_MODEL", "").strip()
        if override_model:
            return GeneralLlm(
                model=override_model,
                temperature=0.1,
                timeout=360,
                allowed_tries=2,
                max_tokens=_env_int("DEEP_RESEARCH_MAX_TOKENS", 4096),
            )

        provider = os.getenv("DEEP_RESEARCH_PROVIDER", "auto").strip().lower()
        if provider in {"auto", "perplexity", "sonar"}:
            if os.getenv("PERPLEXITY_API_KEY"):
                return GeneralLlm(
                    model=os.getenv("PERPLEXITY_DEEP_RESEARCH_MODEL", "perplexity/sonar-pro"),
                    temperature=0.1,
                    timeout=360,
                    allowed_tries=2,
                    max_tokens=_env_int("DEEP_RESEARCH_MAX_TOKENS", 4096),
                    web_search_options={"search_context_size": "high"},
                    reasoning_effort="high",
                )
            if os.getenv("OPENROUTER_API_KEY"):
                return GeneralLlm(
                    model=os.getenv(
                        "OPENROUTER_PERPLEXITY_DEEP_RESEARCH_MODEL",
                        "openrouter/perplexity/sonar-pro",
                    ),
                    temperature=0.1,
                    timeout=360,
                    allowed_tries=2,
                    max_tokens=_env_int("DEEP_RESEARCH_MAX_TOKENS", 4096),
                    web_search_options={"search_context_size": "high"},
                    reasoning_effort="high",
                )

        if provider in {"auto", "exa"} and os.getenv("EXA_API_KEY"):
            return GeneralLlm(
                model=os.getenv("EXA_DEEP_RESEARCH_MODEL", "exa/exa"),
                temperature=0.1,
                timeout=360,
                allowed_tries=2,
                max_tokens=_env_int("DEEP_RESEARCH_MAX_TOKENS", 4096),
            )

        return None

    @classmethod
    def _perplexity_research_llm(cls, researcher: str) -> GeneralLlm | None:
        requested_model = researcher.strip()
        if requested_model in {"perplexity", "sonar", "perplexity/auto"}:
            if os.getenv("PERPLEXITY_API_KEY"):
                model = os.getenv("PERPLEXITY_RESEARCHER_MODEL", "perplexity/sonar-pro")
            elif os.getenv("OPENROUTER_API_KEY"):
                model = os.getenv(
                    "OPENROUTER_PERPLEXITY_RESEARCHER_MODEL",
                    "openrouter/perplexity/sonar-pro",
                )
            else:
                logger.warning(
                    "Perplexity researcher configured but neither PERPLEXITY_API_KEY nor OPENROUTER_API_KEY is set."
                )
                return None
        else:
            model = requested_model
            if model.startswith("perplexity/") and not os.getenv("PERPLEXITY_API_KEY"):
                logger.warning(
                    "Perplexity researcher model %s requires PERPLEXITY_API_KEY; skipping configured research.",
                    model,
                )
                return None
            if model.startswith("openrouter/perplexity/") and not os.getenv("OPENROUTER_API_KEY"):
                logger.warning(
                    "OpenRouter Perplexity researcher model %s requires OPENROUTER_API_KEY; skipping configured research.",
                    model,
                )
                return None

        return GeneralLlm(
            model=model,
            temperature=0.1,
            timeout=_env_int("RESEARCHER_TIMEOUT_SECONDS", 240),
            allowed_tries=2,
            max_tokens=_env_int("RESEARCHER_MAX_TOKENS", 4096),
            web_search_options={"search_context_size": "high"},
            reasoning_effort="high",
        )

    @classmethod
    def _exa_research_llm(cls, researcher: str) -> GeneralLlm | None:
        requested_model = researcher.strip()
        if requested_model in {"exa", "exa/auto"}:
            model = os.getenv("EXA_RESEARCHER_MODEL", "exa/exa")
        else:
            model = requested_model
        if not os.getenv("EXA_API_KEY"):
            logger.warning(
                "Exa researcher model %s requires EXA_API_KEY; skipping configured research.",
                model,
            )
            return None
        return GeneralLlm(
            model=model,
            temperature=0.1,
            timeout=_env_int("RESEARCHER_TIMEOUT_SECONDS", 240),
            allowed_tries=2,
            max_tokens=_env_int("RESEARCHER_MAX_TOKENS", 4096),
        )

    @classmethod
    def _random_researcher_candidates(cls) -> list[str]:
        raw_candidates = os.getenv(
            "RESEARCHER_RANDOM_MODELS",
            "asknews/news-summaries,perplexity,exa",
        )
        return [
            candidate.strip()
            for candidate in raw_candidates.split(",")
            if candidate.strip()
        ]

    @classmethod
    def _researcher_candidate_is_available(cls, researcher: str) -> bool:
        if researcher.startswith("asknews/"):
            return bool(os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
        if (
            researcher in {"perplexity", "sonar", "perplexity/auto"}
            or researcher.startswith("perplexity/")
        ):
            return bool(os.getenv("PERPLEXITY_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
        if researcher.startswith("openrouter/perplexity/"):
            return bool(os.getenv("OPENROUTER_API_KEY"))
        if researcher in {"exa", "exa/auto"} or researcher.startswith("exa/"):
            return bool(os.getenv("EXA_API_KEY"))
        return True

    @classmethod
    def _available_researcher_candidates(cls) -> list[str]:
        return [
            candidate
            for candidate in cls._random_researcher_candidates()
            if cls._researcher_candidate_is_available(candidate)
        ]

    @classmethod
    def _select_random_researcher(cls, question: MetaculusQuestion) -> str:
        candidates = cls._random_researcher_candidates()
        available_candidates = cls._available_researcher_candidates()
        if not available_candidates:
            logger.warning(
                "No configured random researcher candidates have the needed API keys; using full candidate list."
            )
            available_candidates = candidates or ["perplexity"]
        question_key = (
            str(getattr(question, "id", "") or getattr(question, "id_of_post", ""))
            or getattr(question, "page_url", "")
            or getattr(question, "question_text", "")
        )
        seed = os.getenv("RESEARCHER_RANDOM_SEED", "").strip()
        digest = hashlib.sha256(f"{seed}:{question_key}".encode("utf-8")).hexdigest()
        return available_candidates[int(digest[:12], 16) % len(available_candidates)]

    async def _run_fallback_researcher(
        self,
        prompt: str,
        question: MetaculusQuestion,
        reason: str,
    ) -> str:
        fallback = os.getenv("RESEARCHER_FALLBACK_MODEL", "exa").strip()
        if not fallback or fallback in {"None", "no_research"}:
            return ""
        logger.warning(
            "Falling back from configured researcher to %s for %s: %s",
            fallback,
            question.page_url,
            reason,
        )
        return await self._run_configured_researcher(
            fallback, prompt, question, allow_fallback=False
        )

    @classmethod
    def _cache_ttl_seconds(cls) -> int:
        raw_value = os.getenv("DIRECT_EVIDENCE_CACHE_HOURS", "6")
        try:
            return max(0, int(float(raw_value) * 3600))
        except ValueError:
            return 6 * 3600

    @classmethod
    def _provider_cache_key(
        cls, provider: str, question: MetaculusQuestion, extra: str = ""
    ) -> str:
        question_id = getattr(question, "id", None) or getattr(
            question, "id_of_post", ""
        )
        key_payload = json.dumps(
            {
                "provider": provider,
                "question_id": question_id,
                "question": getattr(question, "question_text", ""),
                "extra": extra,
            },
            sort_keys=True,
        )
        return hashlib.sha256(key_payload.encode("utf-8")).hexdigest()

    @classmethod
    def _read_cached_evidence(cls, cache_key: str) -> list[EvidenceItem] | None:
        if not cls._env_flag("ENABLE_DIRECT_EVIDENCE_CACHE", True):
            return None
        ttl_seconds = cls._cache_ttl_seconds()
        if ttl_seconds <= 0:
            return None
        cache_path = cls._research_cache_dir / f"{cache_key}.json"
        if not cache_path.exists():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if time.time() - payload.get("created_at", 0) > ttl_seconds:
                return None
            return [EvidenceItem(**item) for item in payload.get("items", [])]
        except Exception as error:
            logger.warning("Failed to read direct-evidence cache %s: %r", cache_key, error)
            return None

    @classmethod
    def _write_cached_evidence(
        cls, cache_key: str, evidence_items: list[EvidenceItem]
    ) -> None:
        if not cls._env_flag("ENABLE_DIRECT_EVIDENCE_CACHE", True):
            return
        try:
            cls._research_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cls._research_cache_dir / f"{cache_key}.json"
            cache_payload = {
                "created_at": time.time(),
                "items": [asdict(item) for item in evidence_items],
            }
            cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
        except Exception as error:
            logger.warning("Failed to write direct-evidence cache %s: %r", cache_key, error)

    @classmethod
    async def _cached_evidence_fetch(
        cls,
        provider: str,
        question: MetaculusQuestion,
        fetcher: Any,
        extra: str = "",
    ) -> list[EvidenceItem]:
        cache_key = cls._provider_cache_key(provider, question, extra)
        cached = cls._read_cached_evidence(cache_key)
        if cached is not None:
            return cached
        evidence_items = await fetcher()
        cls._write_cached_evidence(cache_key, evidence_items)
        return evidence_items

    @staticmethod
    def _safe_json_get(url: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        response = requests.get(url, timeout=kwargs.pop("timeout", 20), **kwargs)
        response.raise_for_status()
        return response.json()

    @classmethod
    async def _http_get_json(
        cls,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: int = 20,
    ) -> dict[str, Any] | list[Any]:
        return await asyncio.to_thread(
            cls._safe_json_get,
            url,
            params=params,
            headers=headers,
            timeout=timeout,
        )

    @classmethod
    def _question_terms(cls, question: MetaculusQuestion) -> set[str]:
        return {
            word.lower()
            for word in cls._text_tokens(cls._question_text_blob(question))
            if len(word) > 3 and word.lower() not in cls._stop_words
        }

    @classmethod
    def _relevance_score(cls, candidate_text: str, question: MetaculusQuestion) -> float:
        terms = cls._question_terms(question)
        if not terms:
            return 0.0
        candidate_words = cls._text_tokens(candidate_text)
        matches = terms & candidate_words
        return len(matches) / max(len(terms), 1)

    @classmethod
    def _market_relevance_directness(
        cls,
        candidate_text: str,
        question: MetaculusQuestion,
        score: float,
    ) -> str | None:
        candidate_words = cls._text_tokens(candidate_text)
        if not candidate_words:
            return None

        option_match = cls._has_option_match(candidate_words, question)
        country_match = cls._has_country_match(candidate_words, question)
        election_term_match = bool(candidate_words & cls._election_market_terms)

        if cls._is_election_question(question):
            if not option_match and not (country_match and election_term_match):
                return None
            if option_match and (country_match or election_term_match):
                return "direct"
            if score >= 0.12:
                return "direct" if country_match and election_term_match else "similar"
            return "similar"

        if score >= 0.18:
            return "direct"
        if score >= 0.10:
            return "similar"
        return None

    @classmethod
    def _has_option_match(
        cls, candidate_words: set[str], question: MetaculusQuestion
    ) -> bool:
        for option_terms in cls._question_option_terms(question):
            required_matches = min(2, len(option_terms))
            if len(option_terms & candidate_words) >= required_matches:
                return True
        return False

    @classmethod
    def _has_country_match(
        cls, candidate_words: set[str], question: MetaculusQuestion
    ) -> bool:
        return any(
            bool(alias_group & candidate_words)
            for alias_group in cls._country_alias_groups_for_question(question)
        )

    @classmethod
    def _directness_from_score(cls, score: float) -> str:
        if score >= 0.22:
            return "direct"
        if score >= 0.10:
            return "similar"
        return "weak"

    @staticmethod
    def _normalise_market_probability(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            if isinstance(value, str):
                value = value.strip().replace("$", "").replace("%", "")
            probability = float(value)
        except (TypeError, ValueError):
            return None
        if probability > 1:
            probability /= 100
        if probability < 0 or probability > 1:
            return None
        return probability

    @staticmethod
    def _parse_jsonish_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _format_probability(probability: float | None) -> str:
        if probability is None:
            return "n/a"
        return f"{probability:.1%}"

    @classmethod
    def _format_evidence_items(cls, evidence_items: list[EvidenceItem]) -> str:
        if not evidence_items:
            return ""
        lines: list[str] = []
        for item in evidence_items:
            probability = (
                f" | implied probability {cls._format_probability(item.probability)}"
                if item.probability is not None
                else ""
            )
            value = f" | value {item.value} {item.unit}".rstrip() if item.value else ""
            date = f" | date {item.date}" if item.date else ""
            url = f" | {item.url}" if item.url else ""
            caveats = f" Caveats: {item.caveats}" if item.caveats else ""
            lines.append(
                f"- [{item.provider} / {item.directness}] {item.title}{probability}{value}{date}{url}. "
                f"{item.summary}{caveats}"
            )
        return "\n".join(lines)

    @staticmethod
    def _clamp_probability(probability: float) -> float:
        return max(0.01, min(0.99, probability))

    @classmethod
    def _logit_mean_probability(cls, probabilities: list[float]) -> float:
        logits = [
            math.log(cls._clamp_probability(prob) / (1 - cls._clamp_probability(prob)))
            for prob in probabilities
        ]
        mean_logit = statistics.mean(logits)
        return cls._clamp_probability(1 / (1 + math.exp(-mean_logit)))

    @classmethod
    def _should_escalate_binary(
        cls, predictions: list[ReasonedPrediction[float]]
    ) -> bool:
        if len(predictions) < 2:
            return True
        probabilities = [prediction.prediction_value for prediction in predictions]
        return max(probabilities) - min(probabilities) >= cls._binary_escalation_spread

    @classmethod
    def _should_escalate_multiple_choice(
        cls, predictions: list[ReasonedPrediction[PredictedOptionList]]
    ) -> bool:
        if len(predictions) < 2:
            return True
        option_names = predictions[0].prediction_value.to_dict().keys()
        for option_name in option_names:
            probabilities = [
                prediction.prediction_value.to_dict()[option_name]
                for prediction in predictions
            ]
            if (
                max(probabilities) - min(probabilities)
                >= cls._multiple_choice_escalation_spread
            ):
                return True
        return False

    @staticmethod
    def _value_at_percentile(
        distribution: NumericDistribution, target_percentile: float
    ) -> float:
        percentiles = sorted(
            distribution.declared_percentiles, key=lambda item: item.percentile
        )
        for percentile in percentiles:
            if math.isclose(percentile.percentile, target_percentile, abs_tol=1e-6):
                return percentile.value
        for lower, upper in zip(percentiles, percentiles[1:]):
            if lower.percentile <= target_percentile <= upper.percentile:
                span = upper.percentile - lower.percentile
                if span <= 0:
                    return lower.value
                weight = (target_percentile - lower.percentile) / span
                return lower.value + weight * (upper.value - lower.value)
        return percentiles[min(range(len(percentiles)), key=lambda i: abs(percentiles[i].percentile - target_percentile))].value

    @classmethod
    def _should_escalate_numeric(
        cls,
        predictions: list[ReasonedPrediction[NumericDistribution]],
        question: NumericQuestion | DateQuestion,
    ) -> bool:
        if len(predictions) < 2:
            return True
        medians = [
            cls._value_at_percentile(prediction.prediction_value, 0.5)
            for prediction in predictions
        ]
        spread = max(medians) - min(medians)
        center = abs(statistics.median(medians))
        if isinstance(question, DateQuestion):
            range_width = abs(
                question.upper_bound.timestamp() - question.lower_bound.timestamp()
            )
        else:
            range_width = abs(question.upper_bound - question.lower_bound)
        range_disagreement = (
            range_width > 0
            and spread / range_width >= cls._numeric_escalation_range_fraction
        )
        relative_disagreement = (
            center > 1 and spread / center >= cls._numeric_escalation_relative_fraction
        )
        return range_disagreement or relative_disagreement

    @staticmethod
    def _build_escalation_prompt(
        prompt: str,
        predictions: list[ReasonedPrediction[T]],
        targeted_research: str = "",
    ) -> str:
        model_outputs = "\n\n".join(
            f"## Cheap role output {i + 1}\nPrediction: {prediction.prediction_value}\n\nReasoning:\n{prediction.reasoning}"
            for i, prediction in enumerate(predictions)
        )
        targeted_research_section = (
            clean_indents(
                f"""
                Additional targeted research gathered after the cheap ensemble disagreed:
                {targeted_research}
                """
            )
            if targeted_research.strip()
            else ""
        )
        return clean_indents(
            f"""
            {prompt}

            ---

            Escalation context:
            The cheaper first-pass role ensemble either disagreed materially or had too few successful members.
            Critically review their forecasts below. Do not average blindly; decide which role found the strongest
            assumptions, which evidence is most resolution-relevant, and whether any role appears to have answered
            the wrong question.

            {targeted_research_section}

            {model_outputs}

            Now produce your own final forecast in the exact required format from the original instructions.
            """
        )

    async def _build_targeted_escalation_prompt(
        self,
        question: MetaculusQuestion,
        prompt: str,
        predictions: list[ReasonedPrediction[T]],
    ) -> str:
        targeted_research = await self._run_disagreement_targeted_research(
            question, predictions
        )
        return self._build_escalation_prompt(prompt, predictions, targeted_research)

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            researcher = self.get_llm("researcher")

            prompt = self._build_structured_research_prompt(question)

            research_tasks = [
                (
                    "Configured research",
                    self._run_configured_researcher(researcher, prompt, question),
                )
            ]
            if self._env_flag("ENABLE_SUPPLEMENTAL_WEB_RESEARCH", True):
                research_tasks.append(
                    (
                        "Supplemental web/source research",
                        self._run_supplemental_web_research(question),
                    )
                )
            if self._env_flag("ENABLE_DIRECT_STRUCTURED_RESEARCH", True):
                research_tasks.append(
                    (
                        "Direct API / structured evidence",
                        self._run_direct_structured_research(question),
                    )
                )
            if self._env_flag("ENABLE_MARKET_PRIOR_RESEARCH", True):
                research_tasks.append(
                    (
                        "Market-prior collector",
                        self._run_market_prior_research(question),
                    )
                )
            official_topics = self._official_data_topics(question)
            if (
                self._env_flag("ENABLE_OFFICIAL_DATA_RESEARCH", True)
                and official_topics
            ):
                research_tasks.append(
                    (
                        "Official data router",
                        self._run_official_data_research(question, official_topics),
                    )
                )
            if self._should_run_high_value_deep_research(question):
                research_tasks.append(
                    (
                        "Optional deep research",
                        self._run_optional_deep_research(
                            question,
                            focus="High-value question research pass requested by configuration.",
                            reason="high-value",
                        ),
                    )
                )
            results = await asyncio.gather(
                *(task for _, task in research_tasks), return_exceptions=True
            )
            research_sections: list[str] = []
            for (section_name, _), result in zip(research_tasks, results):
                if isinstance(result, BaseException):
                    logger.warning(
                        "Research section %s failed for URL %s: %r",
                        section_name,
                        question.page_url,
                        result,
                    )
                elif result.strip():
                    research_sections.append(f"## {section_name}\n{result}")

            research = "\n\n".join(research_sections)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    @staticmethod
    def _build_structured_research_prompt(question: MetaculusQuestion) -> str:
        return clean_indents(
            f"""
            You are a research assistant supporting a forecasting bot. Gather evidence only; do not make a final forecast.

            Produce a concise evidence memo with exactly these headings:
            ## Resolution check
            - Explain the exact event, threshold, dates, and source of truth.
            - Note whether the question already appears resolved under the criteria.

            ## Current state
            - Give the latest measured value, status, or public fact pattern.
            - Prefer official or primary sources when available.

            ## Base rates and reference classes
            - Give historical frequency, analogous cases, or a reasonable outside view.
            - If no good base rate exists, say what proxy you used.

            ## Trend and drivers
            - Summarize the main forces pushing the outcome up/down or earlier/later.

            ## Market and expert priors
            - Look for relevant prediction markets, forecasts, polls, analyst estimates, or consensus views.
            - Include Kalshi, Polymarket, Manifold, Metaculus/community forecasts, or similar markets when relevant and allowed.

            ## Counterevidence
            - Give the strongest reasons the obvious answer could be wrong.

            ## Cruxes to verify
            - List 2-4 factual cruxes that would most change a forecast.

            Include source names and URLs where available. Keep the memo compact and factual.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )

    @staticmethod
    def _prediction_timestamp(prediction: Any) -> datetime | None:
        timestamp = getattr(prediction, "timestamp", None) or getattr(
            prediction, "timestamp_start", None
        )
        return timestamp if isinstance(timestamp, datetime) else None

    @classmethod
    def _latest_previous_forecast(cls, question: MetaculusQuestion) -> Any | None:
        previous_forecasts = getattr(question, "previous_forecasts", None) or []
        if not previous_forecasts:
            return None
        return max(
            previous_forecasts,
            key=lambda prediction: cls._prediction_timestamp(prediction)
            or datetime.min.replace(tzinfo=timezone.utc),
        )

    @staticmethod
    def _readable_prediction(prediction: Any | None) -> str:
        if prediction is None:
            return "No previous forecast found."
        try:
            from forecasting_tools.data_models.data_organizer import DataOrganizer

            return DataOrganizer.get_readable_prediction(prediction)  # type: ignore[arg-type]
        except Exception:
            if hasattr(prediction, "model_dump"):
                return json.dumps(prediction.model_dump(mode="json"), ensure_ascii=True)
            return str(prediction)

    @staticmethod
    def _coerce_probability(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            if isinstance(value, str):
                value = value.strip().replace("%", "")
            probability = float(value)
        except (TypeError, ValueError):
            return None
        if probability > 1:
            probability /= 100
        if not 0 <= probability <= 1:
            return None
        return probability

    @classmethod
    def _binary_probability_from_prediction(cls, prediction: Any | None) -> float | None:
        if prediction is None:
            return None
        if isinstance(prediction, (float, int)):
            return cls._coerce_probability(prediction)
        return cls._coerce_probability(getattr(prediction, "prediction_in_decimal", None))

    @classmethod
    def _multiple_choice_probabilities_from_prediction(
        cls, prediction: Any | None
    ) -> dict[str, float] | None:
        if prediction is None:
            return None
        if hasattr(prediction, "to_dict"):
            try:
                return {
                    str(option): float(probability)
                    for option, probability in prediction.to_dict().items()
                }
            except Exception:
                pass
        predicted_options = getattr(prediction, "predicted_options", None)
        if not predicted_options:
            return None
        probabilities: dict[str, float] = {}
        for option in predicted_options:
            name = getattr(option, "option_name", None)
            probability = cls._coerce_probability(getattr(option, "probability", None))
            if name is not None and probability is not None:
                probabilities[str(name)] = probability
        return probabilities or None

    @staticmethod
    def _normalise_confidence(value: Any) -> str:
        confidence = str(value or "").strip().lower()
        if confidence in {"low", "medium", "high"}:
            return confidence
        return "low"

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _extract_json_object(cls, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        fenced_match = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fenced_match:
            cleaned = fenced_match.group(1)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Scout response JSON was not an object.")
        return parsed

    @classmethod
    def _scout_decision_from_response(
        cls,
        question: MetaculusQuestion,
        previous_forecast: Any | None,
        raw_response: str,
    ) -> RefreshScoutDecision:
        previous_timestamp = cls._prediction_timestamp(previous_forecast)
        try:
            payload = cls._extract_json_object(raw_response)
            estimated_probabilities = payload.get("estimated_new_probabilities")
            if isinstance(estimated_probabilities, dict):
                estimated_probabilities = {
                    str(option): probability
                    for option, value in estimated_probabilities.items()
                    if (probability := cls._coerce_probability(value)) is not None
                }
            else:
                estimated_probabilities = None
            return RefreshScoutDecision(
                question_id=getattr(question, "id_of_question", None),
                post_id=getattr(question, "id_of_post", None),
                page_url=getattr(question, "page_url", None),
                question_title=getattr(question, "question_text", ""),
                question_type=question.get_api_type_name(),
                previous_forecast=cls._readable_prediction(previous_forecast),
                previous_forecast_timestamp=(
                    previous_timestamp.isoformat() if previous_timestamp else None
                ),
                should_reforecast=cls._coerce_bool(
                    payload.get("should_reforecast")
                ),
                recommended_action=str(
                    payload.get("recommended_action", "skip")
                ).strip().lower(),
                confidence_in_movement=cls._normalise_confidence(
                    payload.get("confidence_in_movement")
                ),
                movement_reason=str(payload.get("movement_reason", "")).strip(),
                fresh_evidence_summary=str(
                    payload.get("fresh_evidence_summary", "")
                ).strip(),
                estimated_new_forecast=cls._coerce_probability(
                    payload.get("estimated_new_forecast")
                ),
                estimated_new_probabilities=estimated_probabilities,
                material_distribution_shift=cls._coerce_bool(
                    payload.get("material_distribution_shift")
                ),
                fresh_high_signal_evidence=cls._coerce_bool(
                    payload.get("fresh_high_signal_evidence")
                ),
                prior_reasoning_stale=cls._coerce_bool(
                    payload.get("prior_reasoning_stale")
                ),
                raw_response=raw_response,
            )
        except Exception as error:
            logger.warning(
                "Could not parse scout response for URL %s: %r\n%s",
                question.page_url,
                error,
                raw_response,
            )
            return RefreshScoutDecision(
                question_id=getattr(question, "id_of_question", None),
                post_id=getattr(question, "id_of_post", None),
                page_url=getattr(question, "page_url", None),
                question_title=getattr(question, "question_text", ""),
                question_type=question.get_api_type_name(),
                previous_forecast=cls._readable_prediction(previous_forecast),
                previous_forecast_timestamp=(
                    previous_timestamp.isoformat() if previous_timestamp else None
                ),
                should_reforecast=False,
                recommended_action="skip",
                confidence_in_movement="low",
                movement_reason="Scout response could not be parsed.",
                fresh_evidence_summary="",
                raw_response=raw_response,
                parse_error=repr(error),
            )

    def _scout_llm(self) -> GeneralLlm:
        return GeneralLlm(
            model=os.getenv("SCOUT_MODEL", "openrouter/z-ai/glm-5.1"),
            temperature=0.1,
            timeout=_env_int("SCOUT_TIMEOUT_SECONDS", 180),
            allowed_tries=2,
            max_tokens=_env_int("SCOUT_MAX_TOKENS", 1500),
        )

    async def _run_scout_research(self, question: MetaculusQuestion) -> str:
        researcher = os.getenv("SCOUT_RESEARCHER_MODEL", "perplexity").strip()
        if not researcher or researcher in {"None", "no_research"}:
            return ""
        prompt = clean_indents(
            f"""
            You are doing a cheap update check for an existing Metaculus forecast.
            Search only for fresh, resolution-relevant information since the bot's
            previous forecast. Do not make a final forecast.

            Return a compact memo with:
            - New direct evidence, if any.
            - Official/source-of-truth changes, if any.
            - Market, poll, benchmark, company, legal, or government updates, if directly relevant.
            - Whether the prior evidence base appears stale.
            - If nothing important changed, say so clearly.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        try:
            research = await self._run_configured_researcher(
                researcher, prompt, question, allow_fallback=False
            )
        except Exception as error:
            logger.warning(
                "Scout research failed for URL %s with %s: %r",
                question.page_url,
                researcher,
                error,
            )
            return ""
        return self._truncate_for_prompt(research, _env_int("SCOUT_RESEARCH_MAX_CHARS", 5000))

    def _build_scout_prompt(
        self,
        question: MetaculusQuestion,
        previous_forecast: Any | None,
        scout_research: str,
    ) -> str:
        previous_timestamp = self._prediction_timestamp(previous_forecast)
        options_text = (
            f"Options: {question.options}"
            if isinstance(question, MultipleChoiceQuestion)
            else ""
        )
        bounds_text = ""
        if isinstance(question, (NumericQuestion, DateQuestion)):
            bounds_text = clean_indents(
                f"""
                Lower bound: {question.lower_bound}
                Upper bound: {question.upper_bound}
                Open lower bound: {question.open_lower_bound}
                Open upper bound: {question.open_upper_bound}
                Unit: {question.unit_of_measure}
                """
            )
        return clean_indents(
            f"""
            You are a cheap scout for an autonomous Metaculus forecasting bot.
            Your job is to decide whether it is worth spending more money on a full
            reforecast. Do not optimize for looking active; recommend a full
            reforecast only when the full system is likely to materially change.

            Current UTC time: {datetime.now(timezone.utc).isoformat()}
            Question type: {question.get_api_type_name()}
            Page URL: {question.page_url}
            Close time: {question.close_time}
            Scheduled resolution time: {question.scheduled_resolution_time}

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}

            {options_text}
            {bounds_text}

            Bot's previous forecast:
            {self._readable_prediction(previous_forecast)}

            Previous forecast timestamp:
            {previous_timestamp.isoformat() if previous_timestamp else "unknown"}

            Fresh update research:
            {scout_research or "No fresh research was available."}

            Return exactly one JSON object and no other text.
            Schema:
            {{
              "should_reforecast": true,
              "estimated_new_forecast": 0.42,
              "estimated_new_probabilities": {{"Option A": 0.5, "Option B": 0.5}},
              "material_distribution_shift": false,
              "fresh_high_signal_evidence": false,
              "prior_reasoning_stale": false,
              "movement_reason": "new evidence / closing soon / stale reasoning / other",
              "confidence_in_movement": "low|medium|high",
              "fresh_evidence_summary": "brief summary",
              "recommended_action": "skip|full_reforecast"
            }}

            Output rules:
            - For binary questions, set estimated_new_forecast to a decimal probability from 0 to 1.
            - For multiple-choice questions, set estimated_new_probabilities using the exact option names.
            - For numeric or date questions, set material_distribution_shift=true only if the median or central interval would materially move.
            - Set fresh_high_signal_evidence=true only for direct evidence that maps to the resolution criteria.
            - Use "high" confidence only when the evidence is direct and you expect a meaningful forecast move.
            - Use recommended_action="full_reforecast" only if the expensive full system is likely worthwhile.
            """
        )

    async def run_refresh_scout(
        self, question: MetaculusQuestion
    ) -> RefreshScoutDecision:
        previous_forecast = self._latest_previous_forecast(question)
        scout_research = await self._run_scout_research(question)
        prompt = self._build_scout_prompt(question, previous_forecast, scout_research)
        raw_response = await self._scout_llm().invoke(prompt)
        decision = self._scout_decision_from_response(
            question, previous_forecast, raw_response
        )
        decision.gate_reasons = self._refresh_gate_reasons(question, decision)
        decision.gate_triggered = bool(decision.gate_reasons)
        logger.info(
            "Refresh scout for URL %s: gate=%s reasons=%s decision=%s",
            question.page_url,
            decision.gate_triggered,
            decision.gate_reasons,
            decision.recommended_action,
        )
        return decision

    @staticmethod
    def _hours_until_close(question: MetaculusQuestion) -> float | None:
        close_time = getattr(question, "close_time", None)
        if close_time is None:
            return None
        if close_time.tzinfo is None:
            close_time = close_time.replace(tzinfo=timezone.utc)
        return (close_time - datetime.now(timezone.utc)).total_seconds() / 3600

    @classmethod
    def _forecast_age_days(cls, question: MetaculusQuestion) -> float | None:
        previous_forecast = cls._latest_previous_forecast(question)
        timestamp = cls._prediction_timestamp(previous_forecast)
        if timestamp is None:
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400

    @classmethod
    def _binary_log_odds_move(cls, old_probability: float, new_probability: float) -> float:
        old_probability = cls._clamp_probability(old_probability)
        new_probability = cls._clamp_probability(new_probability)
        return abs(
            math.log(new_probability / (1 - new_probability))
            - math.log(old_probability / (1 - old_probability))
        )

    @classmethod
    def _refresh_gate_reasons(
        cls, question: MetaculusQuestion, decision: RefreshScoutDecision
    ) -> list[str]:
        reasons: list[str] = []
        confidence = decision.confidence_in_movement
        hours_until_close = cls._hours_until_close(question)
        if (
            hours_until_close is not None
            and 0 <= hours_until_close <= _env_float("SCOUT_CLOSING_SOON_HOURS", 48)
        ):
            reasons.append(f"question closes soon ({hours_until_close:.1f}h)")

        forecast_age_days = cls._forecast_age_days(question)
        if (
            forecast_age_days is not None
            and forecast_age_days >= _env_float("SCOUT_STALE_DAYS", 7)
        ):
            reasons.append(f"forecast is stale ({forecast_age_days:.1f}d old)")

        previous_forecast = cls._latest_previous_forecast(question)
        if isinstance(question, BinaryQuestion):
            old_probability = cls._binary_probability_from_prediction(previous_forecast)
            new_probability = decision.estimated_new_forecast
            if old_probability is not None and new_probability is not None:
                absolute_move = abs(new_probability - old_probability)
                if absolute_move >= _env_float("SCOUT_BINARY_ABS_MOVE", 0.08):
                    reasons.append(f"binary probability moved {absolute_move:.1%}")
                log_odds_move = cls._binary_log_odds_move(
                    old_probability, new_probability
                )
                if log_odds_move >= _env_float("SCOUT_BINARY_LOG_ODDS_MOVE", 0.35):
                    reasons.append(f"binary log-odds moved {log_odds_move:.2f}")
        elif isinstance(question, MultipleChoiceQuestion):
            old_probabilities = cls._multiple_choice_probabilities_from_prediction(
                previous_forecast
            )
            new_probabilities = decision.estimated_new_probabilities
            if old_probabilities and new_probabilities:
                shared_options = set(old_probabilities) & set(new_probabilities)
                if shared_options:
                    max_move = max(
                        abs(new_probabilities[option] - old_probabilities[option])
                        for option in shared_options
                    )
                    if max_move >= _env_float("SCOUT_MC_MAX_OPTION_MOVE", 0.08):
                        reasons.append(
                            f"multiple-choice option probability moved {max_move:.1%}"
                        )
                    old_top = max(old_probabilities, key=old_probabilities.get)
                    new_top = max(new_probabilities, key=new_probabilities.get)
                    if old_top != new_top:
                        reasons.append(
                            f"multiple-choice top option changed ({old_top} -> {new_top})"
                        )
        elif decision.material_distribution_shift and confidence != "low":
            reasons.append("scout expects material distribution shift")

        if (
            decision.recommended_action == "full_reforecast"
            and decision.should_reforecast
            and confidence == "high"
        ):
            reasons.append("scout high-confidence full-reforecast recommendation")
        if decision.fresh_high_signal_evidence and confidence != "low":
            reasons.append("fresh high-signal evidence")
        if decision.prior_reasoning_stale and confidence != "low":
            reasons.append("scout says prior reasoning is stale")
        return reasons

    async def _run_configured_researcher(
        self,
        researcher: str | GeneralLlm,
        prompt: str,
        question: MetaculusQuestion,
        allow_fallback: bool = True,
    ) -> str:
        if isinstance(researcher, GeneralLlm):
            return await researcher.invoke(prompt)
        researcher = researcher.strip()
        if researcher in {"all", "all-available", "all-researchers"}:
            selected_researchers = self._available_researcher_candidates()
            if not selected_researchers:
                logger.warning(
                    "No all-researcher candidates have the needed API keys for %s.",
                    question.page_url,
                )
                if allow_fallback:
                    return await self._run_fallback_researcher(
                        prompt, question, "No all-researcher candidates are available"
                    )
                return ""

            logger.info(
                "Running all configured researchers for %s: %s",
                question.page_url,
                selected_researchers,
            )

            async def run_one(candidate: str) -> str:
                try:
                    result = await self._run_configured_researcher(
                        candidate, prompt, question, allow_fallback=False
                    )
                except Exception as exc:
                    logger.warning(
                        "Configured researcher %s failed for %s: %r",
                        candidate,
                        question.page_url,
                        exc,
                    )
                    return ""
                if not result.strip():
                    return ""
                return f"### {candidate}\n{result}"

            results = await asyncio.gather(
                *(run_one(candidate) for candidate in selected_researchers)
            )
            sections = [result for result in results if result.strip()]
            if sections:
                return "\n\n".join(
                    [
                        "Ran all available configured researchers: "
                        + ", ".join(selected_researchers),
                        *sections,
                    ]
                )
            if allow_fallback:
                return await self._run_fallback_researcher(
                    prompt, question, "All configured researchers returned no content"
                )
            return ""
        if researcher in {"random", "random-researcher", "researcher-random"}:
            selected_researcher = self._select_random_researcher(question)
            logger.info(
                "Random researcher selected %s for %s",
                selected_researcher,
                question.page_url,
            )
            result = await self._run_configured_researcher(
                selected_researcher, prompt, question, allow_fallback=allow_fallback
            )
            if result.strip():
                return f"Selected random researcher: {selected_researcher}\n\n{result}"
            return result
        elif (
            researcher == "asknews/news-summaries"
            or researcher == "asknews/deep-research/low-depth"
            or researcher == "asknews/deep-research/medium-depth"
            or researcher == "asknews/deep-research/high-depth"
        ):
            if not os.getenv("ASKNEWS_CLIENT_ID") or not os.getenv("ASKNEWS_SECRET"):
                logger.warning(
                    "AskNews researcher configured but ASKNEWS_CLIENT_ID/ASKNEWS_SECRET are missing; skipping AskNews research for %s.",
                    question.page_url,
                )
                if allow_fallback:
                    return await self._run_fallback_researcher(
                        prompt, question, "AskNews credentials are missing"
                    )
                return ""
            asknews_query = (
                question.question_text
                if researcher == "asknews/news-summaries"
                else prompt
            )
            try:
                return await AskNewsSearcher().call_preconfigured_version(
                    researcher, asknews_query
                )
            except Exception as exc:
                if allow_fallback:
                    return await self._run_fallback_researcher(
                        prompt, question, f"AskNews failed with {exc!r}"
                    )
                raise
        elif (
            researcher in {"perplexity", "sonar", "perplexity/auto"}
            or researcher.startswith("perplexity/")
            or researcher.startswith("openrouter/perplexity/")
        ):
            perplexity_llm = self._perplexity_research_llm(researcher)
            if perplexity_llm is None:
                if allow_fallback:
                    return await self._run_fallback_researcher(
                        prompt, question, "Perplexity credentials are missing"
                    )
                return ""
            try:
                return await perplexity_llm.invoke(prompt)
            except Exception as exc:
                if allow_fallback:
                    return await self._run_fallback_researcher(
                        prompt, question, f"Perplexity failed with {exc!r}"
                    )
                raise
        elif researcher in {"exa", "exa/auto"} or researcher.startswith("exa/"):
            exa_llm = self._exa_research_llm(researcher)
            if exa_llm is None:
                return ""
            return await exa_llm.invoke(prompt)
        elif researcher.startswith("smart-searcher"):
            model_name = researcher.removeprefix("smart-searcher/")
            searcher = SmartSearcher(
                model=model_name,
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
                use_advanced_filters=False,
            )
            return await searcher.invoke(prompt)
        elif not researcher or researcher == "None" or researcher == "no_research":
            return ""
        else:
            return await self.get_llm("researcher", "llm").invoke(prompt)

    async def _run_supplemental_web_research(
        self, question: MetaculusQuestion
    ) -> str:
        if not os.getenv("OPENROUTER_API_KEY"):
            return ""
        supplemental_prompt = clean_indents(
            f"""
            Build a structured forecasting research memo. Search broadly, but only report evidence that matters for the exact resolution criteria.

            Required sections:
            ## Resolution check
            ## Current state
            ## Base rates and reference classes
            ## Trend and drivers
            ## Market and expert priors
            ## Counterevidence
            ## Cruxes to verify

            Prioritize official sources, primary data, prediction markets or expert forecasts when allowed by the tournament, and recent reporting.
            Include source names and URLs when available. If a section has no strong evidence, say so briefly. Do not make a final forecast.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        searcher = SmartSearcher(
            model="openrouter/mistralai/mistral-large-2512",
            temperature=0,
            num_searches_to_run=3,
            num_sites_per_search=8,
            use_advanced_filters=False,
        )
        return await searcher.invoke(supplemental_prompt)

    async def _run_direct_structured_research(
        self, question: MetaculusQuestion
    ) -> str:
        provider_tasks: list[tuple[str, asyncio.Task[list[EvidenceItem]]]] = []
        topics = self._official_data_topics(question)

        if self._env_flag("ENABLE_DIRECT_MARKET_APIS", True):
            provider_tasks.extend(
                [
                    (
                        "Kalshi",
                        asyncio.create_task(self._fetch_cached_kalshi_markets(question)),
                    ),
                    (
                        "Polymarket",
                        asyncio.create_task(
                            self._fetch_cached_polymarket_markets(question)
                        ),
                    ),
                    (
                        "Manifold",
                        asyncio.create_task(
                            self._fetch_cached_manifold_markets(question)
                        ),
                    ),
                ]
            )
        if "economic" in topics and os.getenv("FRED_API_KEY"):
            provider_tasks.append(
                ("FRED", asyncio.create_task(self._fetch_cached_fred_series(question)))
            )
        if "finance" in topics and self._env_flag("ENABLE_SEC_EDGAR_RESEARCH", True):
            provider_tasks.append(
                (
                    "SEC EDGAR",
                    asyncio.create_task(self._fetch_cached_sec_edgar(question)),
                )
            )
        if "polling" in topics and self._env_flag("ENABLE_FIVETHIRTYEIGHT_POLLS", True):
            provider_tasks.append(
                (
                    "FiveThirtyEight polling data",
                    asyncio.create_task(
                        self._fetch_cached_fivethirtyeight_polls(question)
                    ),
                )
            )
        if "geopolitical" in topics and self._env_flag("ENABLE_ACLED_RESEARCH", True):
            provider_tasks.append(
                ("ACLED", asyncio.create_task(self._fetch_cached_acled_events(question)))
            )
        if "ai" in topics:
            provider_tasks.append(
                (
                    "AI benchmark registry",
                    asyncio.create_task(self._fetch_ai_benchmark_registry(question)),
                )
            )

        if not provider_tasks:
            return ""

        results = await asyncio.gather(
            *(task for _, task in provider_tasks), return_exceptions=True
        )
        evidence_items: list[EvidenceItem] = []
        for (provider_name, _), result in zip(provider_tasks, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Direct provider %s failed for URL %s: %r",
                    provider_name,
                    question.page_url,
                    result,
                )
                continue
            evidence_items.extend(result)

        if not self._env_flag("INCLUDE_WEAK_MARKET_EVIDENCE", False):
            evidence_items = [
                item
                for item in evidence_items
                if item.provider not in self._market_providers
                or item.directness != "weak"
            ]

        evidence_items = sorted(
            evidence_items,
            key=lambda item: {"direct": 0, "similar": 1, "weak": 2}.get(
                item.directness, 3
            ),
        )
        evidence_items = evidence_items[: int(os.getenv("DIRECT_EVIDENCE_MAX_ITEMS", "20"))]
        if not evidence_items:
            return ""

        raw_evidence = self._format_evidence_items(evidence_items)
        if not self._env_flag("ENABLE_DIRECT_EVIDENCE_SYNTHESIS", True):
            return raw_evidence
        synthesis = await self._synthesize_direct_evidence(question, evidence_items)
        return clean_indents(
            f"""
            ## Raw structured evidence
            {raw_evidence}

            ## Structured evidence synthesis
            {synthesis}
            """
        )

    async def _synthesize_direct_evidence(
        self, question: MetaculusQuestion, evidence_items: list[EvidenceItem]
    ) -> str:
        prompt = clean_indents(
            f"""
            Synthesize these structured evidence items for a forecaster.
            Do not produce a final forecast.

            Explain:
            - Which items are direct evidence for the exact resolution criteria.
            - Which items are only analogous/contextual.
            - Any market-implied probabilities and caveats.
            - Any source-of-truth data values and update/revision caveats.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            Evidence JSON:
            {json.dumps([asdict(item) for item in evidence_items], indent=2)[:12000]}
            """
        )
        try:
            return await self.get_llm("summarizer", "llm").invoke(prompt)
        except Exception as error:
            logger.warning(
                "Direct evidence synthesis failed for URL %s: %r",
                question.page_url,
                error,
            )
            return "Synthesis unavailable; use raw structured evidence above."

    async def _fetch_cached_kalshi_markets(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        return await self._cached_evidence_fetch(
            "kalshi", question, lambda: self._fetch_kalshi_markets(question)
        )

    async def _fetch_kalshi_markets(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        base_url = os.getenv(
            "KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
        ).rstrip("/")
        markets_by_key: dict[str, dict[str, Any]] = {}
        params_to_try = [{"status": "open", "limit": 1000}]
        params_to_try.extend(
            {"status": "open", "limit": 100, "query": query}
            for query in self._market_search_queries(question)[:5]
        )
        for params in params_to_try:
            try:
                payload = await self._http_get_json(
                    f"{base_url}/markets",
                    params=params,
                    timeout=25,
                )
            except Exception as error:
                logger.warning("Kalshi market fetch failed: %r", error)
                continue
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for market in markets:
                if not isinstance(market, dict):
                    continue
                key = str(market.get("ticker") or market.get("title") or "")
                if key:
                    markets_by_key[key] = market
        evidence_items: list[EvidenceItem] = []
        for market in markets_by_key.values():
            title = str(market.get("title") or market.get("subtitle") or "")
            candidate_text = " ".join(
                str(market.get(key, ""))
                for key in (
                    "title",
                    "subtitle",
                    "yes_sub_title",
                    "no_sub_title",
                    "rules_primary",
                    "category",
                )
            )
            score = self._relevance_score(candidate_text, question)
            directness = self._market_relevance_directness(
                candidate_text, question, score
            )
            if directness is None:
                continue
            yes_bid = self._normalise_market_probability(market.get("yes_bid"))
            yes_ask = self._normalise_market_probability(market.get("yes_ask"))
            last_price = self._normalise_market_probability(market.get("last_price"))
            if yes_bid is None:
                yes_bid = self._normalise_market_probability(
                    market.get("yes_bid_dollars")
                )
            if yes_ask is None:
                yes_ask = self._normalise_market_probability(
                    market.get("yes_ask_dollars")
                )
            if last_price is None:
                last_price = self._normalise_market_probability(
                    market.get("last_price_dollars")
                )
            probability_values = [
                value for value in (yes_bid, yes_ask, last_price) if value is not None
            ]
            probability = (
                statistics.mean(probability_values) if probability_values else None
            )
            ticker = market.get("ticker", "")
            url = f"https://kalshi.com/markets/{ticker}" if ticker else "https://kalshi.com/markets"
            evidence_items.append(
                EvidenceItem(
                    source="Kalshi",
                    provider="kalshi",
                    title=title or str(ticker),
                    url=url,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary=(
                        f"Open Kalshi market matched by local text relevance. "
                        f"Volume: {market.get('volume', 'n/a')}; open interest: {market.get('open_interest', 'n/a')}."
                    ),
                    value=f"yes_bid={yes_bid}, yes_ask={yes_ask}, last={last_price}",
                    probability=probability,
                    date=str(market.get("close_time") or market.get("expiration_time") or ""),
                    directness=directness,
                    caveats="Kalshi market may not match Metaculus resolution exactly; verify title/rules.",
                    raw={
                        key: market.get(key)
                        for key in (
                            "ticker",
                            "title",
                            "yes_bid",
                            "yes_bid_dollars",
                            "yes_ask",
                            "yes_ask_dollars",
                            "last_price",
                            "last_price_dollars",
                            "volume",
                            "open_interest",
                            "close_time",
                        )
                    },
                )
            )
        return evidence_items[: self._direct_evidence_max_items_per_provider]

    async def _fetch_cached_polymarket_markets(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        return await self._cached_evidence_fetch(
            "polymarket", question, lambda: self._fetch_polymarket_markets(question)
        )

    @classmethod
    def _iter_polymarket_market_payloads(
        cls, payload: dict[str, Any] | list[Any]
    ) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            if (
                isinstance(payload.get("markets"), list)
                and not payload.get("question")
                and (payload.get("slug") or payload.get("title") or payload.get("name"))
            ):
                payload_items = [payload]
            else:
                payload_items = []
                for key in ("markets", "events"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        payload_items.extend(value)
                if not payload_items:
                    payload_items = [payload]
        elif isinstance(payload, list):
            payload_items = payload
        else:
            payload_items = []

        markets: list[dict[str, Any]] = []
        for item in payload_items:
            if not isinstance(item, dict):
                continue
            nested_markets = item.get("markets")
            if isinstance(nested_markets, list):
                event_title = item.get("title") or item.get("name") or item.get("slug")
                event_slug = item.get("slug")
                for market in nested_markets:
                    if not isinstance(market, dict):
                        continue
                    enriched_market = dict(market)
                    if event_title and not enriched_market.get("eventTitle"):
                        enriched_market["eventTitle"] = event_title
                    if event_slug and not enriched_market.get("eventSlug"):
                        enriched_market["eventSlug"] = event_slug
                    markets.append(enriched_market)
            elif item.get("question") or item.get("title"):
                markets.append(item)
        return markets

    @classmethod
    def _slugify_market_query(cls, query: str) -> str:
        normalized = cls._normalise_relevance_text(query)
        slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
        return re.sub(r"-+", "-", slug)

    @classmethod
    def _polymarket_candidate_event_slugs(
        cls, question: MetaculusQuestion
    ) -> list[str]:
        slugs: list[str] = []
        for query in cls._market_search_queries(question):
            slug = cls._slugify_market_query(query)
            if slug and slug not in slugs:
                slugs.append(slug)
        return slugs[:8]

    async def _fetch_polymarket_markets(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        gamma_url = os.getenv(
            "POLYMARKET_GAMMA_MARKETS_URL", "https://gamma-api.polymarket.com/markets"
        )
        events_url = os.getenv(
            "POLYMARKET_GAMMA_EVENTS_URL",
            f"{gamma_url.rstrip('/').rsplit('/markets', 1)[0]}/events",
        )
        markets_by_key: dict[str, dict[str, Any]] = {}
        search_queries = self._market_search_queries(question)

        request_specs: list[tuple[str, dict[str, Any]]] = []
        for query in search_queries[:6]:
            request_specs.append(
                (
                    gamma_url,
                    {
                        "active": "true",
                        "closed": "false",
                        "limit": 100,
                        "search": query,
                    },
                )
            )
            request_specs.append(
                (
                    events_url,
                    {
                        "active": "true",
                        "closed": "false",
                        "limit": 25,
                        "search": query,
                    },
                )
            )
        for slug in self._polymarket_candidate_event_slugs(question):
            request_specs.append((events_url, {"slug": slug, "limit": 5}))
        request_specs.append(
            (
                gamma_url,
                {"active": "true", "closed": "false", "limit": 500},
            )
        )

        for url, params in request_specs:
            try:
                payload = await self._http_get_json(url, params=params, timeout=25)
            except Exception as error:
                logger.warning("Polymarket market fetch failed: %r", error)
                continue
            for market in self._iter_polymarket_market_payloads(payload):
                key = str(
                    market.get("id")
                    or market.get("conditionId")
                    or market.get("slug")
                    or market.get("question")
                    or ""
                )
                if key:
                    markets_by_key[key] = market
        evidence_items: list[EvidenceItem] = []
        for market in markets_by_key.values():
            title = str(market.get("question") or market.get("title") or "")
            candidate_text = " ".join(
                str(market.get(key, ""))
                for key in (
                    "question",
                    "title",
                    "description",
                    "category",
                    "slug",
                    "eventSlug",
                    "eventTitle",
                    "groupItemTitle",
                )
            )
            score = self._relevance_score(candidate_text, question)
            directness = self._market_relevance_directness(
                candidate_text, question, score
            )
            if directness is None:
                continue
            outcome_prices = self._parse_jsonish_list(market.get("outcomePrices"))
            outcomes = self._parse_jsonish_list(market.get("outcomes"))
            probability = None
            if outcome_prices:
                yes_index = 0
                if outcomes:
                    for index, outcome in enumerate(outcomes):
                        if str(outcome).strip().lower() in {"yes", "true"}:
                            yes_index = index
                            break
                if yes_index < len(outcome_prices):
                    probability = self._normalise_market_probability(
                        outcome_prices[yes_index]
                    )
            slug = market.get("slug") or market.get("marketSlug") or ""
            url = f"https://polymarket.com/market/{slug}" if slug else "https://polymarket.com/markets"
            evidence_items.append(
                EvidenceItem(
                    source="Polymarket",
                    provider="polymarket",
                    title=title,
                    url=url,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary=(
                        f"Active Polymarket market matched by local text relevance. "
                        f"Volume: {market.get('volume', 'n/a')}; liquidity: {market.get('liquidity', 'n/a')}."
                    ),
                    value=f"outcomes={outcomes}, outcomePrices={outcome_prices}",
                    probability=probability,
                    date=str(market.get("endDate") or market.get("end_date") or ""),
                    directness=directness,
                    caveats="Polymarket outcomes may map imperfectly to Metaculus criteria; check market rules.",
                    raw={
                        key: market.get(key)
                        for key in (
                            "id",
                            "question",
                            "slug",
                            "outcomes",
                            "outcomePrices",
                            "volume",
                            "liquidity",
                            "eventSlug",
                            "eventTitle",
                            "endDate",
                        )
                    },
                )
            )
        return evidence_items[: self._direct_evidence_max_items_per_provider]

    async def _fetch_cached_manifold_markets(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        return await self._cached_evidence_fetch(
            "manifold", question, lambda: self._fetch_manifold_markets(question)
        )

    async def _fetch_manifold_markets(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        search_terms = self._market_search_query(question)
        try:
            payload = await self._http_get_json(
                "https://api.manifold.markets/v0/search-markets",
                params={"term": search_terms, "limit": 20},
                timeout=20,
            )
        except Exception as error:
            logger.warning("Manifold market fetch failed: %r", error)
            return []
        markets = payload if isinstance(payload, list) else payload.get("markets", [])
        evidence_items: list[EvidenceItem] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            title = str(market.get("question") or market.get("title") or "")
            candidate_text = " ".join(
                str(market.get(key, ""))
                for key in ("question", "description", "textDescription", "outcomeType")
            )
            score = self._relevance_score(candidate_text, question)
            directness = self._market_relevance_directness(
                candidate_text, question, score
            )
            if directness is None:
                continue
            probability = self._normalise_market_probability(market.get("probability"))
            url = market.get("url") or (
                f"https://manifold.markets/{market.get('creatorUsername', '')}/{market.get('slug', '')}"
                if market.get("slug")
                else "https://manifold.markets/"
            )
            evidence_items.append(
                EvidenceItem(
                    source="Manifold",
                    provider="manifold",
                    title=title,
                    url=str(url),
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary=(
                        f"Manifold market search result for '{search_terms}'. "
                        f"Volume/liquidity: {market.get('volume', market.get('volume24Hours', 'n/a'))}."
                    ),
                    probability=probability,
                    date=str(market.get("closeTime") or ""),
                    directness=directness,
                    caveats="Manifold is often useful for priors but may be thinly traded or community-driven.",
                    raw={
                        key: market.get(key)
                        for key in (
                            "id",
                            "question",
                            "probability",
                            "url",
                            "slug",
                            "volume",
                            "closeTime",
                            "outcomeType",
                        )
                    },
                )
            )
        return evidence_items[: self._direct_evidence_max_items_per_provider]

    @classmethod
    def _market_search_query(cls, question: MetaculusQuestion) -> str:
        question_text = getattr(question, "question_text", "")
        cleaned = re.sub(r"\b(will|by|before|after|resolve|happen|there|be)\b", " ", question_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^a-zA-Z0-9 \-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:140] or question_text[:140]

    @classmethod
    def _market_search_queries(cls, question: MetaculusQuestion) -> list[str]:
        queries: list[str] = []

        def add(query: str) -> None:
            query = re.sub(r"\s+", " ", query).strip()
            if query and query.lower() not in {item.lower() for item in queries}:
                queries.append(query[:140])

        question_text = str(getattr(question, "question_text", ""))
        add(cls._market_search_query(question))
        for country in cls._country_names_for_question(question):
            if cls._is_election_question(question):
                add(f"{country} presidential election")
                add(f"{country} election")
            else:
                add(country)

        option_texts = [
            str(option).strip()
            for option in getattr(question, "options", None) or []
            if str(option).strip().lower()
            not in {"another", "another candidate", "other", "other candidate"}
        ]
        for option_text in option_texts[:6]:
            add(option_text)
            for country in cls._country_names_for_question(question)[:2]:
                add(f"{option_text} {country}")
                if cls._is_election_question(question):
                    add(f"{option_text} {country} election")

        for phrase_match in re.finditer(
            r"\b\d{4}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+election\b",
            question_text,
        ):
            add(phrase_match.group(0))

        return queries[:10]

    async def _fetch_cached_fred_series(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        series_ids = self._fred_series_for_question(question)
        if not series_ids:
            return []
        extra = ",".join(series_id for series_id, _ in series_ids)
        return await self._cached_evidence_fetch(
            "fred", question, lambda: self._fetch_fred_series(question, series_ids), extra
        )

    @classmethod
    def _fred_series_for_question(
        cls, question: MetaculusQuestion
    ) -> list[tuple[str, str]]:
        question_blob = cls._question_text_blob(question)
        matches: list[tuple[str, str]] = []
        for keyword, series_info in cls._fred_series_by_keyword.items():
            if keyword in question_blob and series_info not in matches:
                matches.append(series_info)
        return matches[:4]

    async def _fetch_fred_series(
        self,
        question: MetaculusQuestion,
        series_ids: list[tuple[str, str]],
    ) -> list[EvidenceItem]:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            return []
        evidence_items: list[EvidenceItem] = []
        for series_id, series_name in series_ids:
            try:
                payload = await self._http_get_json(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": api_key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 3,
                    },
                    timeout=20,
                )
            except Exception as error:
                logger.warning("FRED fetch failed for %s: %r", series_id, error)
                continue
            observations = (
                payload.get("observations", []) if isinstance(payload, dict) else []
            )
            latest = next(
                (
                    observation
                    for observation in observations
                    if observation.get("value") not in {None, "."}
                ),
                None,
            )
            if latest is None:
                continue
            evidence_items.append(
                EvidenceItem(
                    source="FRED",
                    provider="fred",
                    title=f"{series_name} ({series_id})",
                    url=f"https://fred.stlouisfed.org/series/{series_id}",
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary="Latest FRED observation for a mapped economic series.",
                    value=str(latest.get("value", "")),
                    unit="series units",
                    date=str(latest.get("date", "")),
                    directness="similar",
                    caveats="Check units, release timing, revisions, and whether this exact series matches the resolution criteria.",
                    raw={"series_id": series_id, "latest": latest},
                )
            )
        return evidence_items[: self._direct_evidence_max_items_per_provider]

    async def _fetch_cached_sec_edgar(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        company_matches = await self._sec_company_matches(question)
        if not company_matches:
            return []
        extra = ",".join(match["cik_str"] for match in company_matches)
        return await self._cached_evidence_fetch(
            "sec-edgar",
            question,
            lambda: self._fetch_sec_edgar(question, company_matches),
            extra,
        )

    async def _sec_company_matches(
        self, question: MetaculusQuestion
    ) -> list[dict[str, Any]]:
        try:
            ticker_payload = await self._http_get_json(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": self._sec_user_agent()},
                timeout=20,
            )
        except Exception as error:
            logger.warning("SEC ticker map fetch failed: %r", error)
            return []
        question_blob = self._question_text_blob(question)
        ticker_tokens = set(re.findall(r"\b[A-Z]{1,5}\b", getattr(question, "question_text", "")))
        company_entries = (
            ticker_payload.values()
            if isinstance(ticker_payload, dict)
            else ticker_payload
        )
        matches: list[dict[str, Any]] = []
        for entry in company_entries:
            ticker = str(entry.get("ticker", "")).upper()
            title = str(entry.get("title", ""))
            title_lc = title.lower()
            if ticker in ticker_tokens or (
                len(title_lc) >= 5 and title_lc in question_blob
            ):
                cik_str = str(entry.get("cik_str", "")).zfill(10)
                matches.append(
                    {
                        "ticker": ticker,
                        "title": title,
                        "cik_str": cik_str,
                    }
                )
            if len(matches) >= 3:
                break
        return matches

    @staticmethod
    def _sec_user_agent() -> str:
        return os.getenv(
            "SEC_USER_AGENT",
            "metac-bot-template research bot; set SEC_USER_AGENT with contact email",
        )

    async def _fetch_sec_edgar(
        self,
        question: MetaculusQuestion,
        company_matches: list[dict[str, Any]],
    ) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []
        for company in company_matches:
            cik = company["cik_str"]
            headers = {"User-Agent": self._sec_user_agent()}
            try:
                submissions = await self._http_get_json(
                    f"https://data.sec.gov/submissions/CIK{cik}.json",
                    headers=headers,
                    timeout=20,
                )
            except Exception as error:
                logger.warning("SEC submissions fetch failed for %s: %r", cik, error)
                continue
            recent = submissions.get("filings", {}).get("recent", {}) if isinstance(submissions, dict) else {}
            forms = recent.get("form", [])[:10]
            filing_dates = recent.get("filingDate", [])[:10]
            accession_numbers = recent.get("accessionNumber", [])[:10]
            recent_filings = [
                f"{form} filed {filing_date}"
                for form, filing_date in zip(forms, filing_dates)
                if form and filing_date
            ][:5]
            url = f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}"
            evidence_items.append(
                EvidenceItem(
                    source="SEC EDGAR submissions",
                    provider="sec-edgar",
                    title=f"{company['title']} ({company['ticker']}) recent filings",
                    url=url,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary="; ".join(recent_filings)
                    or "SEC company found, but no recent filing summary was parsed.",
                    date=filing_dates[0] if filing_dates else "",
                    directness="similar",
                    caveats="Filing existence is not itself a forecast; inspect filings for resolution-specific facts.",
                    raw={
                        "company": company,
                        "recent_forms": forms[:5],
                        "recent_filing_dates": filing_dates[:5],
                        "accession_numbers": accession_numbers[:5],
                    },
                )
            )
            evidence_items.extend(await self._fetch_sec_companyfacts(company))
        return evidence_items[: self._direct_evidence_max_items_per_provider]

    async def _fetch_sec_companyfacts(
        self, company: dict[str, Any]
    ) -> list[EvidenceItem]:
        cik = company["cik_str"]
        headers = {"User-Agent": self._sec_user_agent()}
        try:
            companyfacts = await self._http_get_json(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                headers=headers,
                timeout=20,
            )
        except Exception as error:
            logger.warning("SEC companyfacts fetch failed for %s: %r", cik, error)
            return []
        us_gaap = companyfacts.get("facts", {}).get("us-gaap", {}) if isinstance(companyfacts, dict) else {}
        fact_names = [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "NetIncomeLoss",
            "Assets",
            "OperatingIncomeLoss",
        ]
        evidence_items: list[EvidenceItem] = []
        for fact_name in fact_names:
            fact = us_gaap.get(fact_name)
            if not isinstance(fact, dict):
                continue
            units = fact.get("units", {})
            if not isinstance(units, dict) or not units:
                continue
            unit_name, values = next(iter(units.items()))
            if not isinstance(values, list):
                continue
            numeric_values = [
                value
                for value in values
                if value.get("val") is not None and value.get("end")
            ]
            if not numeric_values:
                continue
            latest = max(numeric_values, key=lambda value: value.get("filed", ""))
            evidence_items.append(
                EvidenceItem(
                    source="SEC EDGAR companyfacts",
                    provider="sec-companyfacts",
                    title=f"{company['title']} {fact_name}",
                    url=f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary=f"Latest parsed SEC company fact for {fact_name}.",
                    value=str(latest.get("val", "")),
                    unit=unit_name,
                    date=str(latest.get("end", "")),
                    directness="similar",
                    caveats="Company facts may be quarterly/annual and may not map to the question's exact metric.",
                    raw={"company": company, "fact": fact_name, "latest": latest},
                )
            )
            if len(evidence_items) >= 2:
                break
        return evidence_items

    async def _fetch_cached_fivethirtyeight_polls(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        return await self._cached_evidence_fetch(
            "fivethirtyeight-polls",
            question,
            lambda: self._fetch_fivethirtyeight_polls(question),
        )

    async def _fetch_fivethirtyeight_polls(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        table_names = [
            "president_polls",
            "senate_polls",
            "house_polls",
            "generic_ballot_polls",
        ]
        evidence_items: list[EvidenceItem] = []
        for table_name in table_names:
            try:
                payload = await self._http_get_json(
                    f"https://projects.fivethirtyeight.com/polls-page/{table_name}.json",
                    timeout=20,
                )
            except Exception:
                try:
                    payload = await self._http_get_json(
                        f"https://fivethirtyeight.datasettes.com/polls/{table_name}.json",
                        params={
                            "_shape": "array",
                            "_size": 10,
                            "_sort_desc": "end_date",
                        },
                        timeout=20,
                    )
                except Exception as error:
                    logger.info("FiveThirtyEight fetch failed for %s: %r", table_name, error)
                    continue
            rows = payload if isinstance(payload, list) else payload.get("rows", [])
            for row in rows[:10]:
                if not isinstance(row, dict):
                    continue
                row_text = json.dumps(row, sort_keys=True)
                score = self._relevance_score(row_text, question)
                if score < 0.05:
                    continue
                pollster = row.get("pollster") or row.get("pollster_name") or "poll"
                date = row.get("end_date") or row.get("endDate") or row.get("created_at") or ""
                evidence_items.append(
                    EvidenceItem(
                        source="FiveThirtyEight polling data",
                        provider="fivethirtyeight",
                        title=f"{table_name}: {pollster}",
                        url="https://data.fivethirtyeight.com/",
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                        summary=self._truncate_for_prompt(row_text, 500),
                        date=str(date),
                        directness=self._directness_from_score(score),
                        caveats="Polling rows may require aggregation and likely do not directly map to final election outcome.",
                        raw=row,
                    )
                )
                if len(evidence_items) >= self._direct_evidence_max_items_per_provider:
                    return evidence_items
        return evidence_items

    async def _fetch_cached_acled_events(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        return await self._cached_evidence_fetch(
            "acled",
            question,
            lambda: self._fetch_acled_events(question),
        )

    async def _fetch_acled_events(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        api_url = os.getenv("ACLED_API_URL", "https://api.acleddata.com/acled/read")
        api_key = os.getenv("ACLED_API_KEY")
        email = os.getenv("ACLED_EMAIL")
        if not api_key or not email:
            return []
        params: dict[str, Any] = {
            "key": api_key,
            "email": email,
            "limit": 20,
        }
        countries = self._candidate_country_terms(question)
        if countries:
            params["country"] = "|".join(countries[:3])
        try:
            payload = await self._http_get_json(api_url, params=params, timeout=25)
        except Exception as error:
            logger.warning("ACLED fetch failed: %r", error)
            return []
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        evidence_items: list[EvidenceItem] = []
        for row in rows[: self._direct_evidence_max_items_per_provider]:
            row_text = json.dumps(row, sort_keys=True)
            evidence_items.append(
                EvidenceItem(
                    source="ACLED",
                    provider="acled",
                    title=f"{row.get('event_type', 'event')} in {row.get('country', '')}",
                    url="https://acleddata.com/",
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary=self._truncate_for_prompt(row_text, 500),
                    date=str(row.get("event_date", "")),
                    directness="similar",
                    caveats="ACLED event counts need careful filtering by country/date/event type before use as a forecast input.",
                    raw=row,
                )
            )
        return evidence_items

    @staticmethod
    def _candidate_country_terms(question: MetaculusQuestion) -> list[str]:
        country_names = [
            "Ukraine",
            "Russia",
            "China",
            "Taiwan",
            "Israel",
            "Iran",
            "Gaza",
            "Lebanon",
            "Syria",
            "Yemen",
            "Sudan",
            "United States",
            "United Kingdom",
            "France",
            "Germany",
            "Colombia",
            "India",
            "Pakistan",
            "North Korea",
            "South Korea",
        ]
        question_text = " ".join(
            [
                getattr(question, "question_text", ""),
                getattr(question, "background_info", ""),
                getattr(question, "resolution_criteria", ""),
            ]
        )
        return [
            country
            for country in country_names
            if country.lower() in question_text.lower()
        ]

    async def _fetch_ai_benchmark_registry(
        self, question: MetaculusQuestion
    ) -> list[EvidenceItem]:
        evidence_items: list[EvidenceItem] = []
        question_blob = self._question_text_blob(question)
        for source_name, url, description in self._ai_benchmark_registry:
            score = self._relevance_score(f"{source_name} {description}", question)
            if score < 0.04 and not any(
                keyword in question_blob
                for keyword in ("ai", "model", "benchmark", "llm", "agi")
            ):
                continue
            evidence_items.append(
                EvidenceItem(
                    source=source_name,
                    provider="ai-benchmark-registry",
                    title=source_name,
                    url=url,
                    retrieved_at=datetime.now(timezone.utc).isoformat(),
                    summary=description,
                    directness="weak" if score < 0.10 else "similar",
                    caveats="Registry pointer only; use official-data search or deep research to extract the current leaderboard value.",
                )
            )
        return evidence_items[: self._direct_evidence_max_items_per_provider]

    async def _run_market_prior_research(
        self, question: MetaculusQuestion
    ) -> str:
        if not os.getenv("OPENROUTER_API_KEY"):
            return ""
        prompt = clean_indents(
            f"""
            Search specifically for prediction-market priors relevant to this forecasting question.

            Search targets:
            - Kalshi
            - Polymarket
            - Manifold Markets
            - Metaculus/community forecasts only as a secondary reference if directly relevant

            Return a compact table with columns:
            Platform | Market title | URL/source | Directness | Raw market price | Implied probability | Notes

            Requirements:
            - Separate direct markets from merely analogous markets.
            - Convert prices/shares to an implied probability when possible.
            - For multiple-choice, numeric, or date questions, explain what event the market is actually pricing and how it maps to the Metaculus resolution criteria.
            - Include volume/liquidity or last-updated information when available.
            - If you find no close market, say "No close market prior found" and list the best analogous searches tried.
            - Do not make a final forecast.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        searcher = SmartSearcher(
            model=os.getenv(
                "MARKET_PRIOR_SEARCH_MODEL",
                "openrouter/mistralai/mistral-large-2512",
            ),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=8,
            use_advanced_filters=False,
        )
        research = await searcher.invoke(prompt)
        logger.info(
            "Market-prior research for URL %s:\n%s", question.page_url, research
        )
        return research

    async def _run_official_data_research(
        self, question: MetaculusQuestion, topics: list[str]
    ) -> str:
        if not os.getenv("OPENROUTER_API_KEY"):
            return ""
        topic_hints = "\n".join(
            f"- {topic}: {self._official_data_source_hints[topic]}"
            for topic in topics
            if topic in self._official_data_source_hints
        )
        prompt = clean_indents(
            f"""
            Run a lightweight official-data research pass for this forecasting question.

            Detected topic categories:
            {", ".join(topics)}

            Preferred source hints:
            {topic_hints}

            Output exactly these sections:
            ## Official/source-of-truth data
            - Latest relevant official values, statuses, documents, dates, or benchmark results.
            - Source name and URL for each key fact.

            ## Data caveats
            - Lag, revision risk, unit mismatches, definitional mismatch, or whether a source is unofficial.

            ## Forecast relevance
            - Briefly state how each fact changes the evidence base. Do not make a final forecast.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        searcher = SmartSearcher(
            model=os.getenv(
                "OFFICIAL_DATA_SEARCH_MODEL",
                "openrouter/mistralai/mistral-large-2512",
            ),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=8,
            use_advanced_filters=False,
        )
        research = await searcher.invoke(prompt)
        logger.info(
            "Official data research for URL %s:\n%s", question.page_url, research
        )
        return research

    async def _run_optional_deep_research(
        self,
        question: MetaculusQuestion,
        focus: str,
        reason: str,
    ) -> str:
        llm = self._deep_research_llm()
        if llm is None:
            logger.info(
                "Skipping optional deep research for URL %s because no deep research provider is configured.",
                question.page_url,
            )
            return ""
        prompt = clean_indents(
            f"""
            You are doing a deeper research pass for a forecasting question.
            Reason this pass is being run: {reason}

            Focus:
            {focus}

            Produce a compact, citation-heavy memo with:
            ## Deep research findings
            - The most resolution-relevant facts from high-quality sources.

            ## Base rates / analogues
            - Historical cases or outside-view data where available.

            ## Market / expert priors
            - Prediction markets, polls, expert forecasts, or consensus estimates where available.

            ## Unresolved cruxes
            - Facts still uncertain after research.

            Do not produce a final forecast.

            Question:
            {question.question_text}

            Background:
            {question.background_info}

            Resolution criteria:
            {question.resolution_criteria}

            Fine print:
            {question.fine_print}
            """
        )
        research = await llm.invoke(prompt)
        logger.info(
            "Optional deep research via %s for URL %s:\n%s",
            llm.model,
            question.page_url,
            research,
        )
        return research

    async def _run_disagreement_targeted_research(
        self,
        question: MetaculusQuestion,
        predictions: list[ReasonedPrediction[T]],
    ) -> str:
        cruxes = ""
        if os.getenv("OPENROUTER_API_KEY"):
            try:
                cruxes = await self._extract_disagreement_cruxes(question, predictions)
            except Exception as error:
                logger.warning(
                    "Disagreement crux extraction failed for URL %s: %r",
                    question.page_url,
                    error,
                )
        research_tasks: list[tuple[str, asyncio.Task[str]]] = []
        if cruxes.strip() and os.getenv("OPENROUTER_API_KEY"):
            research_tasks.append(
                (
                    "Targeted crux search",
                    asyncio.create_task(
                        self._run_targeted_crux_search(question, cruxes)
                    ),
                )
            )
        if self._env_flag("ENABLE_DEEP_RESEARCH_ON_DISAGREEMENT", True):
            research_tasks.append(
                (
                    "Optional deep research",
                    asyncio.create_task(
                        self._run_optional_deep_research(
                            question,
                            focus=cruxes
                            or "The cheap forecaster ensemble disagreed, but no clean crux extraction was available.",
                            reason="ensemble-disagreement",
                        )
                    ),
                )
            )

        if not research_tasks:
            return ""

        results = await asyncio.gather(
            *(task for _, task in research_tasks), return_exceptions=True
        )
        research_sections: list[str] = []
        for (section_name, _), result in zip(research_tasks, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "%s failed for URL %s: %r",
                    section_name,
                    question.page_url,
                    result,
                )
            elif result.strip():
                research_sections.append(f"## {section_name}\n{result}")

        return "\n\n".join(research_sections)

    async def _extract_disagreement_cruxes(
        self,
        question: MetaculusQuestion,
        predictions: list[ReasonedPrediction[T]],
    ) -> str:
        model_outputs = "\n\n".join(
            self._truncate_for_prompt(
                f"## Cheap role output {i + 1}\nPrediction: {prediction.prediction_value}\n\nReasoning:\n{prediction.reasoning}",
                3500,
            )
            for i, prediction in enumerate(predictions)
        )
        prompt = clean_indents(
            f"""
            The cheap role-specific forecasting ensemble disagreed on this Metaculus question.
            Identify the factual or interpretive cruxes that most explain the disagreement.

            Return 2-4 short bullets. Focus on facts that can be checked with targeted web research:
            current values, official data, market priors, resolution interpretation, base rates, or specific assumptions.
            Do not make a forecast.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            Model outputs:
            {model_outputs}
            """
        )
        analyzer = self.get_llm("summarizer", "llm")
        cruxes = await analyzer.invoke(prompt)
        logger.info(
            "Disagreement cruxes for URL %s:\n%s", question.page_url, cruxes
        )
        return self._truncate_for_prompt(cruxes, 2500)

    async def _run_targeted_crux_search(
        self, question: MetaculusQuestion, cruxes: str
    ) -> str:
        prompt = clean_indents(
            f"""
            Search the web to resolve the specific cruxes below for a forecasting question.
            Prioritize official data, primary sources, prediction markets, expert forecasts, and source-of-truth details.
            Be concise. Include URLs or source names. Do not produce a final probability or distribution.

            Output:
            ## Targeted crux research
            - For each crux, state the best evidence found and whether it supports a higher/lower forecast, earlier/later date, or a particular option.
            - Flag unresolved uncertainty or source conflict.

            Question:
            {question.question_text}

            Resolution criteria:
            {question.resolution_criteria}

            Cruxes:
            {cruxes}
            """
        )
        searcher = SmartSearcher(
            model="openrouter/mistralai/mistral-large-2512",
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=8,
            use_advanced_filters=False,
        )
        research = await searcher.invoke(prompt)
        logger.info(
            "Targeted crux research for URL %s:\n%s", question.page_url, research
        )
        return research

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}


            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._forecasting_checklist()}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.
            {self._get_conditional_disclaimer_if_necessary(question)}

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )

        return await self._binary_prompt_to_forecast(question, prompt)

    async def _binary_prompt_to_forecast(
        self,
        question: BinaryQuestion,
        prompt: str,
    ) -> ReasonedPrediction[float]:
        tasks = [
            asyncio.create_task(
                self._binary_prediction_from_llm(
                    spec.llm, question, self._role_prompt(prompt, spec), spec
                )
            )
            for spec in self._base_forecaster_specs()
        ]
        predictions = await self._gather_ensemble_predictions(tasks, question)
        if self._should_escalate_binary(predictions):
            logger.info(
                "Binary ensemble disagreement triggered GPT-5.5 escalation for URL %s.",
                question.page_url,
            )
            escalation_spec = self._escalation_forecaster_spec()
            predictions.append(
                await self._binary_prediction_from_llm(
                    escalation_spec.llm,
                    question,
                    self._role_prompt(
                        await self._build_targeted_escalation_prompt(
                            question, prompt, predictions
                        ),
                        escalation_spec,
                    ),
                    escalation_spec,
                )
            )
        predictions_for_aggregation = self._collapse_same_model_binary_predictions(
            predictions
        )
        model_probabilities = [
            prediction.prediction_value for prediction in predictions_for_aggregation
        ]
        decimal_pred = self._logit_mean_probability(model_probabilities)

        logger.info(
            f"Forecasted URL {question.page_url} with ensemble prediction: {decimal_pred}. "
            f"Member probabilities: {model_probabilities}"
        )
        return ReasonedPrediction(
            prediction_value=decimal_pred,
            reasoning=self._combine_model_reasoning(predictions_for_aggregation),
        )

    async def _binary_prediction_from_llm(
        self,
        llm: GeneralLlm,
        question: BinaryQuestion,
        prompt: str,
        role_spec: ForecasterRoleSpec | None = None,
    ) -> ReasonedPrediction[float]:
        reasoning = await llm.invoke(prompt)
        role_name = role_spec.name if role_spec else "Unspecified role"
        logger.info(
            "Reasoning from %s (%s) for URL %s: %s",
            llm.model,
            role_name,
            question.page_url,
            reasoning,
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
        )
        decimal_pred = self._clamp_probability(
            binary_prediction.prediction_in_decimal
        )

        logger.info(
            f"Forecasted URL {question.page_url} with {llm.model} ({role_name}) prediction: {decimal_pred}."
        )
        return ReasonedPrediction(
            prediction_value=decimal_pred,
            reasoning=f"Role: {role_name}\nModel: {llm.model}\n\n{reasoning}",
        )

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._forecasting_checklist()}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _multiple_choice_prompt_to_forecast(
        self,
        question: MultipleChoiceQuestion,
        prompt: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        tasks = [
            asyncio.create_task(
                self._multiple_choice_prediction_from_llm(
                    spec.llm, question, self._role_prompt(prompt, spec), spec
                )
            )
            for spec in self._base_forecaster_specs()
        ]
        predictions = await self._gather_ensemble_predictions(tasks, question)
        if self._should_escalate_multiple_choice(predictions):
            logger.info(
                "Multiple-choice ensemble disagreement triggered GPT-5.5 escalation for URL %s.",
                question.page_url,
            )
            escalation_spec = self._escalation_forecaster_spec()
            predictions.append(
                await self._multiple_choice_prediction_from_llm(
                    escalation_spec.llm,
                    question,
                    self._role_prompt(
                        await self._build_targeted_escalation_prompt(
                            question, prompt, predictions
                        ),
                        escalation_spec,
                    ),
                    escalation_spec,
                )
            )
        predictions_for_aggregation = (
            await self._collapse_same_model_multiple_choice_predictions(
                predictions, question
            )
        )
        option_lists = [
            prediction.prediction_value
            for prediction in predictions_for_aggregation
        ]
        aggregated_prediction = await MultipleChoiceReport.aggregate_predictions(
            option_lists, question
        )

        logger.info(
            f"Forecasted URL {question.page_url} with ensemble prediction: {aggregated_prediction}."
        )
        return ReasonedPrediction(
            prediction_value=aggregated_prediction,
            reasoning=self._combine_model_reasoning(predictions_for_aggregation),
        )

    async def _multiple_choice_prediction_from_llm(
        self,
        llm: GeneralLlm,
        question: MultipleChoiceQuestion,
        prompt: str,
        role_spec: ForecasterRoleSpec | None = None,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            """
        )
        reasoning = await llm.invoke(prompt)
        role_name = role_spec.name if role_spec else "Unspecified role"
        logger.info(
            "Reasoning from %s (%s) for URL %s: %s",
            llm.model,
            role_name,
            question.page_url,
            reasoning,
        )
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with {llm.model} ({role_name}) prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list,
            reasoning=f"Role: {role_name}\nModel: {llm.model}\n\n{reasoning}",
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._forecasting_checklist()}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 2.5 should always be less than the value for percentile 5, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 2.5: XX (lowest number value)
            Percentile 5: XX
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 50: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            Percentile 95: XX
            Percentile 97.5: XX (highest number value)
            "
            """
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    async def _numeric_prompt_to_forecast(
        self,
        question: NumericQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        tasks = [
            asyncio.create_task(
                self._numeric_prediction_from_llm(
                    spec.llm, question, self._role_prompt(prompt, spec), spec
                )
            )
            for spec in self._base_forecaster_specs()
        ]
        predictions = await self._gather_ensemble_predictions(tasks, question)
        if self._should_escalate_numeric(predictions, question):
            logger.info(
                "Numeric ensemble disagreement triggered GPT-5.5 escalation for URL %s.",
                question.page_url,
            )
            escalation_spec = self._escalation_forecaster_spec()
            predictions.append(
                await self._numeric_prediction_from_llm(
                    escalation_spec.llm,
                    question,
                    self._role_prompt(
                        await self._build_targeted_escalation_prompt(
                            question, prompt, predictions
                        ),
                        escalation_spec,
                    ),
                    escalation_spec,
                )
            )
        predictions_for_aggregation = await self._collapse_same_model_numeric_predictions(
            predictions, question
        )
        distributions = [
            prediction.prediction_value
            for prediction in predictions_for_aggregation
        ]
        aggregated_prediction = await NumericReport.aggregate_predictions(
            distributions, question
        )

        logger.info(
            f"Forecasted URL {question.page_url} with ensemble prediction: "
            f"{aggregated_prediction.declared_percentiles}."
        )
        return ReasonedPrediction(
            prediction_value=aggregated_prediction,
            reasoning=self._combine_model_reasoning(predictions_for_aggregation),
        )

    async def _numeric_prediction_from_llm(
        self,
        llm: GeneralLlm,
        question: NumericQuestion,
        prompt: str,
        role_spec: ForecasterRoleSpec | None = None,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await llm.invoke(prompt)
        role_name = role_spec.name if role_spec else "Unspecified role"
        logger.info(
            "Reasoning from %s (%s) for URL %s: %s",
            llm.model,
            role_name,
            question.page_url,
            reasoning,
        )
        deterministic_percentiles = self._parse_numeric_percentiles_from_text(
            reasoning
        )
        if deterministic_percentiles:
            try:
                prediction = NumericDistribution.from_question(
                    deterministic_percentiles, question
                )
            except Exception as error:
                logger.warning(
                    "Deterministic numeric percentile parsing failed for URL %s: %r",
                    question.page_url,
                    error,
                )
            else:
                logger.info(
                    "Parsed numeric percentiles deterministically for URL %s with %s (%s): %s.",
                    question.page_url,
                    llm.model,
                    role_name,
                    prediction.declared_percentiles,
                )
                return ReasonedPrediction(
                    prediction_value=prediction,
                    reasoning=f"Role: {role_name}\nModel: {llm.model}\n\n{reasoning}",
                )
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with {llm.model} ({role_name}) prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=f"Role: {role_name}\nModel: {llm.model}\n\n{reasoning}",
        )

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {self._forecasting_checklist()}

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ.No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 2.5 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
            Percentile 2.5: YYYY-MM-DD (oldest date)
            Percentile 5: YYYY-MM-DD
            Percentile 10: YYYY-MM-DD
            Percentile 20: YYYY-MM-DD
            Percentile 40: YYYY-MM-DD
            Percentile 50: YYYY-MM-DD
            Percentile 60: YYYY-MM-DD
            Percentile 80: YYYY-MM-DD
            Percentile 90: YYYY-MM-DD
            Percentile 95: YYYY-MM-DD
            Percentile 97.5: YYYY-MM-DD (newest date)
            "
            """
        )
        forecast = await self._date_prompt_to_forecast(question, prompt)
        return forecast

    async def _date_prompt_to_forecast(
        self,
        question: DateQuestion,
        prompt: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        tasks = [
            asyncio.create_task(
                self._date_prediction_from_llm(
                    spec.llm, question, self._role_prompt(prompt, spec), spec
                )
            )
            for spec in self._base_forecaster_specs()
        ]
        predictions = await self._gather_ensemble_predictions(tasks, question)
        if self._should_escalate_numeric(predictions, question):
            logger.info(
                "Date ensemble disagreement triggered GPT-5.5 escalation for URL %s.",
                question.page_url,
            )
            escalation_spec = self._escalation_forecaster_spec()
            predictions.append(
                await self._date_prediction_from_llm(
                    escalation_spec.llm,
                    question,
                    self._role_prompt(
                        await self._build_targeted_escalation_prompt(
                            question, prompt, predictions
                        ),
                        escalation_spec,
                    ),
                    escalation_spec,
                )
            )
        predictions_for_aggregation = await self._collapse_same_model_numeric_predictions(
            predictions, question
        )
        distributions = [
            prediction.prediction_value
            for prediction in predictions_for_aggregation
        ]
        aggregated_prediction = await NumericReport.aggregate_predictions(
            distributions, question
        )

        logger.info(
            f"Forecasted URL {question.page_url} with ensemble prediction: "
            f"{aggregated_prediction.declared_percentiles}."
        )
        return ReasonedPrediction(
            prediction_value=aggregated_prediction,
            reasoning=self._combine_model_reasoning(predictions_for_aggregation),
        )

    async def _date_prediction_from_llm(
        self,
        llm: GeneralLlm,
        question: DateQuestion,
        prompt: str,
        role_spec: ForecasterRoleSpec | None = None,
    ) -> ReasonedPrediction[NumericDistribution]:
        reasoning = await llm.invoke(prompt)
        role_name = role_spec.name if role_spec else "Unspecified role"
        logger.info(
            "Reasoning from %s (%s) for URL %s: %s",
            llm.model,
            role_name,
            question.page_url,
            reasoning,
        )
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with {llm.model} ({role_name}) prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=f"Role: {role_name}\nModel: {llm.model}\n\n{reasoning}",
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast) # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


def _select_refresh_scout_questions(
    questions: list[MetaculusQuestion],
    *,
    max_questions: int | None,
    shuffle_seed: int | None,
) -> list[MetaculusQuestion]:
    forecasted_questions = [
        question
        for question in questions
        if bool(getattr(question, "already_forecasted", False))
    ]
    return _select_question_batch(
        forecasted_questions,
        max_questions=max_questions,
        shuffle_seed=shuffle_seed,
    )


def _refresh_decision_priority(decision: RefreshScoutDecision) -> tuple[int, int]:
    reason_text = " ".join(decision.gate_reasons).lower()
    close_priority = 0 if "closes soon" in reason_text else 1
    confidence_priority = {"high": 0, "medium": 1, "low": 2}.get(
        decision.confidence_in_movement, 2
    )
    return (close_priority, confidence_priority)


async def _refresh_tournament_forecasts(
    bot: SpringTemplateBot2026,
    client: MetaculusClient,
    target_tournament_ids: list[str | int],
    *,
    max_scout_questions: int | None,
    max_full_reforecasts: int | None,
    shuffle_seed: int | None,
    continue_on_question_errors: bool,
) -> tuple[list[Any], list[RefreshScoutDecision]]:
    scout_decisions: list[RefreshScoutDecision] = []
    questions_for_full_reforecast: list[MetaculusQuestion] = []
    selected_question_keys: set[str] = set()

    for tournament_id in target_tournament_ids:
        questions = _get_open_tournament_questions(client, tournament_id)
        already_forecasted_count = len(
            [question for question in questions if getattr(question, "already_forecasted", False)]
        )
        scout_questions = _select_refresh_scout_questions(
            questions,
            max_questions=max_scout_questions,
            shuffle_seed=shuffle_seed,
        )
        logger.info(
            "Refresh scout selected %s of %s already-forecasted open questions "
            "from tournament %s (%s open total).",
            len(scout_questions),
            already_forecasted_count,
            tournament_id,
            len(questions),
        )
        for question in scout_questions:
            try:
                decision = await bot.run_refresh_scout(question)
            except BaseException as error:
                logger.error(
                    "Refresh scout failed for %s: %r",
                    getattr(question, "page_url", ""),
                    error,
                )
                previous_forecast = bot._latest_previous_forecast(question)
                previous_timestamp = bot._prediction_timestamp(previous_forecast)
                decision = RefreshScoutDecision(
                    question_id=getattr(question, "id_of_question", None),
                    post_id=getattr(question, "id_of_post", None),
                    page_url=getattr(question, "page_url", None),
                    question_title=getattr(question, "question_text", ""),
                    question_type=question.get_api_type_name(),
                    previous_forecast=bot._readable_prediction(previous_forecast),
                    previous_forecast_timestamp=(
                        previous_timestamp.isoformat() if previous_timestamp else None
                    ),
                    should_reforecast=False,
                    recommended_action="skip",
                    confidence_in_movement="low",
                    movement_reason="Scout failed.",
                    fresh_evidence_summary="",
                    parse_error=repr(error),
                )
            scout_decisions.append(decision)
            if not decision.gate_triggered:
                continue
            question_key_value = (
                getattr(question, "id_of_question", None)
                or getattr(question, "id_of_post", None)
                or getattr(question, "page_url", "")
            )
            question_key = str(question_key_value)
            if question_key in selected_question_keys:
                continue
            selected_question_keys.add(question_key)
            questions_for_full_reforecast.append(question)

    gated_decisions_by_url = {
        decision.page_url: decision
        for decision in scout_decisions
        if decision.gate_triggered and decision.page_url
    }
    questions_for_full_reforecast.sort(
        key=lambda question: _refresh_decision_priority(
            gated_decisions_by_url.get(
                getattr(question, "page_url", None),
                RefreshScoutDecision(
                    question_id=getattr(question, "id_of_question", None),
                    post_id=getattr(question, "id_of_post", None),
                    page_url=getattr(question, "page_url", None),
                    question_title=getattr(question, "question_text", ""),
                    question_type=question.get_api_type_name(),
                    previous_forecast="",
                    previous_forecast_timestamp=None,
                    should_reforecast=False,
                    recommended_action="skip",
                    confidence_in_movement="low",
                    movement_reason="",
                    fresh_evidence_summary="",
                ),
            )
        )
    )
    if max_full_reforecasts is not None and max_full_reforecasts >= 0:
        questions_for_full_reforecast = questions_for_full_reforecast[
            :max_full_reforecasts
        ]

    logger.info(
        "Refresh scout selected %s questions for full reforecast.",
        len(questions_for_full_reforecast),
    )
    if not questions_for_full_reforecast:
        return [], scout_decisions

    bot.skip_previously_forecasted_questions = False
    if continue_on_question_errors:
        forecast_reports = []
        for index, question in enumerate(questions_for_full_reforecast, start=1):
            logger.info(
                "Full refresh reforecast %s/%s: %s",
                index,
                len(questions_for_full_reforecast),
                getattr(question, "page_url", ""),
            )
            try:
                forecast_reports.extend(
                    await bot.forecast_questions([question], return_exceptions=True)
                )
            except BaseException as error:
                forecast_reports.append(error)
                logger.error(
                    "Full refresh reforecast failed for %s: %r",
                    getattr(question, "page_url", ""),
                    error,
                )
                if _looks_like_credit_exhaustion(error):
                    logger.error(
                        "Stopping refresh reforecasts because the model provider appears to be out of credits/quota."
                    )
                    break
    else:
        forecast_reports = await bot.forecast_questions(
            questions_for_full_reforecast, return_exceptions=True
        )
    return list(forecast_reports), scout_decisions


def _write_refresh_scout_logs(
    scout_decisions: list[RefreshScoutDecision],
    *,
    run_id: str,
    log_dir: Path,
) -> None:
    if not scout_decisions or not _env_bool("ENABLE_EXPERIMENT_LOGGING", True):
        return
    run_dir = log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(decision) for decision in scout_decisions]
    (run_dir / "refresh_scout_decisions.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    lines = ["# Refresh Scout Decisions", ""]
    for decision in scout_decisions:
        gate = "full reforecast" if decision.gate_triggered else "skip"
        reasons = "; ".join(decision.gate_reasons) if decision.gate_reasons else "-"
        lines.append(
            f"- `{decision.post_id or decision.question_id}` | {gate} | "
            f"{decision.confidence_in_movement} | {reasons} | "
            f"{decision.question_title}"
        )
    (run_dir / "refresh_scout_decisions.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    default_experiment_mode = os.getenv(
        "EXPERIMENT_MODE",
        "random" if _env_bool("ENABLE_EXPERIMENT_VARIANTS", False) else "off",
    )

    parser = argparse.ArgumentParser(
        description="Run the TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tournament",
            "repredict_tournament",
            "refresh_tournament",
            "minibench",
            "benchmarking_questions",
            "metaculus_cup",
            "question_urls",
            "test_questions",
        ],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    parser.add_argument(
        "--tournament-id",
        action="append",
        default=None,
        help="Tournament slug/id to forecast. Can be passed multiple times. Defaults to summer-futureeval-2026 for tournament modes.",
    )
    parser.add_argument(
        "--question-url",
        action="append",
        default=None,
        help="Metaculus question URL to forecast in question_urls mode. Can be passed multiple times or as comma-separated URLs.",
    )
    parser.add_argument(
        "--practice",
        action="store_true",
        help="Run forecasts without publishing to Metaculus. Useful for quick experiments.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=(
            int(os.getenv("MAX_QUESTIONS"))
            if os.getenv("MAX_QUESTIONS", "").strip().isdigit()
            else None
        ),
        help="Limit each tournament target to this many open questions. Useful for bounded benchmark smoke tests.",
    )
    parser.add_argument(
        "--question-shuffle-seed",
        type=int,
        default=(
            int(os.getenv("QUESTION_SHUFFLE_SEED"))
            if os.getenv("QUESTION_SHUFFLE_SEED", "").strip().isdigit()
            else None
        ),
        help="Shuffle open tournament questions before applying --max-questions.",
    )
    parser.add_argument(
        "--continue-on-question-errors",
        action="store_true",
        default=_env_bool("CONTINUE_ON_QUESTION_ERRORS", False),
        help="Forecast selected tournament questions one at a time and keep going after per-question failures.",
    )
    parser.add_argument(
        "--max-scout-questions",
        type=int,
        default=(
            int(os.getenv("MAX_SCOUT_QUESTIONS"))
            if os.getenv("MAX_SCOUT_QUESTIONS", "").strip().isdigit()
            else None
        ),
        help="Refresh mode only: maximum already-forecasted open questions to scout. Defaults to all.",
    )
    parser.add_argument(
        "--max-full-reforecasts",
        type=int,
        default=(
            int(os.getenv("MAX_FULL_REFORECASTS"))
            if os.getenv("MAX_FULL_REFORECASTS", "").strip().isdigit()
            else 5
        ),
        help="Refresh mode only: cap full reforecasts triggered by scout gates. Defaults to 5.",
    )
    parser.add_argument(
        "--experiment-mode",
        choices=["off", "random", "variant"],
        default=default_experiment_mode,
        help="Choose whether to run a random or fixed experiment variant. Default comes from EXPERIMENT_MODE/ENABLE_EXPERIMENT_VARIANTS.",
    )
    parser.add_argument(
        "--experiment-variant",
        type=int,
        default=(
            int(os.getenv("EXPERIMENT_VARIANT"))
            if os.getenv("EXPERIMENT_VARIANT", "").strip().isdigit()
            else None
        ),
        help="Fixed experiment variant id. If provided with experiment-mode=off, variant mode is used.",
    )
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=(
            int(os.getenv("EXPERIMENT_SEED"))
            if os.getenv("EXPERIMENT_SEED", "").strip().isdigit()
            else None
        ),
        help="Seed for random experiment-variant selection.",
    )
    parser.add_argument(
        "--experiment-log-dir",
        type=str,
        default=os.getenv("EXPERIMENT_LOG_DIR", "experiment_logs"),
        help="Directory for experiment manifests and JSONL forecast logs.",
    )
    parser.add_argument(
        "--allow-publish-experiments",
        action="store_true",
        default=_env_bool("ALLOW_PUBLISH_EXPERIMENTS", False),
        help="Allow random/fixed experiment variants to publish forecasts. Default is false for safety.",
    )
    args = parser.parse_args()
    run_mode: Literal[
        "tournament",
        "repredict_tournament",
        "refresh_tournament",
        "minibench",
        "benchmarking_questions",
        "metaculus_cup",
        "question_urls",
        "test_questions",
    ] = args.mode
    assert run_mode in [
        "tournament",
        "repredict_tournament",
        "refresh_tournament",
        "minibench",
        "benchmarking_questions",
        "metaculus_cup",
        "question_urls",
        "test_questions",
    ], "Invalid run mode"

    selected_variant, experiment_seed = _select_experiment_variant(
        args.experiment_mode, args.experiment_variant, args.experiment_seed
    )
    _apply_experiment_variant(selected_variant)
    SpringTemplateBot2026.apply_runtime_config_from_env()
    variant_name = selected_variant.name if selected_variant else "no_variant"
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"_{run_mode}_{variant_name}"
    )
    publish_reports = not args.practice
    if selected_variant and not args.allow_publish_experiments:
        logger.info(
            "Experiment variant selected; disabling Metaculus publishing. "
            "Pass --allow-publish-experiments to override."
        )
        publish_reports = False
    _log_startup_key_status()

    template_bot = SpringTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=True,
        publish_reports_to_metaculus=publish_reports,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        llms={
            "default": GeneralLlm(
                model=os.getenv(
                    "DEFAULT_FORECASTER_MODEL",
                    "openrouter/mistralai/mistral-large-2512",
                ),
                temperature=0.25,
                timeout=180,
                allowed_tries=2,
                max_tokens=_env_int("FORECASTER_MAX_TOKENS", 4096),
            ),
            "summarizer": GeneralLlm(
                model=os.getenv("SUMMARIZER_MODEL", "openrouter/openai/gpt-5-nano"),
                temperature=0.1,
                timeout=120,
                allowed_tries=2,
                max_tokens=_env_int("SUMMARIZER_MAX_TOKENS", 2048),
            ),
            "parser": GeneralLlm(
                model=os.getenv("PARSER_MODEL", "openrouter/openai/gpt-5-nano"),
                temperature=0,
                timeout=120,
                allowed_tries=2,
                max_tokens=_env_int("PARSER_MAX_TOKENS", 1024),
            ),
            "researcher": os.getenv("RESEARCHER_MODEL", "random"),
        },
    )

    client = MetaculusClient()
    manual_question_urls = _dedupe_preserving_order(
        _split_csv_args(
            (args.question_url or [])
            + ([os.getenv("QUESTION_URLS", "")] if os.getenv("QUESTION_URLS") else [])
        )
    )
    default_tournament_ids_by_mode: dict[str, list[str | int]] = {
        "tournament": ["summer-futureeval-2026"],
        "repredict_tournament": ["summer-futureeval-2026"],
        "refresh_tournament": ["summer-futureeval-2026"],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "benchmarking_questions": ["bot-benchmarking-question-list"],
    }
    target_tournament_ids: list[str | int] = (
        manual_question_urls
        if run_mode == "question_urls"
        else args.tournament_id
        or default_tournament_ids_by_mode.get(run_mode, ["summer-futureeval-2026"])
    )
    refresh_scout_decisions: list[RefreshScoutDecision] = []
    if run_mode in ["tournament", "minibench", "benchmarking_questions"]:
        forecast_reports = []
        for tournament_id in target_tournament_ids:
            if args.max_questions is not None:
                questions = _get_open_tournament_questions(client, tournament_id)
                eligible_questions = (
                    _filter_previously_forecasted_questions(questions)
                    if template_bot.skip_previously_forecasted_questions
                    else questions
                )
                selected_questions = _select_question_batch(
                    eligible_questions,
                    max_questions=args.max_questions,
                    shuffle_seed=args.question_shuffle_seed,
                )
                logger.info(
                    "Selected %s of %s eligible open questions from tournament %s "
                    "(%s open total, max_questions=%s, shuffle_seed=%s)",
                    len(selected_questions),
                    len(eligible_questions),
                    tournament_id,
                    len(questions),
                    args.max_questions,
                    args.question_shuffle_seed,
                )
                if args.continue_on_question_errors:
                    forecast_reports.extend(
                        _forecast_questions_resiliently(
                            template_bot, selected_questions
                        )
                    )
                else:
                    forecast_reports.extend(
                        asyncio.run(
                            template_bot.forecast_questions(
                                selected_questions, return_exceptions=True
                            )
                        )
                    )
            else:
                forecast_reports.extend(
                    asyncio.run(
                        template_bot.forecast_on_tournament(
                            tournament_id, return_exceptions=True
                        )
                    )
                )
    elif run_mode == "repredict_tournament":
        # Reforecast all currently open questions in the explicitly targeted tournaments.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = []
        for tournament_id in target_tournament_ids:
            forecast_reports.extend(
                asyncio.run(
                    template_bot.forecast_on_tournament(
                        tournament_id, return_exceptions=True
                    )
                )
            )
    elif run_mode == "refresh_tournament":
        forecast_reports, refresh_scout_decisions = asyncio.run(
            _refresh_tournament_forecasts(
                template_bot,
                client,
                target_tournament_ids,
                max_scout_questions=args.max_scout_questions,
                max_full_reforecasts=args.max_full_reforecasts,
                shuffle_seed=args.question_shuffle_seed,
                continue_on_question_errors=args.continue_on_question_errors,
            )
        )
    elif run_mode == "metaculus_cup":
        # The Metaculus cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564 or AI_2027_TOURNAMENT_ID = "ai-2027"
        # The Metaculus cup may not be initialized near the beginning of a season (i.e. January, May, September)
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "question_urls":
        if not manual_question_urls:
            raise ValueError(
                "question_urls mode requires at least one --question-url or QUESTION_URLS value."
            )
        template_bot.skip_previously_forecasted_questions = False
        questions = [client.get_question_by_url(url) for url in manual_question_urls]
        forecast_reports = (
            _forecast_questions_resiliently(template_bot, questions)
            if args.continue_on_question_errors
            else asyncio.run(
                template_bot.forecast_questions(questions, return_exceptions=True)
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
            "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            client.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    summary_error: Exception | None = None
    try:
        template_bot.log_report_summary(forecast_reports)
    except Exception as error:
        summary_error = error
        logger.error("Forecast report summary found errors: %r", error)
    _write_experiment_logs(
        forecast_reports,
        run_id=run_id,
        log_dir=Path(args.experiment_log_dir),
        mode=run_mode,
        target_tournament_ids=target_tournament_ids,
        selected_variant=selected_variant,
        experiment_seed=experiment_seed,
        publish_reports=publish_reports,
    )
    _write_refresh_scout_logs(
        refresh_scout_decisions,
        run_id=run_id,
        log_dir=Path(args.experiment_log_dir),
    )
    if summary_error is not None:
        successful_reports = [
            report
            for report in forecast_reports
            if not isinstance(report, BaseException)
        ]
        if args.continue_on_question_errors and successful_reports:
            logger.warning(
                "Continuing despite %s report summary error(s) because "
                "--continue-on-question-errors is enabled and %s forecast(s) succeeded.",
                len(forecast_reports) - len(successful_reports),
                len(successful_reports),
            )
        else:
            raise summary_error
