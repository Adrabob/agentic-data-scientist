# Orchestrator for an "agentic" offline data scientist pipeline.
# Handles dataset loading, profiling, planning, training, evaluation, reflection,
# and optional re-planning cycles. Designed primarily for classification tasks.
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

# Agent components and tooling used by the orchestrator
from agents.planner import create_plan
from agents.reflector import reflect, should_replan, apply_replan_strategy
from agents.memory import JSONMemory, build_feature_summary
from tools.data_profiler import run_eda, dataset_fingerprint
from tools.modelling import build_preprocessor, select_models, train_models
from tools.evaluation import evaluate_best, write_markdown_report, save_json


# Lightweight container for run metadata and parameters
@dataclass
class RunContext:
    run_id: str
    started_at: str
    data_path: str
    target: str
    output_dir: str
    seed: int
    test_size: float
    max_replans: int


logger = logging.getLogger("agentic_data_scientist")


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format (no microseconds) with Z suffix."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# Map plan step tags to the hint key they depend on. Missing steps default to
# enabled — conservative, since the orchestrator always runs the training core.
_STEP_HINT_MAP: Dict[str, str] = {
    "smote_resample": "resample_strategy",
    "class_weight": "resample_strategy",
    "drop_ids": "drop_columns",
    "cast_fix": "cast_fix_cols",
    "log_transform": "log_transform_cols",
    "target_encode": "target_encode_cols",
    "regularise_models": "plan_template",
    "escalate_models": "priority_models",
}


def _step_enabled(step_name: str, hints: Dict[str, Any]) -> bool:
    """Decide whether a named plan step should run given current hints.

    Step names from the planner may carry inline tags like
    `"build_preprocessor[drop_ids,cast_fix]"`; we only look at the base
    name before any bracket. Unknown steps default to True so callers
    can freely add new steps without plumbing here.
    """
    base = str(step_name).split("[", 1)[0].strip()
    key = _STEP_HINT_MAP.get(base)
    if key is None:
        return True

    value = hints.get(key)
    if key == "resample_strategy":
        if base == "smote_resample":
            return value == "smote"
        if base == "class_weight":
            return value == "class_weight"
    if key == "plan_template":
        return value == "regularised"
    # Default: step is enabled iff the hint has a truthy value.
    return bool(value)


def _configure_logging(output_dir: str, verbose: bool) -> logging.FileHandler:
    """Attach a per-run FileHandler to the root logger.

    We do not call `basicConfig` because it is a no-op once pytest / sklearn has
    configured handlers in-process. Instead we set the root level and add a
    StreamHandler (once) plus a FileHandler scoped to this run's output_dir.
    Returns the FileHandler so the caller can remove it after the run.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    for noisy in ("matplotlib", "PIL", "fontTools", "seaborn"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                     for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    fh = logging.FileHandler(os.path.join(output_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)
    return fh


class AgenticDataScientist:
    """
    Offline Agentic Data Scientist (classification-focused).

    Responsibilities:
    - load and profile datasets
    - create a plan (via planner)
    - build preprocessors and select candidate models
    - train and evaluate models
    - reflect on results and optionally re-plan
    - persist artefacts and update memory
    """

    def __init__(self, memory_path: str = "agent_memory.json", verbose: bool = True):
        # Verbose controls logging output
        self.verbose = verbose
        # Simple persistent memory used to remember prior runs for a dataset fingerprint
        self.memory = JSONMemory(memory_path)

        # Context and transient state populated when run() is executed
        self.ctx: Optional[RunContext] = None
        self.state: Dict[str, Any] = {}

    def log(self, msg: str) -> None:
        """Route legacy log calls through the module logger.

        Kept for backward compatibility with code that still calls `self.log`;
        new call sites should use `logger.info(...)` / `logger.debug(...)`
        directly.
        """
        logger.info(msg)

    def run(
        self,
        data_path: str,
        target: str,
        output_root: str = "outputs",
        seed: int = 42,
        test_size: float = 0.2,
        max_replans: int = 1,
    ) -> str:
        """
        Main orchestration entry point.

        Parameters:
        - data_path: path to the CSV dataset
        - target: target column name or 'auto' to infer
        - output_root: directory where outputs are stored (subdir will be created)
        - seed/test_size: training reproducibility and test split
        - max_replans: maximum number of times to re-plan and re-run

        Returns: path to the output directory for this run
        """
        # Create a unique run id and output directory for artefacts
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        output_dir = os.path.join(output_root, run_id)
        os.makedirs(output_dir, exist_ok=True)

        # Structured logging: mirror to console + per-run run.log file.
        file_handler = _configure_logging(output_dir, self.verbose)

        # Populate run context with parameters and metadata
        self.ctx = RunContext(
            run_id=run_id,
            started_at=now_iso(),
            data_path=data_path,
            target=target,
            output_dir=output_dir,
            seed=seed,
            test_size=test_size,
            max_replans=max_replans,
        )
        # Internal state used to track replanning + retry attempts
        self.state = {"replan_count": 0, "retry_count": 0}

        logger.info("Run %s started at %s", run_id, self.ctx.started_at)
        logger.debug("Params: seed=%s test_size=%s max_replans=%s",
                     seed, test_size, max_replans)

        try:
            # Load + profile the dataset via the unified EDA entry point.
            # run_eda handles header detection, column cleaning, data-quality
            # checks, target inference (when user_target is None), and imbalance
            # analysis in one call, and also writes EDA plots to the output dir.
            user_target = None if target.strip().lower() == "auto" else target
            logger.info("Loading dataset: %s", data_path)
            df, profile = run_eda(
                csv_path=data_path,
                user_target=user_target,
                output_dir=output_dir,
            )
            # Target is either what the user asked for or whatever analyze_target inferred
            self.ctx.target = profile["target"]
            logger.info("Loaded %d rows × %d cols, target=%s",
                        df.shape[0], df.shape[1], self.ctx.target)

            # Fingerprint the dataset for memory lookups
            fp = dataset_fingerprint(df, self.ctx.target)
            current_summary = build_feature_summary(profile)

            # Memory lookup strategy:
            # 1) exact fingerprint match → tag as memory_priority_exact
            # 2) otherwise, top similar dataset with similarity_score > 0.7 →
            #    tag as memory_priority_similar (planner distinguishes the two)
            prev = self.memory.get_dataset_record(fp)
            hint_source: Optional[str] = None
            if prev and prev.get("best_model"):
                hint_source = "exact"
                logger.info("Memory hit (exact): previously best=%s for fp=%s",
                            prev.get("best_model"), fp)
            else:
                similar = self.memory.find_similar_datasets(current_summary, top_k=3)
                if similar and similar[0].get("similarity_score", 0.0) > 0.7:
                    prev = dict(similar[0])
                    hint_source = "similar"
                    logger.info(
                        "Memory hit (similar): best=%s score=%.3f fp=%s",
                        prev.get("best_model"),
                        prev.get("similarity_score", 0.0),
                        prev.get("fingerprint"),
                    )
                else:
                    prev = None
            if prev is not None:
                prev = dict(prev)
                prev["hint_source"] = hint_source

            # Create an initial plan informed by the profile and optional memory hint.
            # `plan` is a dict with keys: steps, rationale, hints.
            plan = create_plan(profile, memory_hint=prev)
            logger.info("Plan: %s", plan["steps"])

            # Execution loop: trains and evaluates, then optionally replans
            while True:
                # Hints emitted by the planner drive preprocessor/model choices.
                hints = plan.get("hints", {}) if isinstance(plan, dict) else {}
                drop_cols = hints.get("drop_columns") or None
                priority_models = hints.get("priority_models") or None

                # Log per-step enable/skip decisions before executing the loop body.
                steps = plan.get("steps", []) if isinstance(plan, dict) else list(plan or [])
                for step in steps:
                    enabled = _step_enabled(step, hints)
                    logger.debug("Step %s: %s", step, "enabled" if enabled else "skipped")

                # Build preprocessing pipeline tailored to the profile
                preprocessor = build_preprocessor(profile, drop_columns=drop_cols)
                # Choose candidate models to try based on the profile
                candidates = select_models(
                    profile, seed=self.ctx.seed, priority_models=priority_models
                )
                logger.info("Candidate models: %s", [n for n, _ in candidates])

                # Train + evaluate. If the whole training stage fails (e.g.
                # preprocessor blows up on unexpected dtypes) fall back to a
                # minimal LogReg+Dummy config once before giving up.
                try:
                    results = train_models(
                        df=df,
                        target=self.ctx.target,
                        preprocessor=preprocessor,
                        candidates=candidates,
                        seed=self.ctx.seed,
                        test_size=self.ctx.test_size,
                        output_dir=self.ctx.output_dir,
                        verbose=self.verbose,
                        cv_folds=int(hints.get("cv_folds", 5)),
                    )
                    eval_payload = evaluate_best(
                        results, output_dir=self.ctx.output_dir
                    )
                except Exception as exc:
                    logger.exception("Train/evaluate failed: %s", exc)
                    if self.state["retry_count"] < 1:
                        self.state["retry_count"] += 1
                        hints["priority_models"] = ["LogisticRegression", "DummyMostFrequent"]
                        # Drop categorical columns too on retry — cheap way to
                        # sidestep one-hot expansion issues.
                        extra_drop = list(
                            profile.get("feature_types", {}).get("categorical", [])
                        )
                        merged_drop = list({*(hints.get("drop_columns") or []), *extra_drop})
                        hints["drop_columns"] = merged_drop
                        plan["hints"] = hints
                        logger.warning(
                            "Retrying with minimal config (attempt %d)",
                            self.state["retry_count"],
                        )
                        continue
                    save_json(
                        os.path.join(self.ctx.output_dir, "failed_run.json"),
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "retry_count": self.state["retry_count"],
                            "hints": hints,
                        },
                    )
                    raise

                # Reflect on the evaluation in the context of the dataset profile.
                best_predictions = {
                    "y_test": results["best"].get("y_test"),
                    "y_pred": results["best"].get("y_pred"),
                }
                reflection = reflect(
                    dataset_profile=profile,
                    evaluation=eval_payload["best_metrics"],
                    all_metrics=eval_payload["all_metrics"],
                    best_predictions=best_predictions,
                )

                # Persist core run artefacts for later review
                save_json(os.path.join(self.ctx.output_dir, "eda_summary.json"), profile)
                save_json(os.path.join(self.ctx.output_dir, "plan.json"), plan)
                save_json(os.path.join(self.ctx.output_dir, "metrics.json"), eval_payload)
                save_json(os.path.join(self.ctx.output_dir, "reflection.json"), reflection)

                # Generate a human-readable markdown report summarising the run
                write_markdown_report(
                    out_path=os.path.join(self.ctx.output_dir, "report.md"),
                    ctx=self.ctx,
                    fingerprint=fp,
                    dataset_profile=profile,
                    plan=plan,
                    eval_payload=eval_payload,
                    reflection=reflection,
                )

                # Update the memory store with a richer experience record so
                # future runs on *similar* (not just identical) datasets can
                # benefit. The schema is intentionally small — just enough for
                # the planner's memory_priority branch + the similarity metric.
                top_three = [
                    (m.get("model"), float(m.get("balanced_accuracy", 0.0)))
                    for m in eval_payload.get("all_metrics", [])[:3]
                    if m.get("balanced_accuracy") is not None
                ]
                self.memory.upsert_dataset_record(fp, {
                    "last_seen": now_iso(),
                    "target": self.ctx.target,
                    "shape": profile["shape"],
                    "best_model": eval_payload["best_metrics"]["model"],
                    "best_metrics": eval_payload["best_metrics"],
                    "feature_summary": current_summary,
                    "applied_hints": dict(plan.get("hints", {})) if isinstance(plan, dict) else {},
                    "top_3_models": top_three,
                    "reflection_issues": reflection.get("structured_issues", []),
                })

                # Decide whether the agent should attempt to re-plan and re-run
                if not should_replan(reflection):
                    break

                if self.state["replan_count"] >= self.ctx.max_replans:
                    logger.info("Replan suggested, but max_replans reached. Stopping.")
                    break

                self.state["replan_count"] += 1
                logger.info("Replanning attempt #%d...", self.state["replan_count"])

                # apply_replan_strategy returns an updated (plan, profile) pair
                plan, profile = apply_replan_strategy(plan, profile, reflection)

            logger.info("Done. Outputs saved to: %s", self.ctx.output_dir)
            return self.ctx.output_dir
        finally:
            logging.getLogger().removeHandler(file_handler)
            file_handler.close()
