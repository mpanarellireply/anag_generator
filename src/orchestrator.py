import json
import logging
import os
import time
from datetime import datetime
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

from src.excel_parser import read_excel, group_by_function
from src.agents.parser_agent import ParserAgent
from src.agents.generator_agent import GeneratorAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.refiner_agent import RefinerAgent
from src.agents.logic_agent import LogicAgent
from src.agents.translator_agent import TranslatorAgent
from src.models import FunctionSpec, ReviewResult


class Orchestrator:
    """Coordinates the Parser -> Generator -> Reviewer pipeline.

    Processes functions one at a time: bulk parse first, then per-function
    Generate -> Logic -> Review -> Refine -> Translate.
    """

    GLOBAL_PHASES = [
        "Excel Parser",
        "Parser Agent",
    ]

    PER_FUNCTION_PHASES = [
        "Generator Agent",
        "Logic Agent",
        "Reviewer Agent",
        "Refiner Agent",
        "Translator Agent",
    ]

    PHASES = GLOBAL_PHASES + PER_FUNCTION_PHASES

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        template_dir: str = ".",
        output_dir: str = "output",
        cache_dir: str = "cache",
        debug_dir: str = "debug",
        progress_callback=None,
    ):
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0,
        )
        self.parser_agent = ParserAgent(self.llm)
        self.generator_agent = GeneratorAgent(template_dir)
        self.reviewer_agent = ReviewerAgent(self.llm)
        self.refiner_agent = RefinerAgent(self.llm)
        self.translator_agent = TranslatorAgent(self.llm)
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "parsed_specs.json")
        self.debug_dir = debug_dir
        self._run_timestamp: str | None = None
        self._progress_callback = progress_callback

    def _report_progress(
        self,
        phase: str,
        status: str,
        elapsed: float | None = None,
        function_name: str | None = None,
        meta: dict | None = None,
    ):
        """Report phase progress via callback.

        status: 'running' | 'done' | 'skipped' | 'error'
        function_name: None for global phases, str for per-function phases.
        meta: extra data for sentinel events.
        """
        if self._progress_callback:
            self._progress_callback(phase, status, elapsed, function_name, meta)

    def _load_cache(self) -> list[FunctionSpec] | None:
        """Load parsed specs from cache if available."""
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            specs = [FunctionSpec(**item) for item in data]
            logger.debug("Loaded %d specs from cache: %s", len(specs), self.cache_path)
            return specs
        except Exception as e:
            logger.warning("Failed to load cache (%s), will re-parse", e)
            return None

    def _save_cache(self, specs: list[FunctionSpec]):
        """Save parsed specs to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        data = [s.model_dump() for s in specs]
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Saved %d specs to cache: %s", len(specs), self.cache_path)

    def _debug_path(self, function_name: str, filename: str) -> str:
        """Return the path to a debug file for a given function."""
        folder_name = f"{function_name}_{self._run_timestamp}"
        func_dir = os.path.join(self.debug_dir, folder_name)
        os.makedirs(func_dir, exist_ok=True)
        return os.path.join(func_dir, filename)

    def _save_debug(self, function_name: str, filename: str, content: str):
        """Save debug content to a file."""
        path = self._debug_path(function_name, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _save_phase_debug(
        self,
        phase_prefix: str,
        function_name: str,
        responses: dict[str, str],
        before_sql: str | None = None,
        after_sql: str | None = None,
        extra: dict[str, str] | None = None,
    ):
        """Save debug files for a pipeline phase."""
        if function_name in responses:
            self._save_debug(function_name, f"{phase_prefix}_response.txt", responses[function_name])
        if before_sql is not None:
            self._save_debug(function_name, f"{phase_prefix}_before.sql", before_sql)
        if after_sql is not None:
            self._save_debug(function_name, f"{phase_prefix}_after.sql", after_sql)
        if extra:
            for suffix, content in extra.items():
                self._save_debug(function_name, f"{phase_prefix}_{suffix}", content)

    def _merge_into_cache(self, new_specs: list[FunctionSpec]):
        """Merge newly parsed specs into existing cache (update or append)."""
        cached = self._load_cache() or []
        cache_map = {s.function_name: s for s in cached}
        for spec in new_specs:
            cache_map[spec.function_name] = spec
        all_specs = list(cache_map.values())
        self._save_cache(all_specs)

    def _save_single_sql(self, function_name: str, sql_content: str) -> str:
        """Write a single SQL file to the output directory. Returns the output path."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{function_name}.sql")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(sql_content)
        return output_path

    def _run_function_pipeline(
        self,
        spec: FunctionSpec,
        logic_agent: LogicAgent | None,
        max_refine: int = 3,
        skip_logic: bool = False,
        skip_review: bool = False,
        skip_refine: bool = False,
    ) -> tuple[str | None, ReviewResult | None, dict[str, float]]:
        """Run pipeline phases for a single function.

        Returns: (final_sql or None, review_result or None, timings dict)
        If Generator fails, returns (None, None, timings) — function is skipped.
        """
        fname = spec.function_name
        timings: dict[str, float] = {}
        sql: str | None = None
        review: ReviewResult | None = None
        step_success = True

        # --- Generator ---
        self._report_progress("Generator Agent", "running", function_name=fname)
        t0 = time.time()
        try:
            logger.info("[Generator] Generating SQL for function: %s", fname)
            sql = self.generator_agent.generate(spec)
        except Exception as e:
            logger.error("[Generator] FATAL for %s: %s", fname, e, exc_info=True)
            timings["Generator Agent"] = time.time() - t0
            self._report_progress("Generator Agent", "error", timings["Generator Agent"], function_name=fname)
            return None, None, timings
        timings["Generator Agent"] = time.time() - t0
        self._report_progress("Generator Agent", "done", timings["Generator Agent"], function_name=fname)

        # --- Logic ---
        if not skip_logic and logic_agent is not None:
            self._report_progress("Logic Agent", "running", function_name=fname)
            t0 = time.time()
            try:
                logger.info("[Logic] Completing SQL for function: %s", fname)
                sql_before = sql
                sql = logic_agent.complete(spec, sql)
                self._save_phase_debug(
                    "02_logic", fname, logic_agent.last_responses,
                    before_sql=sql_before, after_sql=sql,
                )
            except Exception as e:
                logger.error("[Logic] ERROR for %s: %s -- continuing with template SQL", fname, e, exc_info=True)
                step_success = False
            finally:
                timings["Logic Agent"] = time.time() - t0
                self._report_progress("Logic Agent", "done" if step_success else "error", timings["Logic Agent"], function_name=fname)
                step_success = True
        else:
            self._report_progress("Logic Agent", "skipped", function_name=fname)

        # --- Reviewer ---
        if not skip_review:
            self._report_progress("Reviewer Agent", "running", function_name=fname)
            t0 = time.time()
            try:
                logger.info("[Reviewer] Reviewing SQL for function: %s", fname)
                review = self.reviewer_agent.review(spec, sql)
                self._save_phase_debug(
                    "03_reviewer", fname, self.reviewer_agent.last_responses,
                    extra={"result.json": json.dumps(review.model_dump(), indent=2, ensure_ascii=False)},
                )
            except Exception as e:
                logger.error("[Reviewer] ERROR for %s: %s", fname, e, exc_info=True)
                step_success = False
            finally:
                timings["Reviewer Agent"] = time.time() - t0
                self._report_progress("Reviewer Agent", "done" if step_success else "error", timings["Reviewer Agent"], function_name=fname)
                step_success = True
        else:
            self._report_progress("Reviewer Agent", "skipped", function_name=fname)

        # --- Refiner ---
        if not skip_refine:
            self._report_progress("Refiner Agent", "running", function_name=fname)
            t0 = time.time()

            if review and review.status == "FAIL":
                # Iterative refine with reviewer feedback
                for iteration in range(max_refine):
                    if review.status != "FAIL":
                        step_success = True
                        break
                    try:
                        logger.info("[Refiner] Refining SQL for function: %s, iteration: %d", fname, iteration + 1)
                        sql_before = sql
                        sql = self.refiner_agent.refine(spec, sql, review)
                        self._save_phase_debug(
                            f"04_refiner_iter{iteration + 1}", fname,
                            self.refiner_agent.last_responses,
                            before_sql=sql_before, after_sql=sql,
                        )
                        # Re-review
                        logger.info("[Refiner->Review] Re-reviewing SQL for function: %s, iteration: %d", fname, iteration + 1)
                        review = self.reviewer_agent.review(spec, sql)
                        self._save_phase_debug(
                            f"04_reviewer_iter{iteration + 1}", fname,
                            self.reviewer_agent.last_responses,
                            extra={"result.json": json.dumps(review.model_dump(), indent=2, ensure_ascii=False)},
                        )
                    except Exception as e:
                        logger.error("[Refiner] ERROR iter %d for %s: %s", iteration + 1, fname, e, exc_info=True)
                        step_success = False
                        break
            elif not review and not skip_review:
                # Standalone refine (reviewer failed, no review available)
                try:
                    logger.info("[Refiner] Refining SQL for function: %s (standalone)", fname)
                    sql_before = sql
                    sql = self.refiner_agent.refine_standalone(spec, sql)
                    self._save_phase_debug(
                        "04_refiner_standalone", fname,
                        self.refiner_agent.last_responses,
                        before_sql=sql_before, after_sql=sql,
                    )
                except Exception as e:
                    logger.error("[Refiner] ERROR standalone for %s: %s", fname, e, exc_info=True)
                    step_success = False
            # else: review.status == "PASS" or skip_review -> nothing to refine

            timings["Refiner Agent"] = time.time() - t0
            self._report_progress("Refiner Agent", "done" if step_success else "error", timings["Refiner Agent"], function_name=fname)
        else:
            self._report_progress("Refiner Agent", "skipped", function_name=fname)

        step_success = True
        # --- Translator ---
        self._report_progress("Translator Agent", "running", function_name=fname)
        t0 = time.time()
        try:
            logger.info("[Translator] Translating SQL for function: %s", fname)
            sql_before = sql
            sql = self.translator_agent.translate(sql, function_name=fname)
            self._save_phase_debug(
                "05_translator", fname, self.translator_agent.last_responses,
                before_sql=sql_before, after_sql=sql,
            )
        except Exception as e:
            logger.error("[Translator] ERROR for %s: %s -- using untranslated SQL", fname, e, exc_info=True)
            step_success = False
        finally:
            timings["Translator Agent"] = time.time() - t0
            self._report_progress("Translator Agent", "done" if step_success else "error", timings["Translator Agent"], function_name=fname)
            step_success = True

        return sql, review, timings

    def run(
        self,
        excel_path: str,
        function_name: str | None = None,
        force_parse: bool = False,
        vertical_code: str = "",
        start_code: int = 1,
        example_sql: str | None = None,
        max_refine: int = 3,
        skip_logic: bool = False,
        skip_review: bool = False,
        skip_refine: bool = False,
    ) -> dict:
        """Run the full pipeline and return a summary report.

        Stage A (bulk): Excel Parser + Parser Agent for all functions.
        Stage B (per-function): Generate -> Logic -> Review -> Refine -> Translate one at a time.
        """
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_start = time.time()
        timings: dict[str, float] = {}
        self.saved_files: list[str] = []
        logger.info("=" * 60)
        logger.info("SQL Package Generator - Multi-Agent Pipeline O.o.O.o.O.o.O.o")
        logger.info("=" * 60)

        # ========== STAGE A: Bulk Parse ==========

        # Step 1: Read Excel and group by function
        logger.info("[Step 1] Reading Excel file...")
        self._report_progress("Excel Parser", "running")
        t0 = time.time()
        df = read_excel(excel_path)
        raw_functions = group_by_function(df)
        logger.info("Found %d unique functions in Excel", len(raw_functions))

        # Filter raw functions early if --function is specified
        if function_name:
            raw_functions = [r for r in raw_functions if r.function_name == function_name]
            if not raw_functions:
                all_names = [r.function_name for r in group_by_function(df)]
                logger.error("Function '%s' not found in Excel.", function_name)
                logger.info("Available: %s...", ", ".join(sorted(all_names)[:10]))
                return {"total_functions": 0, "files_generated": 0,
                        "reviews_passed": 0, "reviews_failed": 0,
                        "reviews": [], "output_dir": self.output_dir}
            logger.info("Filtered to function: %s", function_name)
        timings["Excel Parser"] = time.time() - t0
        self._report_progress("Excel Parser", "done", timings["Excel Parser"])
        logger.debug("Excel Parser time: %.2fs", timings["Excel Parser"])

        # Step 2: Check cache or parse with LLM
        self._report_progress("Parser Agent", "running")
        t0 = time.time()
        cached_specs = self._load_cache() if not force_parse else None
        if cached_specs and not force_parse:
            cached_map = {s.function_name: s for s in cached_specs}
            # Find which raw functions are already cached
            to_parse = [r for r in raw_functions if r.function_name not in cached_map]
            already_cached = [cached_map[r.function_name] for r in raw_functions
                             if r.function_name in cached_map]

            if to_parse:
                logger.info("[Step 2] Parsing %d new functions with LLM (%d already cached)...",
                            len(to_parse), len(already_cached))
                new_specs = self.parser_agent.parse_all(to_parse, vertical_code=vertical_code, start_code=start_code)
                logger.info("Successfully parsed %d/%d functions", len(new_specs), len(to_parse))
                self._merge_into_cache(new_specs)
                specs = already_cached + new_specs
            else:
                logger.info("[Step 2] All %d functions loaded from cache", len(already_cached))
                logger.warning("IF input xlsx file has changed, please use --force-parse flag")
                specs = already_cached
        else:
            logger.info("[Step 2] Parsing %d functions with LLM...", len(raw_functions))
            specs = self.parser_agent.parse_all(raw_functions, vertical_code=vertical_code, start_code=start_code)
            logger.info("Successfully parsed %d/%d functions", len(specs), len(raw_functions))
            self._merge_into_cache(specs)
        timings["Parser Agent"] = time.time() - t0
        self._report_progress("Parser Agent", "done", timings["Parser Agent"])
        logger.debug("Parser Agent time: %.2fs", timings["Parser Agent"])

        # Debug: save parser output
        for spec in specs:
            self._save_phase_debug(
                "01_parser", spec.function_name,
                self.parser_agent.last_responses,
                extra={"spec.json": json.dumps(spec.model_dump(), indent=2, ensure_ascii=False)},
            )

        # Notify discovered function names (for web UI to render function list)
        discovered_names = [s.function_name for s in specs]
        self._report_progress(
            "__functions_discovered__", "done",
            meta={"function_names": discovered_names},
        )

        # ========== STAGE B: Per-Function Pipeline ==========

        # Create LogicAgent once (avoids re-reading example SQL per function)
        logic_agent = LogicAgent(self.llm, example_sql_path=example_sql) if not skip_logic else None

        all_reviews: list[ReviewResult] = []
        sql_map: dict[str, str] = {}

        for i, spec in enumerate(specs):
            fname = spec.function_name
            logger.info("=" * 40)
            logger.info("Processing function %d/%d: %s", i + 1, len(specs), fname)
            logger.info("=" * 40)

            sql, review, _ = self._run_function_pipeline(
                spec,
                logic_agent=logic_agent,
                max_refine=max_refine,
                skip_logic=skip_logic,
                skip_review=skip_review,
                skip_refine=skip_refine,
            )

            if sql is None:
                logger.error("Function %s FAILED entirely -- skipping", fname)
                self._report_progress(
                    "__function_error__", "error",
                    function_name=fname,
                    meta={"error": f"Generation failed for {fname}"},
                )
                continue

            # Save immediately
            output_path = self._save_single_sql(fname, sql)
            self.saved_files.append(output_path)
            sql_map[fname] = sql

            if review:
                all_reviews.append(review)

            # Save final debug
            self._save_debug(fname, f"{fname}_{self._run_timestamp}.sql", sql)

            # Notify that this function is fully done
            self._report_progress(
                "__function_done__", "done",
                function_name=fname,
                meta={"output_file": output_path},
            )

        timings["Total"] = time.time() - pipeline_start

        # Build summary
        summary = self._build_summary(specs, sql_map, all_reviews, timings)
        self._print_summary(summary)

        return summary

    def _build_summary(
        self,
        specs: list[FunctionSpec],
        sql_map: dict[str, str],
        reviews: list[ReviewResult],
        timings: dict[str, float] | None = None,
    ) -> dict:
        passed = sum(1 for r in reviews if r.status == "PASS")
        failed = sum(1 for r in reviews if r.status == "FAIL")

        return {
            "total_functions": len(specs),
            "files_generated": len(sql_map),
            "reviews_passed": passed,
            "reviews_failed": failed,
            "reviews": [r.model_dump() for r in reviews],
            "output_dir": self.output_dir,
            "timings": timings or {},
        }

    def _print_summary(self, summary: dict):
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info("Functions parsed:    %d", summary["total_functions"])
        logger.info("SQL files generated: %d", summary["files_generated"])
        logger.info("Reviews passed:      %d", summary["reviews_passed"])
        logger.info("Reviews failed:      %d", summary["reviews_failed"])
        logger.info("Output directory:    %s", summary["output_dir"])

        if summary["reviews_failed"] > 0:
            logger.warning("Failed reviews:")
            for review in summary["reviews"]:
                if review["status"] == "FAIL":
                    logger.warning("  - %s:", review["function_name"])
                    for issue in review.get("issues", []):
                        logger.warning("    * %s", issue)

        timings = summary.get("timings", {})
        if timings:
            logger.info("Execution times:")
            for name, elapsed in timings.items():
                if name == "Total":
                    continue
                logger.info("  %-20s %7.1fs", name, elapsed)
            if "Total" in timings:
                logger.info("  %-20s %7.1fs", "Total", timings["Total"])

        logger.info("=" * 60)
