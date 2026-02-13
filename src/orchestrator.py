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
    """Coordinates the Parser -> Generator -> Reviewer pipeline."""

    def __init__(
        self,
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        template_dir: str = ".",
        output_dir: str = "output",
        cache_dir: str = "cache",
        debug_dir: str = "debug",
    ):
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0,
        )
        self.parser_agent = ParserAgent(self.llm)
        self.generator_agent = GeneratorAgent(template_dir, output_dir)
        self.reviewer_agent = ReviewerAgent(self.llm)
        self.refiner_agent = RefinerAgent(self.llm)
        self.translator_agent = TranslatorAgent(self.llm)
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "parsed_specs.json")
        self.debug_dir = debug_dir
        self._run_timestamp: str | None = None

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

    def _read_sql(self, sql_path: str) -> str:
        """Read SQL file content for debug snapshots."""
        with open(sql_path, "r", encoding="utf-8") as f:
            return f.read()

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

    def run(
        self,
        excel_path: str,
        function_name: str | None = None,
        force_parse: bool = False,
        start_code: int = 1,
        example_sql: str | None = None,
        max_refine: int = 3,
        skip_logic: bool = False,
        skip_review: bool = False,
        skip_refine: bool = False,
    ) -> dict:
        """Run the full pipeline and return a summary report."""
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_start = time.time()
        timings = {}
        logger.info("=" * 60)
        logger.info("SQL Package Generator - Multi-Agent Pipeline O.o.O.o.O.o.O.o")
        logger.info("=" * 60)

        # Step 1: Read Excel and group by function
        logger.info("[Step 1] Reading Excel file...")
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
        logger.debug("Excel Parser time: %.2fs", timings["Excel Parser"])

        # Step 2: Check cache or parse with LLM
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
                new_specs = self.parser_agent.parse_all(to_parse, start_code=start_code)
                logger.info("Successfully parsed %d/%d functions", len(new_specs), len(to_parse))
                self._merge_into_cache(new_specs)
                specs = already_cached + new_specs
            else:
                logger.info("[Step 2] All %d functions loaded from cache", len(already_cached))
                logger.warning("IF input xlsx file has changed, please use --force-parse flag")
                specs = already_cached
        else:
            logger.info("[Step 2] Parsing %d functions with LLM...", len(raw_functions))
            specs = self.parser_agent.parse_all(raw_functions, start_code=start_code)
            logger.info("Successfully parsed %d/%d functions", len(specs), len(raw_functions))
            self._merge_into_cache(specs)
        timings["Parser Agent"] = time.time() - t0
        logger.debug("Parser Agent time: %.2fs", timings["Parser Agent"])

        # Debug: save parser output
        for spec in specs:
            self._save_phase_debug(
                "01_parser", spec.function_name,
                self.parser_agent.last_responses,
                extra={"spec.json": json.dumps(spec.model_dump(), indent=2, ensure_ascii=False)},
            )

        # Step 3: Generate SQL files
        logger.info("[Step 3] Generating SQL files...")
        t0 = time.time()
        sql_paths = self.generator_agent.generate_all(specs)
        logger.info("Generated %d SQL files in %s/", len(sql_paths), self.output_dir)
        timings["Generator Agent"] = time.time() - t0
        logger.debug("Generator Agent time: %.2fs", timings["Generator Agent"])

        # Step 4: Complete TODO logic
        if not skip_logic:
            logger.info("[Step 4] Completing TODO logic with LogicAgent...")
            t0 = time.time()
            # Snapshot SQL before logic
            sql_before_logic = {os.path.splitext(os.path.basename(p))[0]: self._read_sql(p) for p in sql_paths}
            logic_agent = LogicAgent(self.llm, example_sql_path=example_sql)
            logic_agent.complete_all(specs, sql_paths)
            # Debug: save logic agent output
            for p in sql_paths:
                fname = os.path.splitext(os.path.basename(p))[0]
                self._save_phase_debug(
                    "02_logic", fname, logic_agent.last_responses,
                    before_sql=sql_before_logic.get(fname),
                    after_sql=self._read_sql(p),
                )
            timings["Logic Agent"] = time.time() - t0
            logger.debug("Logic Agent time: %.2fs", timings["Logic Agent"])
        else:
            logger.info("[Step 4] Skipping LogicAgent (--skip-logic)")

        # Step 5: Review generated files
        reviews = []
        if not skip_review:
            logger.info("[Step 5] Reviewing generated SQL files...")
            t0 = time.time()
            reviews = self.reviewer_agent.review_all(specs, sql_paths)
            # Debug: save reviewer output
            for r in reviews:
                self._save_phase_debug(
                    "03_reviewer", r.function_name,
                    self.reviewer_agent.last_responses,
                    extra={"result.json": json.dumps(r.model_dump(), indent=2, ensure_ascii=False)},
                )
            timings["Reviewer Agent"] = time.time() - t0
            logger.debug("Reviewer Agent time: %.2fs", timings["Reviewer Agent"])

            # Step 6: Refine loop (max 3 iterations)
            if not skip_refine:
                t0 = time.time()
                review_map = {r.function_name: r for r in reviews}
                for iteration in range(max_refine):
                    failed = [r for r in review_map.values() if r.status == "FAIL"]
                    if not failed:
                        break
                    logger.info("[Step 6] Refine iteration %d/%d (%d failed files)...",
                                iteration + 1, max_refine, len(failed))
                    # Snapshot SQL before refine
                    path_map = {os.path.splitext(os.path.basename(p))[0]: p for p in sql_paths}
                    sql_before_refine = {
                        fname: self._read_sql(path_map[fname])
                        for fname in [r.function_name for r in failed]
                        if fname in path_map
                    }
                    refined_paths = self.refiner_agent.refine_all(specs, sql_paths, failed)
                    # Debug: save refiner output per iteration
                    for p in refined_paths:
                        fname = os.path.splitext(os.path.basename(p))[0]
                        self._save_phase_debug(
                            f"04_refiner_iter{iteration + 1}", fname,
                            self.refiner_agent.last_responses,
                            before_sql=sql_before_refine.get(fname),
                            after_sql=self._read_sql(p),
                        )
                    if not refined_paths:
                        logger.info("No files were refined, stopping.")
                        break
                    logger.info("Re-reviewing %d refined files...", len(refined_paths))
                    re_reviews = self.reviewer_agent.review_all(specs, refined_paths)
                    # Debug: save re-review output
                    for r in re_reviews:
                        self._save_phase_debug(
                            f"04_reviewer_iter{iteration + 1}", r.function_name,
                            self.reviewer_agent.last_responses,
                            extra={"result.json": json.dumps(r.model_dump(), indent=2, ensure_ascii=False)},
                        )
                    for r in re_reviews:
                        review_map[r.function_name] = r
                reviews = list(review_map.values())
                timings["Refiner Agent"] = time.time() - t0
                logger.debug("Refiner Agent time: %.2fs", timings["Refiner Agent"])
            else:
                logger.info("[Step 6] Skipping Refiner (--skip-refine)")
        else:
            logger.info("[Step 5] Skipping Review and Refine (--skip-review)")

        # Step 7: Translate comments to Italian
        logger.info("[Step 7] Translating comments to Italian...")
        t0 = time.time()
        # Snapshot SQL before translation
        sql_before_translate = {
            os.path.splitext(os.path.basename(p))[0]: self._read_sql(p) for p in sql_paths
        }
        self.translator_agent.translate_all(sql_paths)
        # Debug: save translator output
        for p in sql_paths:
            fname = os.path.splitext(os.path.basename(p))[0]
            self._save_phase_debug(
                "05_translator", fname,
                self.translator_agent.last_responses,
                before_sql=sql_before_translate.get(fname),
                after_sql=self._read_sql(p),
            )
        timings["Translator Agent"] = time.time() - t0
        logger.debug("Translator Agent time: %.2fs", timings["Translator Agent"])

        # Debug: save final output
        for p in sql_paths:
            fname = os.path.splitext(os.path.basename(p))[0]
            self._save_debug(fname, "06_final_output.sql", self._read_sql(p))

        timings["Total"] = time.time() - pipeline_start

        # Build summary
        summary = self._build_summary(specs, sql_paths, reviews, timings)
        self._print_summary(summary)

        return summary

    def _build_summary(
        self,
        specs: list[FunctionSpec],
        sql_paths: list[str],
        reviews: list[ReviewResult],
        timings: dict[str, float] | None = None,
    ) -> dict:
        passed = sum(1 for r in reviews if r.status == "PASS")
        failed = sum(1 for r in reviews if r.status == "FAIL")

        return {
            "total_functions": len(specs),
            "files_generated": len(sql_paths),
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
