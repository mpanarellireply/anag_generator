import json
import os
from langchain_openai import ChatOpenAI

from src.excel_parser import read_excel, group_by_function
from src.agents.parser_agent import ParserAgent
from src.agents.generator_agent import GeneratorAgent
from src.agents.reviewer_agent import ReviewerAgent
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
    ):
        self.llm = ChatOpenAI(
            model=model,
            api_key=openai_api_key,
            temperature=0,
        )
        self.parser_agent = ParserAgent(self.llm)
        self.generator_agent = GeneratorAgent(template_dir, output_dir)
        self.reviewer_agent = ReviewerAgent(self.llm)
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, "parsed_specs.json")

    def _load_cache(self) -> list[FunctionSpec] | None:
        """Load parsed specs from cache if available."""
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            specs = [FunctionSpec(**item) for item in data]
            print(f"  Loaded {len(specs)} specs from cache: {self.cache_path}")
            return specs
        except Exception as e:
            print(f"  Warning: failed to load cache ({e}), will re-parse")
            return None

    def _save_cache(self, specs: list[FunctionSpec]):
        """Save parsed specs to cache."""
        os.makedirs(self.cache_dir, exist_ok=True)
        data = [s.model_dump() for s in specs]
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(specs)} specs to cache: {self.cache_path}")

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
    ) -> dict:
        """Run the full pipeline and return a summary report."""
        print("=" * 60)
        print("SQL Package Generator - Multi-Agent Pipeline")
        print("=" * 60)

        # Step 1: Read Excel and group by function
        print("\n[Step 1] Reading Excel file...")
        df = read_excel(excel_path)
        raw_functions = group_by_function(df)
        print(f"  Found {len(raw_functions)} unique functions in Excel")

        # Filter raw functions early if --function is specified
        if function_name:
            raw_functions = [r for r in raw_functions if r.function_name == function_name]
            if not raw_functions:
                all_names = [r.function_name for r in group_by_function(df)]
                print(f"\n  Error: Function '{function_name}' not found in Excel.")
                print(f"  Available: {', '.join(sorted(all_names)[:10])}...")
                return {"total_functions": 0, "files_generated": 0,
                        "reviews_passed": 0, "reviews_failed": 0,
                        "reviews": [], "output_dir": self.output_dir}
            print(f"  Filtered to function: {function_name}")

        # Step 2: Check cache or parse with LLM
        cached_specs = self._load_cache() if not force_parse else None
        if cached_specs and not force_parse:
            cached_map = {s.function_name: s for s in cached_specs}
            # Find which raw functions are already cached
            to_parse = [r for r in raw_functions if r.function_name not in cached_map]
            already_cached = [cached_map[r.function_name] for r in raw_functions
                             if r.function_name in cached_map]

            if to_parse:
                print(f"\n[Step 2] Parsing {len(to_parse)} new functions with LLM "
                      f"({len(already_cached)} already cached)...")
                new_specs = self.parser_agent.parse_all(to_parse)
                print(f"  Successfully parsed {len(new_specs)}/{len(to_parse)} functions")
                self._merge_into_cache(new_specs)
                specs = already_cached + new_specs
            else:
                print(f"\n[Step 2] All {len(already_cached)} functions loaded from cache")
                specs = already_cached
        else:
            print(f"\n[Step 2] Parsing {len(raw_functions)} functions with LLM...")
            specs = self.parser_agent.parse_all(raw_functions)
            print(f"  Successfully parsed {len(specs)}/{len(raw_functions)} functions")
            self._merge_into_cache(specs)

        # Step 3: Generate SQL files
        print("\n[Step 3] Generating SQL files...")
        sql_paths = self.generator_agent.generate_all(specs)
        print(f"  Generated {len(sql_paths)} SQL files in {self.output_dir}/")

        # Step 4: Review generated files
        print("\n[Step 4] Reviewing generated SQL files...")
        reviews = self.reviewer_agent.review_all(specs, sql_paths)

        # Build summary
        summary = self._build_summary(specs, sql_paths, reviews)
        self._print_summary(summary)

        return summary

    def _build_summary(
        self,
        specs: list[FunctionSpec],
        sql_paths: list[str],
        reviews: list[ReviewResult],
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
        }

    def _print_summary(self, summary: dict):
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Functions parsed:   {summary['total_functions']}")
        print(f"  SQL files generated: {summary['files_generated']}")
        print(f"  Reviews passed:     {summary['reviews_passed']}")
        print(f"  Reviews failed:     {summary['reviews_failed']}")
        print(f"  Output directory:   {summary['output_dir']}")

        if summary["reviews_failed"] > 0:
            print("\n  Failed reviews:")
            for review in summary["reviews"]:
                if review["status"] == "FAIL":
                    print(f"    - {review['function_name']}:")
                    for issue in review.get("issues", []):
                        print(f"      * {issue}")

        print("=" * 60)
