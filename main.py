import argparse
import os
import sys

from dotenv import load_dotenv

from src.orchestrator import Orchestrator


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate PL/SQL validation functions from Excel analysis"
    )
    parser.add_argument(
        "excel_path",
        help="Path to the Excel file containing the COBOL analysis",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated SQL files (default: output/)",
    )
    parser.add_argument(
        "--template-dir",
        default=".",
        help="Directory containing the template.j2 file (default: current dir)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--function",
        default=None,
        help="Process only this specific function (exact FUNZIONE.1 name, e.g. NFD_CTRL_FBROK)",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Force re-parsing from Excel even if cache exists",
    )
    parser.add_argument(
        "--cache-dir",
        default="parser_cache",
        help="Directory for parsed specs cache (default: parser_cache/)",
    )
    parser.add_argument(
        "--start-code",
        type=int,
        default=1,
        help="Starting number for sequential NCD/NED codes per function (default: 1, e.g. 1224 -> NCD01224)",
    )
    parser.add_argument(
        "--example-sql",
        default=None,
        help="Path to an example SQL file to use as style reference for the LogicAgent (optional)",
    )
    parser.add_argument(
        "--skip-logic",
        action="store_true",
        help="Skip the LogicAgent step (TODO placeholders will remain)",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip the Reviewer and Refiner steps",
    )
    parser.add_argument(
        "--skip-refine",
        action="store_true",
        help="Skip the Refiner step (a single review is still performed)",
    )
    parser.add_argument(
        "--max-refine",
        type=int,
        default=3,
        help="Maximum number of refine iterations (default: 3)",
    )
    parser.add_argument(
        "--debug-dir",
        default="debug",
        help="Directory for debug output of each pipeline phase (default: debug/)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.excel_path):
        print(f"Error: Excel file not found: {args.excel_path}")
        sys.exit(1)

    template_path = os.path.join(args.template_dir, "template.j2")
    if not os.path.exists(template_path):
        print(f"Error: Template not found: {template_path}")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Create a .env file with: OPENAI_API_KEY=sk-...")
        sys.exit(1)

    orchestrator = Orchestrator(
        openai_api_key=api_key,
        model=args.model,
        template_dir=args.template_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        debug_dir=args.debug_dir,
    )

    summary = orchestrator.run(
        args.excel_path,
        function_name=args.function,
        force_parse=args.force_parse,
        start_code=args.start_code,
        example_sql=args.example_sql,
        max_refine=args.max_refine,
        skip_logic=args.skip_logic,
        skip_review=args.skip_review,
        skip_refine=args.skip_refine,
    )

    sys.exit(0 if summary["reviews_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
