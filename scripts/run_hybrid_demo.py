from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the hybrid TCN + TimeOmni pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to hybrid config YAML.")
    parser.add_argument("--engine-id", type=int, action="append", default=None, help="Specific engine ID to analyze.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of engines to process.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional custom output path for reports.")
    parser.add_argument("--prompts-jsonl", type=str, default=None, help="Optional custom output path for exported prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from hybrid_rul.paths import ensure_dir, load_yaml, resolve_path
    from hybrid_rul.pipelines.hybrid_pipeline import HybridPipeline

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    pipeline = HybridPipeline(config=config, config_path=config_path)
    result = pipeline.run(engine_ids=args.engine_id, limit=args.limit)

    output_dir = resolve_path(config_path.parent, config["paths"]["output_dir"])
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = Path(args.output_json).resolve() if args.output_json else output_dir / f"hybrid_reports_{timestamp}.json"
    ensure_dir(report_path.parent)
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(result["reports"], fp, indent=2, ensure_ascii=False)

    prompts = result["prompts"]
    prompt_path = None
    if prompts and config["reasoning"].get("export_prompts", True):
        prompt_path = (
            Path(args.prompts_jsonl).resolve()
            if args.prompts_jsonl
            else output_dir / f"timeomni_prompts_{timestamp}.jsonl"
        )
        ensure_dir(prompt_path.parent)
        with prompt_path.open("w", encoding="utf-8") as fp:
            for item in prompts:
                fp.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Hybrid report saved to: {report_path}")
    if prompt_path is not None:
        print(f"Prompt export saved to: {prompt_path}")


if __name__ == "__main__":
    main()
