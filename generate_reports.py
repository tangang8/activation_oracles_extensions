from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from report_pages import save_oracle_rollouts_html, save_rollouts_html


def _load_json(path: Path) -> Any:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _is_target_judged_entry(entry: dict[str, Any]) -> bool:
    return "target_response" in entry and "compliance" in entry


def _is_oracle_entry(entry: dict[str, Any]) -> bool:
    return "oracle_response" in entry


def _infer_report_type(cache_path: Path, payload: Any) -> str:
    path_text = str(cache_path)
    if "target_rollouts_judged" in path_text:
        return "target"
    if "oracle_rollouts_judged" in path_text or "oracle_prompt_rollouts" in path_text:
        return "oracle"
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            if _is_target_judged_entry(first):
                return "target"
            if _is_oracle_entry(first):
                return "oracle"
    raise ValueError(
        "Could not infer report type. Pass --report-type target|oracle explicitly."
    )


def _target_compliance_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(entries)
    scored = 0
    compliant = 0
    partial = 0
    for entry in entries:
        comp = entry.get("compliance", {})
        score = comp.get("score") if isinstance(comp, dict) else None
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            scored += 1
            if score > 1:
                compliant += 1
            if score in (2, 3):
                partial += 1
    if scored == 0:
        return {
            "total": total,
            "compliant": 0,
            "compliance_rate": 0.0,
            "partial_compliant": 0,
            "partial_compliance_rate": 0.0,
            "scored": 0,
        }
    return {
        "total": total,
        "compliant": compliant,
        "compliance_rate": compliant / scored,
        "partial_compliant": partial,
        "partial_compliance_rate": partial / scored,
        "scored": scored,
    }


def _default_output_path(cache_path: Path, report_type: str, output_dir: Path) -> Path:
    stem = cache_path.stem
    suffix = "target" if report_type == "target" else "oracle"
    return output_dir / f"{stem}_{suffix}_report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HTML reports from cached rollout files.")
    parser.add_argument("--cache-path", required=True, help="Path to cache JSON file.")
    parser.add_argument(
        "--report-type",
        choices=("auto", "target", "oracle"),
        default="auto",
        help="Which report to generate. Default: auto.",
    )
    parser.add_argument(
        "--output-dir",
        default="website",
        help="Directory to write the HTML report into (default: website).",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="Qwen/Qwen3-8B",
        help="Tokenizer model used for oracle token visualization (default: Qwen/Qwen3-8B).",
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_path).resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    payload = _load_json(cache_path)
    if not isinstance(payload, list):
        raise ValueError("Cache payload must be a JSON list of entries.")

    report_type = args.report_type
    if report_type == "auto":
        report_type = _infer_report_type(cache_path, payload)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = _default_output_path(cache_path, report_type, output_dir)

    if report_type == "target":
        if payload and isinstance(payload[0], dict) and not _is_target_judged_entry(payload[0]):
            raise ValueError("Target report requires target judged entries with compliance fields.")
        summary = _target_compliance_summary(payload)
        final = save_rollouts_html(
            rollout_entries=payload,
            compliance_results=summary,
            output_path=str(out_path),
        )
    else:
        if payload and isinstance(payload[0], dict) and not _is_oracle_entry(payload[0]):
            raise ValueError("Oracle report requires oracle entries with oracle_response fields.")
        first_oracle_prompt = ""
        if payload and isinstance(payload[0], dict):
            first_oracle_prompt = str(payload[0].get("oracle_prompt", ""))
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model)
        final = save_oracle_rollouts_html(
            oracle_results=payload,
            oracle_prompt=first_oracle_prompt,
            tokenizer=tokenizer,
            output_path=str(out_path),
        )

    print(f"Saved report: {final}")


if __name__ == "__main__":
    main()
