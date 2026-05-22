from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cache_utils import preview_hash_name


def prompt_key(text: str, preview_len: int = 32) -> str:
    return preview_hash_name(text, preview_len=preview_len, hash_len=12)


def _normalize_target_prompt(item: Any) -> str | None:
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        if "goal" in item:
            text = str(item["goal"]).strip()
            return text or None
        if "prompt" in item:
            text = str(item["prompt"]).strip()
            return text or None
    return None


def load_target_prompts_from_dataset(limit: int = 100, offset: int = 0) -> list[str]:
    from datasets import load_dataset  # type: ignore

    if offset < 0:
        raise ValueError(f"offset must be >= 0, got {offset}")
    if limit <= 0:
        return []

    column = load_dataset("LLM-LAT/harmful-dataset", split="train")["prompt"]
    selected = column[offset : offset + limit]
    prompts: list[str] = []
    for item in selected:
        normalized = _normalize_target_prompt(item)
        if normalized:
            prompts.append(normalized)
    return prompts


def load_oracle_prompts_from_file(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Oracle prompts file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            prompts = [str(item).strip() for item in payload]
        elif isinstance(payload, dict) and isinstance(payload.get("oracle_prompts"), list):
            prompts = [str(item).strip() for item in payload["oracle_prompts"]]
        else:
            raise ValueError("Oracle prompt JSON must be a list or an object with 'oracle_prompts' list.")
        return [prompt for prompt in prompts if prompt]

    if suffix in {".txt", ".jsonl"}:
        lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
        return [line for line in lines if line]

    raise ValueError(f"Unsupported oracle prompts file extension: {suffix}")
