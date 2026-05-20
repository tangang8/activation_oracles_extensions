import hashlib
import json
import re
from pathlib import Path
from typing import Any


def _sanitize_for_path(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "unknown"


def _model_bundle_dir(prefix: str, model_name: str, lora_name: str) -> Path:
    model_name_s = _sanitize_for_path(model_name)
    lora_name_s = _sanitize_for_path(lora_name)
    if lora_name_s == "default":
        return Path(f"{prefix}_{model_name_s}")
    return Path(f"{prefix}_{model_name_s}_lora-{lora_name_s}")


def _run_subdir_path(
    role_prefix: str,
    model_name: str,
    lora_path: str | None,
) -> Path:
    role_dir = _model_bundle_dir(role_prefix, model_name, lora_path or "default")
    return role_dir


def _rollouts_dir_name(base_name: str, generation_kwargs: dict) -> str:
    temp = generation_kwargs.get("temperature")
    return f"{base_name}_temp-{temp}"


def _cache_prompt_file_name(prompt_text: str) -> str:
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    prompt_preview = _sanitize_for_path(prompt_text[:48])
    return f"{prompt_preview}_{prompt_hash}.json"


def _cache_base_dir(
    cache_root: str,
    target_model_name: str,
    target_lora_name: str,
    generation_kwargs: dict,
) -> Path:
    target_dir = _model_bundle_dir("target", target_model_name, target_lora_name)
    rollouts_dir = _rollouts_dir_name("target_rollouts", generation_kwargs)
    return (
        Path(cache_root)
        / target_dir
        / rollouts_dir
    )


def _preview_hash_name(text: str, preview_len: int = 48) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    preview = _sanitize_for_path(text[:preview_len])
    return f"{preview}_{digest}"


def cache_file_path(
    cache_root: str,
    target_model_name: str,
    target_lora_path: str | None,
    judge_model_name: str,
    judge_lora_path: str | None,
    generation_kwargs: dict,
    user_prompt: str,
) -> Path:
    """
    Build target rollout cache file path.

    Layout:
      cache/target_{target_model}[_lora-{target_lora}]/
      target_rollouts_temp-{temperature}/judge_{judge_model}[_lora-{judge_lora}]/
      {prompt_preview_hash}.json
    """
    target_lora_name = target_lora_path or "default"
    run_subdir = _run_subdir_path(
        role_prefix="judge",
        model_name=judge_model_name,
        lora_path=judge_lora_path,
    )
    prompt_file = _cache_prompt_file_name(user_prompt)
    cache_dir = _cache_base_dir(
        cache_root=cache_root,
        target_model_name=target_model_name,
        target_lora_name=target_lora_name,
        generation_kwargs=generation_kwargs,
    )
    return cache_dir / run_subdir / prompt_file


def oracle_cache_file_path(
    cache_root: str,
    target_model_name: str,
    target_lora_path: str | None,
    oracle_model_name: str,
    oracle_lora_path: str | None,
    generation_kwargs: dict,
    oracle_prompt: str,
    user_prompt_preview_text: str,
    cache_key_text: str,
) -> Path:
    """
    Build oracle cache file path for one target sequence under one oracle prompt/config.

    Layout:
      cache/target_{target_model}[_lora-{target_lora}]/
      oracle_rollouts_temp-{temperature}/oracle_{oracle_model}[_lora-{oracle_lora}]/
      {oracle_prompt_preview_hash}/{user_prompt_preview_hash}/{cache_key_hash}.json
    """
    target_dir = _model_bundle_dir("target", target_model_name, target_lora_path or "default")
    oracle_rollouts_dir = _rollouts_dir_name("oracle_rollouts", generation_kwargs)
    oracle_run_subdir = _run_subdir_path(
        role_prefix="oracle",
        model_name=oracle_model_name,
        lora_path=oracle_lora_path,
    )

    oracle_prompt_dir = _preview_hash_name(oracle_prompt)
    user_prompt_dir = _preview_hash_name(user_prompt_preview_text)
    cache_hash = hashlib.sha256(cache_key_text.encode("utf-8")).hexdigest()[:16]
    file_name = f"{cache_hash}.json"

    return (
        Path(cache_root)
        / target_dir
        / oracle_rollouts_dir
        / oracle_run_subdir
        / oracle_prompt_dir
        / user_prompt_dir
        / file_name
    )


def load_json(cache_file: Path) -> Any | None:
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text())
    except json.JSONDecodeError:
        return None


def write_json(cache_file: Path, value: Any) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(value, indent=2, ensure_ascii=True))
