from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_utils import (
    deterministic_oracle_cache_file_path,
    load_json,
    oracle_prompt_rollout_cache_file_path,
    write_json,
)
from distributed_utils import DistributedContext, broadcast_object
from oracle_pipeline import run_oracle_batched
from perf_utils import PerfLogger
from rollout_utils import format_user_target_prompt


OracleRolloutMode = Literal[
    "sampled_target_repeats",
    "all_target_deterministic",
    "prompt_only_repeats",
]

SAMPLED_TARGET_REPEATS: OracleRolloutMode = "sampled_target_repeats"
ALL_TARGET_DETERMINISTIC: OracleRolloutMode = "all_target_deterministic"
PROMPT_ONLY_REPEATS: OracleRolloutMode = "prompt_only_repeats"
VALID_ORACLE_ROLLOUT_MODES: tuple[OracleRolloutMode, ...] = (
    SAMPLED_TARGET_REPEATS,
    ALL_TARGET_DETERMINISTIC,
    PROMPT_ONLY_REPEATS,
)
DEFAULT_ORACLE_ROLLOUT_MODE: OracleRolloutMode = ALL_TARGET_DETERMINISTIC

DEFAULT_ORACLE_INPUT_TYPES = ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
DEFAULT_ORACLE_GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 1000,
}

SAMPLED_ORACLE_INPUT_TYPES = ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
SAMPLED_ORACLE_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "max_new_tokens": 1000,
}

PROMPT_ONLY_ORACLE_INPUT_TYPES = ["full_seq", "token_points"]
PROMPT_ONLY_ORACLE_GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 1.0,
    "max_new_tokens": 1000,
}


def _oracle_cache_variant_key(
    oracle_input_types: list[str] | None,
    oracle_token_point_filter: str,
) -> str | None:
    if oracle_input_types is None and oracle_token_point_filter == "all":
        return None
    return json.dumps(
        {
            "oracle_input_types": oracle_input_types,
            "oracle_token_point_filter": oracle_token_point_filter,
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def parse_oracle_rollout_mode(raw_mode: str | None) -> OracleRolloutMode:
    normalized = (raw_mode or "").strip()
    if not normalized:
        return DEFAULT_ORACLE_ROLLOUT_MODE
    if normalized not in VALID_ORACLE_ROLLOUT_MODES:
        raise ValueError(
            f"Unsupported ORACLE_ROLLOUT_MODE={normalized!r}. "
            f"Expected one of {', '.join(VALID_ORACLE_ROLLOUT_MODES)}."
        )
    return normalized  # type: ignore[return-value]


def oracle_rollouts_dir_base_for_mode(mode: OracleRolloutMode) -> str:
    if mode == PROMPT_ONLY_REPEATS:
        return "oracle_prompt_rollouts"
    return "oracle_rollouts"


def run_oracle_combined_singlelayer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    oracle_prompt: str,
    formatted_target_prompts: str | list[str],
    target_responses: str | list[str],
    user_prompts: str | list[str] | None = None,
    oracle_lora_path: str = "oracle",
    oracle_input_types: list[str] | None = None,
    oracle_repeats: int = 1,
    eval_batch_size: int = 32,
    token_point_indices_by_target: list[list[int]] | None = None,
    oracle_token_point_filter: str = "all",
    generation_kwargs: dict[str, Any] | None = None,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Compatibility wrapper over run_oracle_batched for combined prompt+rollout inputs."""
    if oracle_input_types is None:
        oracle_input_types = ["full_seq", "prompt_segment", "rollout_segment", "token_points"]

    target_responses_is_str, target_responses_list = (
        (True, [target_responses]) if isinstance(target_responses, str) else (False, target_responses)
    )
    if not target_responses_list:
        return []
    if isinstance(formatted_target_prompts, str):
        formatted_target_prompts = [formatted_target_prompts] * len(target_responses_list)
    if len(formatted_target_prompts) != len(target_responses_list):
        raise ValueError("formatted_target_prompts and target_responses must have the same length.")
    if user_prompts is not None:
        if isinstance(user_prompts, str):
            user_prompts = [user_prompts] * len(target_responses_list)
        if len(user_prompts) != len(target_responses_list):
            raise ValueError("user_prompts must have the same length as target_responses.")
    if token_point_indices_by_target is not None and len(token_point_indices_by_target) != len(target_responses_list):
        raise ValueError("token_point_indices_by_target must have the same length as target_responses.")

    batched_kwargs: dict[str, Any] = {}
    if token_point_indices_by_target is not None:
        batched_kwargs["token_point_indices_by_target"] = token_point_indices_by_target

    batched_outputs = run_oracle_batched(
        model=model,
        tokenizer=tokenizer,
        device=device,
        formatted_target_prompts=formatted_target_prompts,
        target_responses=target_responses_list,
        oracle_prompt=oracle_prompt,
        user_prompts=user_prompts,
        target_lora_path=None,
        oracle_lora_path=oracle_lora_path,
        oracle_repeats=oracle_repeats,
        eval_batch_size=eval_batch_size,
        oracle_input_types=oracle_input_types,
        oracle_token_point_filter=oracle_token_point_filter,
        generation_kwargs=generation_kwargs,
        oracle_input_source_type="target_rollout",
        dist_ctx=dist_ctx,
        perf=perf,
        **batched_kwargs,
    )
    for output in batched_outputs:
        token_points = output["points"]["token_points"]
        output["named_token_points"] = {
            name: output["token_points"].get(idx)
            for name, idx in token_points.items()
        }

    if target_responses_is_str:
        return batched_outputs[0]
    return batched_outputs


def visualize_token_selection(
    input_text: str,
    tokenizer: AutoTokenizer,
    token_points: dict[str, int] | None = None,
) -> None:
    """Print token positions and optionally highlight named token points."""
    input_ids = tokenizer(
        input_text, return_tensors="pt", add_special_tokens=False, padding=True
    )["input_ids"][0].tolist()
    print(len(input_ids))

    points_by_idx: dict[int, list[str]] = {}
    if token_points is not None:
        for name, idx in token_points.items():
            points_by_idx.setdefault(idx, []).append(name)

    print("Token selection visualization:")
    print("-" * 60)
    for i, token_id in enumerate(input_ids):
        token_str = tokenizer.decode([token_id]).replace("\n", "\\n").replace("\r", "\\r")
        if i in points_by_idx:
            labels = ", ".join(points_by_idx[i])
            print(f"  [{i:3d}] >>> {token_str}    <-- {labels}")
        else:
            print(f"  [{i:3d}]     {token_str}")
    print("-" * 60)
    if token_points is not None:
        print(f"Selected token points: {token_points}")


def _first_response(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        return str(value[0]).strip()
    if value is None:
        return ""
    return str(value).strip()


def _format_leaf(response_text: str) -> dict[str, Any]:
    return {
        "response_only": response_text,
        "thinking": "",
        "valid_response_format": True,
    }


def _read_prompt_only_cache_index(entry: dict[str, Any], cache_file: Path) -> int:
    if "oracle_rollout_index" in entry:
        return int(entry.get("oracle_rollout_index"))
    if "rollout_index" in entry:
        raise ValueError(
            "Prompt-only oracle cache entry is missing oracle_rollout_index "
            f"and only has rollout_index. cache_file={cache_file}"
        )
    raise ValueError(
        "Prompt-only oracle cache entry is missing oracle_rollout_index. "
        f"cache_file={cache_file}"
    )


def _read_target_cache_rollout_index(entry: dict[str, Any], cache_file: Path) -> int:
    if "rollout_index" in entry:
        return int(entry.get("rollout_index"))
    if "oracle_rollout_index" in entry:
        raise ValueError(
            "Target-backed oracle cache entry is missing rollout_index "
            f"and only has oracle_rollout_index. cache_file={cache_file}"
        )
    raise ValueError(
        "Target-backed oracle cache entry is missing rollout_index. "
        f"cache_file={cache_file}"
    )


def _normalize_generation_kwargs(
    generation_kwargs: dict[str, Any] | None,
    *,
    defaults: dict[str, Any],
    do_sample: bool,
    temperature: float,
) -> dict[str, Any]:
    normalized = dict(defaults)
    if generation_kwargs is not None:
        normalized.update(generation_kwargs)
    normalized["do_sample"] = do_sample
    normalized["temperature"] = temperature
    normalized.setdefault("max_new_tokens", defaults.get("max_new_tokens", 1000))
    return normalized


def _to_deterministic_oracle_entry(
    target_entry: dict[str, Any],
    oracle_result: dict[str, Any],
) -> dict[str, Any]:
    scalar_probe_kinds = ("full_seq", "segment", "prompt_segment", "rollout_segment")
    scalar_responses = {kind: _first_response(oracle_result.get(kind, [])) for kind in scalar_probe_kinds}
    scalar_formats = {kind: _format_leaf(text) for kind, text in scalar_responses.items()}

    tokens_raw = oracle_result.get("tokens", {})
    tokens_response: dict[str, str] = {}
    tokens_format: dict[str, dict[str, Any]] = {}
    if isinstance(tokens_raw, dict):
        for token_idx, values in tokens_raw.items():
            key = str(token_idx)
            text = _first_response(values)
            tokens_response[key] = text
            tokens_format[key] = _format_leaf(text)

    points = oracle_result.get("points", {}).get("token_points", {})
    token_points_raw = oracle_result.get("token_points", {})
    token_point_response: dict[str, str] = {}
    token_point_format: dict[str, dict[str, Any]] = {}
    if isinstance(points, dict) and isinstance(token_points_raw, dict):
        for point_name, token_idx in points.items():
            values = token_points_raw.get(token_idx, token_points_raw.get(str(token_idx), []))
            text = _first_response(values)
            point_key = str(point_name)
            token_point_response[point_key] = text
            token_point_format[point_key] = _format_leaf(text)

    oracle_response = {
        **scalar_responses,
        "tokens": tokens_response,
        "token_points": token_point_response,
    }
    oracle_format = {
        **scalar_formats,
        "tokens": tokens_format,
        "token_points": token_point_format,
    }

    return {
        "rollout_index": int(target_entry["rollout_index"]),
        "target_prompt": target_entry.get("target_prompt", ""),
        "target_response": target_entry.get("target_response", ""),
        "target_format": target_entry.get("target_format", {}),
        "oracle_response": oracle_response,
        "oracle_format": oracle_format,
        "oracle_prompt": oracle_result.get("oracle_prompt", ""),
        "oracle_points": oracle_result.get("points", {}),
    }


def _oracle_result_for_repeat(
    oracle_result: dict[str, Any],
    repeat_idx: int,
) -> dict[str, Any]:
    repeat_result: dict[str, Any] = {
        "points": oracle_result.get("points", {}),
        "oracle_prompt": oracle_result.get("oracle_prompt", ""),
    }
    for key in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
        values = oracle_result.get(key, [])
        if isinstance(values, list) and repeat_idx < len(values):
            repeat_result[key] = [values[repeat_idx]]
        else:
            repeat_result[key] = []

    for bucket_name in ("tokens", "token_points"):
        bucket_values = oracle_result.get(bucket_name, {})
        repeated_bucket: dict[Any, list[Any]] = {}
        if isinstance(bucket_values, dict):
            for token_idx, values in bucket_values.items():
                if isinstance(values, list) and repeat_idx < len(values):
                    repeated_bucket[token_idx] = [values[repeat_idx]]
        repeat_result[bucket_name] = repeated_bucket

    return repeat_result


def _to_prompt_only_oracle_entry(
    *,
    target_prompt: str,
    formatted_target_prompt: str,
    oracle_result: dict[str, Any],
    oracle_rollout_index: int,
) -> dict[str, Any]:
    scalar_probe_kinds = ("full_seq", "segment", "prompt_segment")
    scalar_responses = {kind: _first_response(oracle_result.get(kind, [])) for kind in scalar_probe_kinds}
    scalar_formats = {kind: _format_leaf(text) for kind, text in scalar_responses.items()}

    tokens_raw = oracle_result.get("tokens", {})
    tokens_response: dict[str, str] = {}
    tokens_format: dict[str, dict[str, Any]] = {}
    if isinstance(tokens_raw, dict):
        for token_idx, values in tokens_raw.items():
            key = str(token_idx)
            text = _first_response(values)
            tokens_response[key] = text
            tokens_format[key] = _format_leaf(text)

    points = oracle_result.get("points", {}).get("token_points", {})
    token_points_raw = oracle_result.get("token_points", {})
    token_point_response: dict[str, str] = {}
    token_point_format: dict[str, dict[str, Any]] = {}
    if isinstance(points, dict) and isinstance(token_points_raw, dict):
        for point_name, token_idx in points.items():
            values = token_points_raw.get(token_idx, token_points_raw.get(str(token_idx), []))
            text = _first_response(values)
            key = str(point_name)
            token_point_response[key] = text
            token_point_format[key] = _format_leaf(text)

    return {
        "oracle_rollout_index": oracle_rollout_index,
        "target_prompt": target_prompt,
        "formatted_target_prompt": formatted_target_prompt,
        "oracle_prompt": oracle_result.get("oracle_prompt", ""),
        "oracle_response": {
            **scalar_responses,
            "tokens": tokens_response,
            "token_points": token_point_response,
        },
        "oracle_format": {
            **scalar_formats,
            "tokens": tokens_format,
            "token_points": token_point_format,
        },
        "oracle_points": oracle_result.get("points", {}),
    }


def generate_deterministic_oracle_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    oracle_prompt: str,
    target_rollout_entries: list[dict[str, Any]],
    target_model_name: str,
    target_lora_path: str | None,
    oracle_lora_path: str | None = "oracle",
    cache_root: str = "cache",
    oracle_generation_kwargs: dict[str, Any] | None = None,
    oracle_input_types: list[str] | None = None,
    oracle_token_point_filter: str = "all",
    eval_batch_size: int = 32,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, float]]:
    oracle_generation_kwargs = _normalize_generation_kwargs(
        oracle_generation_kwargs,
        defaults=DEFAULT_ORACLE_GENERATION_KWARGS,
        do_sample=False,
        temperature=0.0,
    )

    explicit_oracle_input_types = oracle_input_types is not None
    if oracle_input_types is None:
        oracle_input_types = list(DEFAULT_ORACLE_INPUT_TYPES)
    cache_variant_key = _oracle_cache_variant_key(
        oracle_input_types if explicit_oracle_input_types else None,
        oracle_token_point_filter,
    )

    if not target_rollout_entries:
        empty_path = deterministic_oracle_cache_file_path(
            cache_root=cache_root,
            target_model_name=target_model_name,
            target_lora_path=target_lora_path,
            oracle_model_name=model.config._name_or_path,
            oracle_lora_path=oracle_lora_path,
            oracle_generation_kwargs=oracle_generation_kwargs,
            target_prompt="",
            oracle_prompt=oracle_prompt,
            cache_variant_key=cache_variant_key,
        )
        return [], empty_path, {"cache/oracle_hits": 0.0, "cache/oracle_missing": 0.0}

    target_prompt = str(target_rollout_entries[0].get("target_prompt", ""))
    cache_file = deterministic_oracle_cache_file_path(
        cache_root=cache_root,
        target_model_name=target_model_name,
        target_lora_path=target_lora_path,
        oracle_model_name=model.config._name_or_path,
        oracle_lora_path=oracle_lora_path,
        oracle_generation_kwargs=oracle_generation_kwargs,
        target_prompt=target_prompt,
        oracle_prompt=oracle_prompt,
        cache_variant_key=cache_variant_key,
    )

    loaded = load_json(cache_file)
    existing_entries = loaded if isinstance(loaded, list) else []
    existing_by_index: dict[int, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = _read_target_cache_rollout_index(entry, cache_file)
        except Exception:
            continue
        existing_by_index[idx] = entry

    missing_target_entries = [
        entry
        for entry in target_rollout_entries
        if int(entry["rollout_index"]) not in existing_by_index
    ]

    rank = dist_ctx.rank if dist_ctx is not None else 0
    world_size = dist_ctx.world_size if dist_ctx is not None else 1
    cache_stats = {
        "cache/oracle_hits": float(len(existing_by_index)),
        "cache/oracle_missing": float(len(missing_target_entries)),
        "cache/oracle_total": float(len(target_rollout_entries)),
    }
    if perf is not None:
        perf.log_event(
            "cache/oracle_deterministic_status",
            cache_stats,
            metadata={"rank": rank, "world_size": world_size},
        )

    if missing_target_entries:
        formatted_target_prompts = [
            format_user_target_prompt(tokenizer, str(entry.get("target_prompt", "")))
            for entry in missing_target_entries
        ]
        target_responses = [str(entry.get("target_response", "")) for entry in missing_target_entries]
        user_prompts = [str(entry.get("target_prompt", "")) for entry in missing_target_entries]

        batched_oracle_results = run_oracle_batched(
            model=model,
            tokenizer=tokenizer,
            device=device,
            formatted_target_prompts=formatted_target_prompts,
            target_responses=target_responses,
            oracle_prompt=oracle_prompt,
            user_prompts=user_prompts,
            cache_root=cache_root,
            target_lora_path=None,
            oracle_lora_path=oracle_lora_path,
            generation_kwargs=oracle_generation_kwargs,
            eval_batch_size=eval_batch_size,
            oracle_repeats=1,
            oracle_input_types=oracle_input_types,
            oracle_token_point_filter=oracle_token_point_filter,
            oracle_input_source_type="target_rollout",
            dist_ctx=dist_ctx,
            perf=perf,
        )
        for result in batched_oracle_results:
            result["oracle_prompt"] = oracle_prompt
        new_entries = [
            _to_deterministic_oracle_entry(target_entry, result)
            for target_entry, result in zip(missing_target_entries, batched_oracle_results, strict=True)
        ]
    else:
        new_entries = []

    is_main = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    final_entries: list[dict[str, Any]] | None = None
    if is_main:
        merged_by_index = dict(existing_by_index)
        for entry in new_entries:
            merged_by_index[int(entry["rollout_index"])] = entry
        final_entries = []
        for target_entry in sorted(target_rollout_entries, key=lambda e: int(e["rollout_index"])):
            idx = int(target_entry["rollout_index"])
            if idx in merged_by_index:
                final_entries.append(merged_by_index[idx])
        write_json(cache_file, final_entries)

    if dist_ctx is not None and dist_ctx.enabled:
        final_entries = broadcast_object(dist_ctx, final_entries, src=0)
    if final_entries is None:
        final_entries = []
    return final_entries, cache_file, cache_stats


def generate_sampled_target_oracle_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    oracle_prompt: str,
    target_rollout_entries: list[dict[str, Any]],
    target_model_name: str,
    target_lora_path: str | None,
    *,
    num_oracle_rollouts: int,
    k_rollouts: int | None,
    oracle_lora_path: str | None = "oracle",
    cache_root: str = "cache",
    oracle_generation_kwargs: dict[str, Any] | None = None,
    oracle_input_types: list[str] | None = None,
    oracle_token_point_filter: str = "all",
    eval_batch_size: int = 32,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, float]]:
    if num_oracle_rollouts <= 0:
        raise ValueError(f"num_oracle_rollouts must be > 0, got {num_oracle_rollouts}")
    if k_rollouts is not None and k_rollouts <= 0:
        raise ValueError(f"k_rollouts must be > 0 when provided, got {k_rollouts}")

    oracle_generation_kwargs = _normalize_generation_kwargs(
        oracle_generation_kwargs,
        defaults=SAMPLED_ORACLE_GENERATION_KWARGS,
        do_sample=True,
        temperature=1.0,
    )
    explicit_oracle_input_types = oracle_input_types is not None
    if oracle_input_types is None:
        oracle_input_types = list(SAMPLED_ORACLE_INPUT_TYPES)
    cache_variant_key = _oracle_cache_variant_key(
        oracle_input_types if explicit_oracle_input_types else None,
        oracle_token_point_filter,
    )

    sorted_targets = sorted(target_rollout_entries, key=lambda entry: int(entry["rollout_index"]))
    if k_rollouts is not None:
        selected_targets = sorted_targets[: min(k_rollouts, len(sorted_targets))]
    else:
        selected_targets = sorted_targets

    if not selected_targets:
        empty_path = deterministic_oracle_cache_file_path(
            cache_root=cache_root,
            target_model_name=target_model_name,
            target_lora_path=target_lora_path,
            oracle_model_name=model.config._name_or_path,
            oracle_lora_path=oracle_lora_path,
            oracle_generation_kwargs=oracle_generation_kwargs,
            target_prompt="",
            oracle_prompt=oracle_prompt,
            cache_variant_key=cache_variant_key,
        )
        return [], empty_path, {"cache/oracle_hits": 0.0, "cache/oracle_missing": 0.0}

    target_prompt = str(selected_targets[0].get("target_prompt", ""))
    cache_file = deterministic_oracle_cache_file_path(
        cache_root=cache_root,
        target_model_name=target_model_name,
        target_lora_path=target_lora_path,
        oracle_model_name=model.config._name_or_path,
        oracle_lora_path=oracle_lora_path,
        oracle_generation_kwargs=oracle_generation_kwargs,
        target_prompt=target_prompt,
        oracle_prompt=oracle_prompt,
        cache_variant_key=cache_variant_key,
    )

    formatted_target_prompts = [
        format_user_target_prompt(tokenizer, str(entry.get("target_prompt", "")))
        for entry in selected_targets
    ]
    target_responses = [str(entry.get("target_response", "")) for entry in selected_targets]
    user_prompts = [str(entry.get("target_prompt", "")) for entry in selected_targets]
    batched_oracle_results = run_oracle_batched(
        model=model,
        tokenizer=tokenizer,
        device=device,
        formatted_target_prompts=formatted_target_prompts,
        target_responses=target_responses,
        oracle_prompt=oracle_prompt,
        user_prompts=user_prompts,
        cache_root=cache_root,
        target_lora_path=None,
        oracle_lora_path=oracle_lora_path,
        generation_kwargs=oracle_generation_kwargs,
        eval_batch_size=eval_batch_size,
        oracle_repeats=num_oracle_rollouts,
        oracle_input_types=oracle_input_types,
        oracle_token_point_filter=oracle_token_point_filter,
        oracle_input_source_type="target_rollout",
        dist_ctx=dist_ctx,
        perf=perf,
    )
    for result in batched_oracle_results:
        result["oracle_prompt"] = oracle_prompt

    expanded_entries: list[dict[str, Any]] = []
    for target_entry, oracle_result in zip(selected_targets, batched_oracle_results, strict=True):
        repeats_for_target = int(oracle_result.get("oracle_repeats", 0))
        for oracle_rollout_index in range(min(num_oracle_rollouts, repeats_for_target)):
            repeat_result = _oracle_result_for_repeat(oracle_result, oracle_rollout_index)
            repeat_result["oracle_prompt"] = oracle_prompt
            entry = _to_deterministic_oracle_entry(target_entry, repeat_result)
            entry["target_rollout_index"] = int(target_entry["rollout_index"])
            entry["oracle_rollout_index"] = oracle_rollout_index
            expanded_entries.append(entry)

    expanded_entries.sort(
        key=lambda entry: (
            int(entry.get("target_rollout_index", -1)),
            int(entry.get("oracle_rollout_index", -1)),
        )
    )
    for rollout_index, entry in enumerate(expanded_entries):
        entry["rollout_index"] = rollout_index

    total_expected = len(selected_targets) * num_oracle_rollouts
    cache_stats = {
        "cache/oracle_hits": float(0),
        "cache/oracle_missing": float(max(0, total_expected - len(expanded_entries))),
        "cache/oracle_total": float(total_expected),
    }

    is_main = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    final_entries: list[dict[str, Any]] | None = None
    if is_main:
        write_json(cache_file, expanded_entries)
        final_entries = expanded_entries

    if dist_ctx is not None and dist_ctx.enabled:
        final_entries = broadcast_object(dist_ctx, final_entries, src=0)
    if final_entries is None:
        final_entries = []
    return final_entries, cache_file, cache_stats


def generate_prompt_only_oracle_rollouts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    oracle_prompt: str,
    target_prompt: str,
    target_model_name: str,
    target_lora_path: str | None,
    *,
    num_oracle_rollouts: int,
    oracle_lora_path: str | None = "oracle",
    cache_root: str = "cache",
    oracle_generation_kwargs: dict[str, Any] | None = None,
    oracle_input_types: list[str] | None = None,
    oracle_token_point_filter: str = "all",
    eval_batch_size: int = 32,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, float]]:
    if num_oracle_rollouts <= 0:
        raise ValueError(f"num_oracle_rollouts must be > 0, got {num_oracle_rollouts}")

    oracle_generation_kwargs = _normalize_generation_kwargs(
        oracle_generation_kwargs,
        defaults=PROMPT_ONLY_ORACLE_GENERATION_KWARGS,
        do_sample=True,
        temperature=1.0,
    )
    if oracle_input_types is None:
        oracle_input_types = list(PROMPT_ONLY_ORACLE_INPUT_TYPES)
    scalar_probe_kinds = {"full_seq", "segment", "prompt_segment", "rollout_segment"}
    expected_scalar_probes = {probe for probe in oracle_input_types if probe in scalar_probe_kinds}

    formatted_target_prompt = format_user_target_prompt(tokenizer, target_prompt)
    cache_file = oracle_prompt_rollout_cache_file_path(
        cache_root=cache_root,
        target_model_name=target_model_name,
        target_lora_path=target_lora_path,
        oracle_model_name=model.config._name_or_path,
        oracle_lora_path=oracle_lora_path,
        oracle_generation_kwargs=oracle_generation_kwargs,
        target_prompt=target_prompt,
        oracle_prompt=oracle_prompt,
    )

    loaded = load_json(cache_file)
    existing_entries = loaded if isinstance(loaded, list) else []
    existing_by_index: dict[int, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = _read_prompt_only_cache_index(entry, cache_file)
        except ValueError:
            raise
        except Exception:
            continue
        oracle_response = entry.get("oracle_response", {})
        if not isinstance(oracle_response, dict):
            continue
        cached_scalar_probes = {probe for probe in scalar_probe_kinds if probe in oracle_response}
        if cached_scalar_probes != expected_scalar_probes:
            continue
        existing_by_index[idx] = entry

    cache_stats = {
        "cache/oracle_hits": float(len(existing_by_index)),
        "cache/oracle_missing": float(max(0, num_oracle_rollouts - len(existing_by_index))),
        "cache/oracle_total": float(num_oracle_rollouts),
    }

    if len(existing_by_index) >= num_oracle_rollouts:
        final_entries = [existing_by_index[i] for i in range(num_oracle_rollouts) if i in existing_by_index]
        return final_entries, cache_file, cache_stats

    batched_oracle_results = run_oracle_batched(
        model=model,
        tokenizer=tokenizer,
        device=device,
        formatted_target_prompts=[formatted_target_prompt],
        target_responses=None,
        oracle_prompt=oracle_prompt,
        user_prompts=[target_prompt],
        cache_root=cache_root,
        target_lora_path=None,
        oracle_lora_path=oracle_lora_path,
        generation_kwargs=oracle_generation_kwargs,
        eval_batch_size=eval_batch_size,
        oracle_repeats=num_oracle_rollouts,
        oracle_input_types=oracle_input_types,
        oracle_token_point_filter=oracle_token_point_filter,
        oracle_input_source_type="prompt_only",
        dist_ctx=dist_ctx,
        perf=perf,
    )
    combined_result = batched_oracle_results[0] if batched_oracle_results else {}
    combined_result["oracle_prompt"] = oracle_prompt

    new_entries = [
        _to_prompt_only_oracle_entry(
            target_prompt=target_prompt,
            formatted_target_prompt=formatted_target_prompt,
            oracle_result=_oracle_result_for_repeat(combined_result, oracle_rollout_index),
            oracle_rollout_index=oracle_rollout_index,
        )
        for oracle_rollout_index in range(num_oracle_rollouts)
    ]
    for entry in new_entries:
        entry["oracle_prompt"] = oracle_prompt

    is_main = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    final_entries: list[dict[str, Any]] | None = None
    if is_main:
        write_json(cache_file, new_entries)
        final_entries = new_entries

    if dist_ctx is not None and dist_ctx.enabled:
        final_entries = broadcast_object(dist_ctx, final_entries, src=0)
    if final_entries is None:
        final_entries = []
    return final_entries, cache_file, cache_stats


def generate_oracle_rollouts_for_mode(
    *,
    mode: OracleRolloutMode,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    oracle_prompt: str,
    target_prompt: str,
    target_rollout_entries: list[dict[str, Any]],
    target_model_name: str,
    target_lora_path: str | None,
    oracle_lora_path: str | None = "oracle",
    cache_root: str = "cache",
    k_rollouts: int | None = None,
    num_oracle_rollouts: int = 1,
    oracle_generation_kwargs_deterministic: dict[str, Any] | None = None,
    oracle_generation_kwargs_sampled: dict[str, Any] | None = None,
    oracle_generation_kwargs_prompt_only: dict[str, Any] | None = None,
    oracle_input_types_deterministic: list[str] | None = None,
    oracle_input_types_sampled: list[str] | None = None,
    oracle_input_types_prompt_only: list[str] | None = None,
    oracle_token_point_filter: str = "all",
    eval_batch_size: int = 32,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, float]]:
    if mode == ALL_TARGET_DETERMINISTIC:
        return generate_deterministic_oracle_rollouts(
            model=model,
            tokenizer=tokenizer,
            device=device,
            oracle_prompt=oracle_prompt,
            target_rollout_entries=target_rollout_entries,
            target_model_name=target_model_name,
            target_lora_path=target_lora_path,
            oracle_lora_path=oracle_lora_path,
            cache_root=cache_root,
            oracle_generation_kwargs=oracle_generation_kwargs_deterministic,
            oracle_input_types=oracle_input_types_deterministic,
            oracle_token_point_filter=oracle_token_point_filter,
            eval_batch_size=eval_batch_size,
            dist_ctx=dist_ctx,
            perf=perf,
        )

    if mode == SAMPLED_TARGET_REPEATS:
        return generate_sampled_target_oracle_rollouts(
            model=model,
            tokenizer=tokenizer,
            device=device,
            oracle_prompt=oracle_prompt,
            target_rollout_entries=target_rollout_entries,
            target_model_name=target_model_name,
            target_lora_path=target_lora_path,
            num_oracle_rollouts=num_oracle_rollouts,
            k_rollouts=k_rollouts,
            oracle_lora_path=oracle_lora_path,
            cache_root=cache_root,
            oracle_generation_kwargs=oracle_generation_kwargs_sampled,
            oracle_input_types=oracle_input_types_sampled,
            oracle_token_point_filter=oracle_token_point_filter,
            eval_batch_size=eval_batch_size,
            dist_ctx=dist_ctx,
            perf=perf,
        )

    if mode == PROMPT_ONLY_REPEATS:
        if target_rollout_entries:
            print(
                "[oracle prompt-only] ignoring target rollout entries; "
                f"count={len(target_rollout_entries)} source=formatted_target_prompt_only"
            )
        return generate_prompt_only_oracle_rollouts(
            model=model,
            tokenizer=tokenizer,
            device=device,
            oracle_prompt=oracle_prompt,
            target_prompt=target_prompt,
            target_model_name=target_model_name,
            target_lora_path=target_lora_path,
            num_oracle_rollouts=num_oracle_rollouts,
            oracle_lora_path=oracle_lora_path,
            cache_root=cache_root,
            oracle_generation_kwargs=oracle_generation_kwargs_prompt_only,
            oracle_input_types=oracle_input_types_prompt_only,
            oracle_token_point_filter=oracle_token_point_filter,
            eval_batch_size=eval_batch_size,
            dist_ctx=dist_ctx,
            perf=perf,
        )

    raise ValueError(
        f"Unsupported oracle rollout mode {mode!r}; expected one of {', '.join(VALID_ORACLE_ROLLOUT_MODES)}."
    )
