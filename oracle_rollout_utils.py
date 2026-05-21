from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_utils import deterministic_oracle_cache_file_path, load_json, write_json
from distributed_utils import DistributedContext, broadcast_object
from oracle_pipeline import run_oracle_batched
from perf_utils import PerfLogger
from rollout_utils import format_user_target_prompt


DEFAULT_ORACLE_INPUT_TYPES = ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
DEFAULT_ORACLE_GENERATION_KWARGS = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 1000,
}


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
        generation_kwargs=generation_kwargs,
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
    eval_batch_size: int = 32,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, float]]:
    if oracle_generation_kwargs is None:
        oracle_generation_kwargs = dict(DEFAULT_ORACLE_GENERATION_KWARGS)
    else:
        oracle_generation_kwargs = dict(oracle_generation_kwargs)
    oracle_generation_kwargs["do_sample"] = False
    oracle_generation_kwargs["temperature"] = 0.0
    oracle_generation_kwargs.setdefault("max_new_tokens", 1000)

    if oracle_input_types is None:
        oracle_input_types = list(DEFAULT_ORACLE_INPUT_TYPES)

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
    )

    loaded = load_json(cache_file)
    existing_entries = loaded if isinstance(loaded, list) else []
    existing_by_index: dict[int, dict[str, Any]] = {}
    for entry in existing_entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("rollout_index"))
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
