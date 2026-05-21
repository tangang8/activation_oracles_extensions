from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache_utils import deterministic_oracle_judge_cache_file_path, load_json, write_json
from distributed_utils import DistributedContext, all_gather_objects, broadcast_object
from perf_utils import PerfLogger
from rollout_utils import (
    THINKING_TAG_PATTERNS_BY_MODEL,
    score_responses_compliance_batched,
)


def _flatten_oracle_responses(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rollout_index = int(entry["rollout_index"])
    user_prompt = str(entry.get("target_prompt", ""))
    oracle_response = entry.get("oracle_response", {})
    oracle_format = entry.get("oracle_format", {})
    flattened: list[dict[str, Any]] = []

    for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
        leaf_format = oracle_format.get(probe_kind, {}) if isinstance(oracle_format, dict) else {}
        if isinstance(leaf_format, dict):
            response_text = str(leaf_format.get("response_only", "")).strip()
        else:
            response_text = str(oracle_response.get(probe_kind, "")).strip()
        flattened.append(
            {
                "rollout_index": rollout_index,
                "path": (probe_kind,),
                "probe_kind": probe_kind,
                "user_prompt": user_prompt,
                "response_text": response_text,
            }
        )

    tokens_response = oracle_response.get("tokens", {})
    tokens_format = oracle_format.get("tokens", {}) if isinstance(oracle_format, dict) else {}
    if isinstance(tokens_response, dict):
        for token_index, raw_response in tokens_response.items():
            token_key = str(token_index)
            leaf_format = tokens_format.get(token_key, {}) if isinstance(tokens_format, dict) else {}
            if isinstance(leaf_format, dict):
                response_text = str(leaf_format.get("response_only", raw_response)).strip()
            else:
                response_text = str(raw_response).strip()
            flattened.append(
                {
                    "rollout_index": rollout_index,
                    "path": ("tokens", token_key),
                    "probe_kind": "tokens",
                    "token_index": token_key,
                    "user_prompt": user_prompt,
                    "response_text": response_text,
                }
            )

    token_point_response = oracle_response.get("token_points", {})
    token_point_format = oracle_format.get("token_points", {}) if isinstance(oracle_format, dict) else {}
    if isinstance(token_point_response, dict):
        for token_point_name, raw_response in token_point_response.items():
            point_key = str(token_point_name)
            leaf_format = token_point_format.get(point_key, {}) if isinstance(token_point_format, dict) else {}
            if isinstance(leaf_format, dict):
                response_text = str(leaf_format.get("response_only", raw_response)).strip()
            else:
                response_text = str(raw_response).strip()
            flattened.append(
                {
                    "rollout_index": rollout_index,
                    "path": ("token_points", point_key),
                    "probe_kind": "token_points",
                    "token_point_name": point_key,
                    "user_prompt": user_prompt,
                    "response_text": response_text,
                }
            )
    return flattened


def _get_path_leaf(root: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = root
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _set_path_leaf(root: dict[str, Any], path: tuple[str, ...], value: dict[str, Any]) -> None:
    node = root
    for key in path[:-1]:
        next_node = node.get(key)
        if not isinstance(next_node, dict):
            next_node = {}
            node[key] = next_node
        node = next_node
    node[path[-1]] = value


def _compliance_shell(entry: dict[str, Any]) -> dict[str, Any]:
    oracle_response = entry.get("oracle_response", {})
    shell: dict[str, Any] = {}
    for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
        if probe_kind in oracle_response:
            shell[probe_kind] = None
    if isinstance(oracle_response.get("tokens"), dict):
        shell["tokens"] = {str(key): None for key in oracle_response["tokens"].keys()}
    if isinstance(oracle_response.get("token_points"), dict):
        shell["token_points"] = {str(key): None for key in oracle_response["token_points"].keys()}
    return shell


def _oracle_judge_summary(judged_entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_probe: dict[str, list[int]] = {}
    total_scored = 0
    for entry in judged_entries:
        compliance = entry.get("compliance", {})
        flattened = []
        if isinstance(compliance, dict):
            for probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
                leaf = compliance.get(probe_kind)
                if isinstance(leaf, dict):
                    flattened.append((probe_kind, leaf))
            for probe_kind in ("tokens", "token_points"):
                container = compliance.get(probe_kind, {})
                if isinstance(container, dict):
                    for leaf in container.values():
                        if isinstance(leaf, dict):
                            flattened.append((probe_kind, leaf))
        for probe_kind, leaf in flattened:
            score = leaf.get("score")
            if isinstance(score, int):
                by_probe.setdefault(probe_kind, []).append(score)
                total_scored += 1

    summary: dict[str, Any] = {"oracle_judge/total_scored": float(total_scored)}
    for probe_kind, scores in by_probe.items():
        if scores:
            summary[f"oracle_judge/{probe_kind}_avg_score"] = float(sum(scores)) / float(len(scores))
            summary[f"oracle_judge/{probe_kind}_count"] = float(len(scores))
    return summary


def judge_deterministic_oracle_rollouts(
    judge_model: AutoModelForCausalLM,
    judge_tokenizer: AutoTokenizer,
    oracle_rollout_entries: list[dict[str, Any]],
    oracle_prompt: str,
    judge_instruction_template: str,
    judge_instruction_file: str,
    judge_instruction_stem: str,
    device: torch.device,
    target_model_name: str,
    target_lora_path: str | None,
    oracle_model_name: str,
    oracle_lora_path: str | None,
    oracle_generation_kwargs: dict[str, Any],
    judge_generation_kwargs: dict[str, Any] | None = None,
    judge_batch_size: int = 8,
    judge_lora_path: str | None = "default",
    cache_root: str = "cache",
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    if judge_generation_kwargs is None:
        judge_generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 1000}
    if judge_batch_size <= 0:
        raise ValueError(f"judge_batch_size must be > 0, got {judge_batch_size}")

    target_prompt = str(oracle_rollout_entries[0].get("target_prompt", "")) if oracle_rollout_entries else ""
    cache_file = deterministic_oracle_judge_cache_file_path(
        cache_root=cache_root,
        target_model_name=target_model_name,
        target_lora_path=target_lora_path,
        judge_model_name=judge_model.config._name_or_path,
        judge_lora_path=judge_lora_path,
        judge_generation_kwargs=judge_generation_kwargs,
        judge_instruction_stem=judge_instruction_stem,
        oracle_model_name=oracle_model_name,
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
            idx = int(entry["rollout_index"])
        except Exception:
            continue
        existing_by_index[idx] = entry

    merged_by_index: dict[int, dict[str, Any]] = {}
    pending_items: list[dict[str, Any]] = []
    for oracle_entry in oracle_rollout_entries:
        idx = int(oracle_entry["rollout_index"])
        base_entry = deepcopy(existing_by_index.get(idx, oracle_entry))
        if "compliance" not in base_entry or not isinstance(base_entry["compliance"], dict):
            base_entry["compliance"] = _compliance_shell(oracle_entry)
        merged_by_index[idx] = base_entry
        for item in _flatten_oracle_responses(oracle_entry):
            existing_leaf = _get_path_leaf(base_entry["compliance"], item["path"])
            if isinstance(existing_leaf, dict):
                continue
            pending_items.append(item)

    rank = dist_ctx.rank if dist_ctx is not None else 0
    world_size = dist_ctx.world_size if dist_ctx is not None else 1
    total_probe_items = sum(len(_flatten_oracle_responses(entry)) for entry in oracle_rollout_entries)
    if perf is not None:
        perf.log_event(
            "cache/oracle_judge_status",
            {
                "cache/oracle_judge_hits": float(max(0, total_probe_items - len(pending_items))),
                "cache/oracle_judge_missing": float(len(pending_items)),
                "cache/oracle_judge_total_rollouts": float(len(oracle_rollout_entries)),
            },
            metadata={"rank": rank, "world_size": world_size},
        )

    if dist_ctx is None or not dist_ctx.enabled:
        local_items = pending_items
    else:
        local_items = [item for i, item in enumerate(pending_items) if i % dist_ctx.world_size == dist_ctx.rank]

    judge_thinking_tag = THINKING_TAG_PATTERNS_BY_MODEL.get(judge_model.config._name_or_path)
    local_updates: list[dict[str, Any]] = []
    if local_items:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in local_items:
            grouped.setdefault(item["user_prompt"], []).append(item)

        with (
            perf.track(
                "oracle/judge_generate",
                {
                    "pending_items": len(local_items),
                    "judge_batch_size": judge_batch_size,
                    "judge_max_new_tokens": judge_generation_kwargs.get("max_new_tokens"),
                    "rank": rank,
                    "world_size": world_size,
                },
            )
            if perf
            else nullcontext()
        ) as judge_metrics:
            judge_t0 = perf_counter()
            for user_prompt, group_items in grouped.items():
                for offset in range(0, len(group_items), judge_batch_size):
                    chunk_items = group_items[offset : offset + judge_batch_size]
                    responses = [item["response_text"] for item in chunk_items]
                    judged = score_responses_compliance_batched(
                        judge_model=judge_model,
                        judge_tokenizer=judge_tokenizer,
                        user_prompt=user_prompt,
                        target_responses=responses,
                        judge_instruction_template=judge_instruction_template,
                        device=device,
                        judge_lora_path=judge_lora_path,
                        generation_kwargs=judge_generation_kwargs,
                        target_thinking_tag=None,
                        judge_thinking_tag=judge_thinking_tag,
                        emit_summary_log=False,
                        stage_label="oracle judging",
                        item_ids=[
                            (
                                f"rollout_index={int(item['rollout_index'])} "
                                f"probe={item['probe_kind']}"
                                + (
                                    f":{item['token_index']}"
                                    if "token_index" in item
                                    else (
                                        f":{item['token_point_name']}"
                                        if "token_point_name" in item
                                        else ""
                                    )
                                )
                            )
                            for item in chunk_items
                        ],
                        malformed_retry_attempts=3,
                    )
                    for item, compliance in zip(chunk_items, judged, strict=True):
                        payload = {
                            **compliance,
                            "judge_instruction_file": judge_instruction_file,
                            "probe_kind": item["probe_kind"],
                        }
                        if "token_index" in item:
                            payload["token_index"] = item["token_index"]
                        if "token_point_name" in item:
                            payload["token_point_name"] = item["token_point_name"]
                        local_updates.append(
                            {
                                "rollout_index": item["rollout_index"],
                                "path": list(item["path"]),
                                "compliance": payload,
                            }
                        )
            if perf:
                elapsed = max(perf_counter() - judge_t0, 1e-12)
                judge_metrics["throughput/oracle_judgments_per_second"] = float(len(local_items)) / elapsed

    if dist_ctx is not None and dist_ctx.enabled:
        gathered_updates = all_gather_objects(dist_ctx, local_updates)
    else:
        gathered_updates = [local_updates]

    is_main = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    final_entries: list[dict[str, Any]] | None = None
    if is_main:
        for updates in gathered_updates:
            if not isinstance(updates, list):
                continue
            for update in updates:
                idx = int(update["rollout_index"])
                if idx not in merged_by_index:
                    continue
                compliance_root = merged_by_index[idx].setdefault("compliance", {})
                if not isinstance(compliance_root, dict):
                    compliance_root = {}
                    merged_by_index[idx]["compliance"] = compliance_root
                _set_path_leaf(compliance_root, tuple(update["path"]), update["compliance"])
        final_entries = [
            merged_by_index[int(entry["rollout_index"])]
            for entry in sorted(oracle_rollout_entries, key=lambda e: int(e["rollout_index"]))
            if int(entry["rollout_index"]) in merged_by_index
        ]
        write_json(cache_file, final_entries)

    if dist_ctx is not None and dist_ctx.enabled:
        final_entries = broadcast_object(dist_ctx, final_entries, src=0)
    if final_entries is None:
        final_entries = []

    summary = _oracle_judge_summary(final_entries)
    summary.update(
        {
            "cache/oracle_judge_missing": float(len(pending_items)),
            "cache/oracle_judge_total_rollouts": float(len(oracle_rollout_entries)),
        }
    )
    return final_entries, cache_file, summary
