import sys
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import directly from sibling activation_oracles repo.
_HERE = Path(__file__).resolve().parent
_ACTIVATION_ORACLES_ROOT = _HERE.parent / "activation_oracles"
if str(_ACTIVATION_ORACLES_ROOT) not in sys.path:
    sys.path.append(str(_ACTIVATION_ORACLES_ROOT))

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer
from nl_probes.utils.dataset_utils import create_training_datapoint
from nl_probes.utils.eval import run_evaluation
from cache_utils import load_json, oracle_cache_file_path, write_json
from distributed_utils import DistributedContext, all_gather_objects, broadcast_object
from oracle_token_points import (
    COMBINED_TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME,
    PROMPT_TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME,
    build_combined_points_spec,
    build_prompt_only_points_spec,
    extract_token_points_combined_default,
    extract_token_points_prompt_default,
)
from perf_utils import PerfLogger

VALID_ORACLE_INPUT_TYPES = {
    "full_seq",
    "segment",
    "prompt_segment",
    "rollout_segment",
    "tokens",
    "token_points",
}


@dataclass(frozen=True)
class OracleInputProvenance:
    source_type: str
    source_index_label: str
    source_index: int
    cache_path: Path


def _oracle_input_source(target_responses: list[str] | None) -> tuple[str, str]:
    if target_responses is None:
        return "prompt_only", "prompt_input_index"
    return "target_rollout", "target_rollout_index"


def _filter_token_points_post_prompt(spec: dict[str, Any]) -> dict[str, Any]:
    prompt_len = int(spec["prompt_len"])
    token_points = spec.get("token_points", {})
    if not isinstance(token_points, dict):
        return spec

    filtered_points = {
        str(name): int(idx)
        for name, idx in token_points.items()
        if int(idx) >= prompt_len
    }
    filtered_spec = dict(spec)
    filtered_spec["token_points"] = filtered_points
    filtered_spec["token_point_indices"] = sorted(set(filtered_points.values()))
    return filtered_spec


def _validate_oracle_probe_config(
    *,
    source_type: str,
    oracle_input_types: list[str],
    combined_specs: list[dict[str, Any]],
    token_point_indices_by_target: list[list[int]],
    segment_start_idx: int,
    segment_end_idx: int | None,
    token_start_idx: int,
    token_end_idx: int | None,
) -> None:
    invalid_input_types = [kind for kind in oracle_input_types if kind not in VALID_ORACLE_INPUT_TYPES]
    if invalid_input_types:
        raise ValueError(
            "Unsupported oracle_input_types value(s): "
            f"{', '.join(invalid_input_types)}. "
            f"Expected values from: {', '.join(sorted(VALID_ORACLE_INPUT_TYPES))}."
        )
    if source_type == "prompt_only" and "rollout_segment" in oracle_input_types:
        raise ValueError("oracle_input_types includes rollout_segment, but prompt-only inputs have no rollout tokens.")

    for target_idx, spec in enumerate(combined_specs):
        total_tokens = int(spec["combined_len"])
        prompt_len = int(spec["prompt_len"])
        rollout_len = int(spec["rollout_len"])
        label = f"{source_type} target_idx={target_idx}"

        if "prompt_segment" in oracle_input_types and prompt_len <= 0:
            raise ValueError(f"prompt_segment requested for {label}, but the prompt has no tokens.")
        if "rollout_segment" in oracle_input_types and rollout_len <= 0:
            raise ValueError(f"rollout_segment requested for {label}, but the rollout has no tokens.")

        if "segment" in oracle_input_types:
            segment_end = total_tokens if segment_end_idx is None else segment_end_idx
            if segment_start_idx < 0:
                raise ValueError(f"segment_start_idx ({segment_start_idx}) must be >= 0 for {label}.")
            if segment_end > total_tokens:
                raise ValueError(
                    f"segment_end_idx ({segment_end}) exceeds tokenized input length ({total_tokens}) for {label}."
                )
            if segment_start_idx >= segment_end:
                raise ValueError(
                    f"segment_start_idx ({segment_start_idx}) must be < segment_end_idx ({segment_end}) for {label}."
                )

        if "tokens" in oracle_input_types:
            token_end = total_tokens if token_end_idx is None else token_end_idx
            if token_start_idx < 0:
                raise ValueError(f"token_start_idx ({token_start_idx}) must be >= 0 for {label}.")
            if token_end > total_tokens:
                raise ValueError(
                    f"token_end_idx ({token_end}) exceeds tokenized input length ({total_tokens}) for {label}."
                )
            if token_start_idx >= token_end:
                raise ValueError(
                    f"token_start_idx ({token_start_idx}) must be < token_end_idx ({token_end}) for {label}."
                )

        if "token_points" in oracle_input_types:
            for token_idx in token_point_indices_by_target[target_idx]:
                if token_idx < 0 or token_idx >= total_tokens:
                    raise ValueError(
                        f"token point index ({token_idx}) is outside tokenized input length ({total_tokens}) "
                        f"for {label}."
                    )


def _encode_formatted_prompts(
    tokenizer: AutoTokenizer,
    formatted_prompts: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        formatted_prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(device)


def _aggregate_oracle_repeat_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeat-level oracle entries into the return format."""
    if not entries:
        return {}
    combined_text = entries[0].get("combined_text", "")
    points = entries[0].get("points", {})
    merged: dict[str, Any] = {
        "combined_text": combined_text,
        "points": points,
        "oracle_repeats": len(entries),
        "full_seq": [],
        "segment": [],
        "prompt_segment": [],
        "rollout_segment": [],
        "tokens": {},
        "token_points": {},
    }
    for entry in entries:
        for key in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
            merged[key].extend(entry.get(key, []))
        for token_key, responses in entry.get("tokens", {}).items():
            token_int = int(token_key)
            merged["tokens"].setdefault(token_int, []).extend(responses)
        for token_key, responses in entry.get("token_points", {}).items():
            token_int = int(token_key)
            merged["token_points"].setdefault(token_int, []).extend(responses)
    return merged

def run_oracle_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    formatted_target_prompts: list[str] | str,
    target_responses: list[str] | None,
    oracle_prompt: str,
    user_prompts: list[str] | str | None = None,
    *,
    cache_root: str = "cache",
    target_lora_path: str | None = None,
    oracle_lora_path: str | None = "oracle",
    generation_kwargs: dict[str, Any] | None = None,
    layer_percent: int = 50,
    injection_layer: int = 1,
    steering_coefficient: float = 1.0,
    eval_batch_size: int = 32,
    oracle_repeats: int = 1,
    oracle_input_types: list[str] | None = None,
    oracle_token_point_filter: str = "all",
    # Original segment mode compatibility ("segment")
    segment_start_idx: int = 0,
    segment_end_idx: int | None = None,
    # Original token range mode ("tokens")
    token_start_idx: int = 0,
    token_end_idx: int | None = None,
    # New fourth mode ("token_points")
    token_point_indices_by_target: list[list[int]] | None = None,
    oracle_input_source_type: str | None = None,
    dist_ctx: DistributedContext | None = None,
    perf: PerfLogger | None = None,
) -> list[dict[str, Any]]:
    """
    Batched single-layer oracle evaluation over target sequences.

    Inputs can be either:
      - prompt-only targets (target_responses is None), or
      - combined prompt+response targets (target_responses provided).
      - Optional user_prompts controls cache filename preview text (raw user prompt).

    Defaults applied when arguments are omitted:
      - If formatted_target_prompts is a single string:
          * prompt-only mode: wrapped as [formatted_target_prompts]
          * combined mode: repeated to match len(target_responses)
      - If oracle_input_types is None:
          * prompt-only: ["full_seq", "token_points"]
          * combined:    ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
      - If generation_kwargs is None:
          * oracle_repeats > 1: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 1000}
          * oracle_repeats == 1: {"do_sample": False, "temperature": 0.0, "max_new_tokens": 1000}
      - If token_point_indices_by_target is None:
          * uses extractor-derived defaults per target from combined_specs
          * prompt-only targets use per-model prompt token-point extractors
      - oracle_token_point_filter="post_prompt" keeps only extractor-derived
        target-rollout token points whose index is after the formatted target
        prompt boundary.

    Supported oracle_input_types:
      - "full_seq"        : full combined sequence
      - "segment"         : contiguous segment (segment_start_idx:segment_end_idx)
      - "prompt_segment"  : formatted target prompt segment only
      - "rollout_segment" : response/rollout segment only
      - "tokens"          : contiguous token-by-token range (token_start_idx:token_end_idx)
      - "token_points"    : sparse token index probes from token_point_indices_by_target

    Cache layout per target rollout:
      cache/target_{target_model}[_lora-{target_lora}]/
      oracle_rollouts_temp-{temperature}/oracle_{oracle_model}[_lora-{oracle_lora}]/
      {oracle_prompt_preview_hash}/{user_prompt_preview_hash}/{cache_key_hash}.json
      where each JSON file stores a list of dicts (one dict per oracle repeat).
    """
    if oracle_repeats <= 0:
        raise ValueError(f"oracle_repeats must be > 0, got {oracle_repeats}")
    if oracle_token_point_filter not in {"all", "post_prompt"}:
        raise ValueError(
            f"Unsupported oracle_token_point_filter={oracle_token_point_filter!r}. "
            "Expected one of: all, post_prompt."
        )
    if isinstance(formatted_target_prompts, str):
        if target_responses is None:
            formatted_target_prompts = [formatted_target_prompts]
        else:
            formatted_target_prompts = [formatted_target_prompts] * len(target_responses)

    if user_prompts is None:
        user_prompts_list = formatted_target_prompts
    elif isinstance(user_prompts, str):
        if target_responses is None:
            user_prompts_list = [user_prompts]
        else:
            user_prompts_list = [user_prompts] * len(target_responses)
    else:
        user_prompts_list = user_prompts

    if target_responses is not None and len(formatted_target_prompts) != len(target_responses):
        raise ValueError("formatted_target_prompts and target_responses must have same length.")
    if len(user_prompts_list) != len(formatted_target_prompts):
        raise ValueError("user_prompts must match number of targets.")
    if not formatted_target_prompts:
        return []

    if oracle_input_types is None:
        if target_responses is None:
            oracle_input_types = ["full_seq", "token_points"]
        else:
            oracle_input_types = ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
    if generation_kwargs is None:
        if oracle_repeats > 1:
            generation_kwargs = {"do_sample": True, "temperature": 1.0, "max_new_tokens": 1000}
        else:
            generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 1000}
    inferred_source_type, source_index_label = _oracle_input_source(target_responses)
    source_type = oracle_input_source_type or inferred_source_type
    if source_type not in {"prompt_only", "target_rollout"}:
        raise ValueError(
            f"Unsupported oracle_input_source_type={source_type!r}. Expected 'prompt_only' or 'target_rollout'."
        )
    if source_type == "prompt_only" and target_responses is not None:
        raise ValueError("oracle_input_source_type=prompt_only requires target_responses=None.")
    if source_type == "target_rollout" and target_responses is None:
        raise ValueError("oracle_input_source_type=target_rollout requires target_responses to be provided.")

    combined_specs: list[dict[str, Any]] = []
    if target_responses is None:
        use_extractor_for_default_token_points = (
            "token_points" in oracle_input_types and token_point_indices_by_target is None
        )
        if use_extractor_for_default_token_points:
            extractor = PROMPT_TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME.get(
                model.config._name_or_path,
                extract_token_points_prompt_default,
            )
            combined_specs = [
                extractor(tokenizer, prompt)
                for prompt in formatted_target_prompts
            ]
        else:
            combined_specs = [
                build_prompt_only_points_spec(
                    tokenizer=tokenizer,
                    formatted_target_prompt=prompt,
                )
                for prompt in formatted_target_prompts
            ]
    else:
        use_extractor_for_default_token_points = (
            "token_points" in oracle_input_types and token_point_indices_by_target is None
        )
        if use_extractor_for_default_token_points:
            extractor = COMBINED_TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME.get(
                model.config._name_or_path,
                extract_token_points_combined_default,
            )
            combined_specs = [
                extractor(tokenizer, prompt, response)
                for prompt, response in zip(formatted_target_prompts, target_responses, strict=True)
            ]
        else:
            combined_specs = [
                build_combined_points_spec(
                    tokenizer=tokenizer,
                    formatted_target_prompt=prompt,
                    target_response=response,
                )
                for prompt, response in zip(formatted_target_prompts, target_responses, strict=True)
            ]
        if oracle_token_point_filter == "post_prompt":
            combined_specs = [_filter_token_points_post_prompt(spec) for spec in combined_specs]
    combined_texts = [spec["combined_text"] for spec in combined_specs]

    if token_point_indices_by_target is None:
        token_point_indices_by_target = [spec["token_point_indices"] for spec in combined_specs]
    if len(token_point_indices_by_target) != len(combined_specs):
        raise ValueError("token_point_indices_by_target must match number of targets.")
    _validate_oracle_probe_config(
        source_type=source_type,
        oracle_input_types=oracle_input_types,
        combined_specs=combined_specs,
        token_point_indices_by_target=token_point_indices_by_target,
        segment_start_idx=segment_start_idx,
        segment_end_idx=segment_end_idx,
        token_start_idx=token_start_idx,
        token_end_idx=token_end_idx,
    )

    target_model_name = model.config._name_or_path
    oracle_model_name = model.config._name_or_path
    is_main = dist_ctx is None or not dist_ctx.enabled or dist_ctx.is_main
    rank = dist_ctx.rank if dist_ctx is not None else 0
    world_size = dist_ctx.world_size if dist_ctx is not None else 1

    def maybe_log(msg: str) -> None:
        if is_main:
            print(msg)

    provenance_by_input: list[OracleInputProvenance] = []
    cached_by_target: list[list[dict[str, Any]]] = []
    cached_counts_by_target: list[int] = []
    cache_status_by_target: list[tuple[OracleInputProvenance, int, int]] = []
    for target_idx, spec in enumerate(combined_specs):
        cache_key = {
            "oracle_prompt": oracle_prompt,
            "combined_text": spec["combined_text"],
            "oracle_input_types": oracle_input_types,
            "segment_start_idx": segment_start_idx,
            "segment_end_idx": segment_end_idx,
            "token_start_idx": token_start_idx,
            "token_end_idx": token_end_idx,
            "token_point_indices": token_point_indices_by_target[target_idx],
            "layer_percent": layer_percent,
            "injection_layer": injection_layer,
            "steering_coefficient": steering_coefficient,
        }
        if oracle_token_point_filter != "all":
            cache_key["oracle_token_point_filter"] = oracle_token_point_filter
        cache_key_text = json.dumps(cache_key, sort_keys=True, ensure_ascii=True)
        cache_path = oracle_cache_file_path(
            cache_root=cache_root,
            target_model_name=target_model_name,
            target_lora_path=target_lora_path,
            oracle_model_name=oracle_model_name,
            oracle_lora_path=oracle_lora_path,
            generation_kwargs=generation_kwargs,
            oracle_prompt=oracle_prompt,
            user_prompt_preview_text=user_prompts_list[target_idx],
            cache_key_text=cache_key_text,
        )
        provenance = OracleInputProvenance(
            source_type=source_type,
            source_index_label=source_index_label,
            source_index=target_idx,
            cache_path=cache_path,
        )
        provenance_by_input.append(provenance)
        cached = load_json(cache_path)
        cached_entries = (
            cached
            if isinstance(cached, list) and all(isinstance(entry, dict) for entry in cached)
            else []
        )
        cached_by_target.append(cached_entries)
        cached_count = len(cached_entries)
        cached_counts_by_target.append(cached_count)
        missing_count = max(0, oracle_repeats - cached_count)
        cache_status_by_target.append((provenance, cached_count, missing_count))

    full_hits = sum(1 for _, _, missing_count in cache_status_by_target if missing_count == 0)
    partial_hits = sum(1 for _, cached_count, missing_count in cache_status_by_target if cached_count > 0 and missing_count > 0)
    misses = sum(1 for _, cached_count, _ in cache_status_by_target if cached_count == 0)
    if perf is not None:
        missing_repeats = sum(missing_count for _, _, missing_count in cache_status_by_target)
        perf.log_event(
            "cache/oracle_status",
            {
                "cache/oracle_full_hits": float(full_hits),
                "cache/oracle_partial_hits": float(partial_hits),
                "cache/oracle_misses": float(misses),
                "cache/oracle_missing_repeats": float(missing_repeats),
                "cache/oracle_targets": float(len(cache_status_by_target)),
            },
            metadata={"rank": rank, "world_size": world_size},
        )
    maybe_log(
        f"[oracle cache] source={source_type} oracle_inputs={len(cache_status_by_target)} "
        f"requested_repeats={oracle_repeats} full_hits={full_hits} partial_hits={partial_hits} misses={misses}"
    )
    missing_status = [
        (provenance, cached_count, missing_count)
        for provenance, cached_count, missing_count in cache_status_by_target
        if missing_count > 0
    ]
    if missing_status:
        preview = ", ".join(
            (
                f"{provenance.source_index_label}={provenance.source_index}"
                f"(cached={cached},missing={missing})"
            )
            for provenance, cached, missing in missing_status[:10]
        )
        if len(missing_status) > 10:
            maybe_log(
                f"[oracle cache] missing oracle inputs: {len(missing_status)} total; first 10 -> {preview}"
            )
        else:
            maybe_log(f"[oracle cache] missing oracle inputs ({len(missing_status)}): {preview}")

    missing_indices = [i for i, cached_count in enumerate(cached_counts_by_target) if cached_count < oracle_repeats]
    if not missing_indices:
        return [
            _aggregate_oracle_repeat_entries(cached_by_target[i][:oracle_repeats])
            for i in range(len(cached_by_target))
        ]

    if dist_ctx is None or not dist_ctx.enabled:
        assigned_missing_indices = missing_indices
    else:
        assigned_missing_indices = [
            target_idx
            for local_idx, target_idx in enumerate(missing_indices)
            if local_idx % dist_ctx.world_size == dist_ctx.rank
        ]

    model_name = model.config._name_or_path
    act_layer = layer_percent_to_layer(model_name, layer_percent)
    act_layers = [act_layer]

    local_updates: dict[int, list[dict[str, Any]]] = {}
    if assigned_missing_indices:
        if target_lora_path is not None:
            model.set_adapter(target_lora_path)
        else:
            model.set_adapter("default")

        assigned_specs = [combined_specs[i] for i in assigned_missing_indices]
        assigned_texts = [combined_texts[i] for i in assigned_missing_indices]
        assigned_token_point_indices = [token_point_indices_by_target[i] for i in assigned_missing_indices]
        assigned_cached_counts = [cached_counts_by_target[i] for i in assigned_missing_indices]

        with (
            perf.track(
                "oracle/collect_activations",
                {
                    "assigned_targets": len(assigned_texts),
                    "layer_percent": layer_percent,
                    "act_layer": act_layer,
                    "rank": rank,
                    "world_size": world_size,
                },
            )
            if perf
            else nullcontext()
        ) as act_metrics:
            inputs_bl = _encode_formatted_prompts(tokenizer, assigned_texts, device)
            submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
            acts_by_layer = collect_activations_multiple_layers(
                model=model,
                submodules=submodules,
                inputs_BL=inputs_bl,
                min_offset=None,
                max_offset=None,
            )
            if perf:
                act_metrics["oracle/assigned_targets"] = float(len(assigned_texts))

        seq_len = int(inputs_bl["input_ids"].shape[1])
        attn = inputs_bl["attention_mask"]
        left_pads = [seq_len - int(attn[i].sum().item()) for i in range(attn.shape[0])]
        target_input_ids_by_target = [
            inputs_bl["input_ids"][i, left_pads[i] :].tolist()
            for i in range(attn.shape[0])
        ]

        injection_submodule = get_hf_submodule(model, injection_layer)
        for local_target_idx, spec in enumerate(assigned_specs):
            global_target_idx = assigned_missing_indices[local_target_idx]
            target_input_ids = target_input_ids_by_target[local_target_idx]
            left_pad = left_pads[local_target_idx]
            prompt_start, prompt_end = spec["prompt_segment"]
            rollout_start, rollout_end = spec["rollout_segment"]
            total_tokens = len(target_input_ids)
            repeat_start = assigned_cached_counts[local_target_idx]

            target_oracle_inputs = []

            def add_probe(
                positions_rel: list[int],
                probe_kind: str,
                repeat_idx: int,
                token_index: int | None = None,
            ) -> None:
                if not positions_rel:
                    return
                positions_abs = [left_pad + p for p in positions_rel]
                acts_bd = acts_by_layer[act_layer][local_target_idx, positions_abs]
                meta = {
                    "target_idx": global_target_idx,
                    "probe_kind": probe_kind,
                    "repeat_idx": repeat_idx,
                    "token_index": token_index,
                }
                dp = create_training_datapoint(
                    datapoint_type="N/A",
                    prompt=oracle_prompt,
                    target_response="N/A",
                    layer=act_layer,
                    num_positions=len(positions_rel),
                    tokenizer=tokenizer,
                    acts_BD=acts_bd,
                    feature_idx=-1,
                    context_input_ids=target_input_ids,
                    context_positions=positions_rel,
                    ds_label="N/A",
                    meta_info=meta,
                )
                target_oracle_inputs.append(dp)

            for repeat_idx in range(repeat_start, oracle_repeats):
                if "full_seq" in oracle_input_types:
                    add_probe(list(range(total_tokens)), probe_kind="full_seq", repeat_idx=repeat_idx)
                if "segment" in oracle_input_types:
                    seg_start = segment_start_idx
                    seg_end = total_tokens if segment_end_idx is None else segment_end_idx
                    add_probe(list(range(seg_start, seg_end)), probe_kind="segment", repeat_idx=repeat_idx)
                if "prompt_segment" in oracle_input_types:
                    add_probe(list(range(prompt_start, prompt_end)), probe_kind="prompt_segment", repeat_idx=repeat_idx)
                if "rollout_segment" in oracle_input_types:
                    add_probe(list(range(rollout_start, rollout_end)), probe_kind="rollout_segment", repeat_idx=repeat_idx)
                if "tokens" in oracle_input_types:
                    tok_start = token_start_idx
                    tok_end = total_tokens if token_end_idx is None else token_end_idx
                    for token_idx in range(tok_start, tok_end):
                        add_probe([token_idx], probe_kind="tokens", repeat_idx=repeat_idx, token_index=token_idx)
                if "token_points" in oracle_input_types:
                    for token_idx in sorted(set(assigned_token_point_indices[local_target_idx])):
                        add_probe([token_idx], probe_kind="token_points", repeat_idx=repeat_idx, token_index=token_idx)

            target_outputs: dict[int, dict[str, Any]] = {
                repeat_idx: {
                    "combined_text": spec["combined_text"],
                    "points": spec,
                    "full_seq": [],
                    "segment": [],
                    "prompt_segment": [],
                    "rollout_segment": [],
                    "tokens": {},
                    "token_points": {},
                }
                for repeat_idx in range(repeat_start, oracle_repeats)
            }

            if target_oracle_inputs:
                with (
                    perf.track(
                        "oracle/run_evaluation",
                        {
                            "eval_batch_size": eval_batch_size,
                            "num_probe_inputs": len(target_oracle_inputs),
                            "oracle_repeats": oracle_repeats,
                            "max_new_tokens": generation_kwargs.get("max_new_tokens"),
                            "rank": rank,
                            "world_size": world_size,
                        },
                    )
                    if perf
                    else nullcontext()
                ) as eval_metrics:
                    eval_t0 = perf_counter()
                    target_responses = run_evaluation(
                        eval_data=target_oracle_inputs,
                        model=model,
                        tokenizer=tokenizer,
                        submodule=injection_submodule,
                        device=device,
                        dtype=torch.bfloat16,
                        global_step=0,
                        lora_path=oracle_lora_path,
                        eval_batch_size=eval_batch_size,
                        steering_coefficient=steering_coefficient,
                        generation_kwargs=generation_kwargs,
                        verbose=False,
                    )
                    if perf:
                        elapsed = max(perf_counter() - eval_t0, 1e-12)
                        eval_metrics["throughput/probe_inputs_per_second"] = (
                            float(len(target_oracle_inputs)) / elapsed
                        )
                        eval_metrics["throughput/targets_per_second"] = 1.0 / elapsed
                for r in target_responses:
                    repeat_idx = int(r.meta_info["repeat_idx"])
                    probe_kind = str(r.meta_info["probe_kind"])
                    token_index = r.meta_info.get("token_index")
                    repeat_entry = target_outputs[repeat_idx]
                    if probe_kind in ("full_seq", "segment", "prompt_segment", "rollout_segment"):
                        repeat_entry[probe_kind].append(r.api_response)
                    elif probe_kind in ("tokens", "token_points"):
                        token_key = int(token_index)
                        probe_bucket = repeat_entry[probe_kind]
                        if token_key not in probe_bucket:
                            probe_bucket[token_key] = []
                        probe_bucket[token_key].append(r.api_response)

            append_entries = [target_outputs[idx] for idx in sorted(target_outputs.keys())]
            local_updates[global_target_idx] = append_entries
            if append_entries:
                # Checkpoint each completed target immediately so long runs can resume
                # even if interrupted before the final gather/merge phase.
                checkpoint_entries = cached_by_target[global_target_idx] + append_entries
                provenance = provenance_by_input[global_target_idx]
                write_json(provenance.cache_path, checkpoint_entries)
                maybe_log(
                    f"[oracle cache] checkpoint {provenance.source_index_label}={provenance.source_index} "
                    f"(added_repeats={len(append_entries)}, checkpoint_total={len(checkpoint_entries)}) "
                    f"cache_file={provenance.cache_path}"
                )

    if dist_ctx is not None and dist_ctx.enabled:
        gathered_updates = all_gather_objects(dist_ctx, local_updates)
    else:
        gathered_updates = [local_updates]

    final_results: list[dict[str, Any]] | None = None
    if is_main:
        updated_targets = 0
        for rank_updates in gathered_updates:
            if not isinstance(rank_updates, dict):
                continue
            for global_idx, append_entries in rank_updates.items():
                if not append_entries:
                    continue
                cached_by_target[global_idx].extend(append_entries)
                provenance = provenance_by_input[global_idx]
                write_json(provenance.cache_path, cached_by_target[global_idx])
                updated_targets += 1
                maybe_log(
                    f"[oracle cache] updated {provenance.source_index_label}={provenance.source_index} "
                    f"(added_repeats={len(append_entries)}, total_cached={len(cached_by_target[global_idx])}) "
                    f"cache_file={provenance.cache_path}"
                )
        maybe_log(f"[oracle cache] updated cache files for {updated_targets} oracle input(s)")

        final_results = [
            _aggregate_oracle_repeat_entries(cached_by_target[i][:oracle_repeats])
            for i in range(len(cached_by_target))
        ]

    if dist_ctx is not None and dist_ctx.enabled:
        final_results = broadcast_object(dist_ctx, final_results, src=0)

    return final_results
