import sys
import json
from pathlib import Path
from typing import Any, Callable

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

def _find_subsequence_start(token_ids: list[int], pattern_ids: list[int]) -> int | None:
    if not pattern_ids or len(pattern_ids) > len(token_ids):
        return None
    width = len(pattern_ids)
    for i in range(0, len(token_ids) - width + 1):
        if token_ids[i : i + width] == pattern_ids:
            return i
    return None


def _find_last_subsequence_start(token_ids: list[int], pattern_ids: list[int]) -> int | None:
    if not pattern_ids or len(pattern_ids) > len(token_ids):
        return None
    width = len(pattern_ids)
    for i in range(len(token_ids) - width, -1, -1):
        if token_ids[i : i + width] == pattern_ids:
            return i
    return None


def _build_combined_points_spec(
    combined_text: str,
    prompt_len: int,
    combined_len: int,
    token_points: dict[str, int] | None = None,
) -> dict[str, Any]:
    rollout_len = combined_len - prompt_len
    if rollout_len <= 0:
        raise ValueError("Combined sequence has no rollout tokens to probe.")
    points = token_points or {}
    return {
        "combined_text": combined_text,
        "prompt_len": prompt_len,
        "combined_len": combined_len,
        "rollout_len": rollout_len,
        "prompt_segment": (0, prompt_len),
        "rollout_segment": (prompt_len, combined_len),
        "token_points": points,
        "token_point_indices": sorted(set(points.values())),
    }


def _combined_text_and_lengths(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    target_response: str,
) -> tuple[str, int, int]:
    prompt_ids = tokenizer(
        formatted_target_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    combined_text = formatted_target_prompt + target_response
    combined_ids = tokenizer(
        combined_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    return combined_text, int(prompt_ids.shape[0]), int(combined_ids.shape[0])


def extract_token_points_combined_default(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    target_response: str,
) -> dict[str, Any]:
    """
    Identify key boundaries for a combined prompt+response sequence.

    Returns token indices over the combined tokenization:
      - prompt segment: [0, prompt_len)
      - rollout segment: [prompt_len, combined_len)
      - token points:
          * last_prompt_token
          * first_rollout_token
          * last_rollout_token
    """
    combined_text, prompt_len, combined_len = _combined_text_and_lengths(
        tokenizer=tokenizer,
        formatted_target_prompt=formatted_target_prompt,
        target_response=target_response,
    )
    token_points = {
        "last_prompt_token": prompt_len - 1,
        "first_rollout_token": prompt_len,
        "last_rollout_token": combined_len - 1,
    }
    return _build_combined_points_spec(
        combined_text=combined_text,
        prompt_len=prompt_len,
        combined_len=combined_len,
        token_points=token_points,
    )


def extract_token_points_combined_qwen(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    target_response: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        formatted_target_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    combined_text = formatted_target_prompt + target_response
    combined_ids = tokenizer(
        combined_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()

    prompt_len = len(prompt_ids)
    combined_len = len(combined_ids)

    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)

    im_end_start = _find_last_subsequence_start(prompt_ids, im_end_ids)
    if im_end_start is None:
        raise ValueError("Could not find <|im_end|> token in formatted prompt.")
    im_end_after = im_end_start + len(im_end_ids)
    if im_end_start - 1 < 0 or im_end_after >= prompt_len:
        raise ValueError("Prompt too short to index around <|im_end|> token.")

    im_start_start = _find_last_subsequence_start(prompt_ids, im_start_ids)
    if im_start_start is None:
        raise ValueError("Could not find trailing <|im_start|> token in formatted prompt.")

    assistant_start = _find_last_subsequence_start(prompt_ids, assistant_ids)
    if assistant_start is None:
        raise ValueError("Could not find trailing assistant token in formatted prompt.")

    rollout_ids = combined_ids[prompt_len:]
    think_close_start_rollout = _find_last_subsequence_start(rollout_ids, think_close_ids)
    if think_close_start_rollout is None:
        raise ValueError("Could not find </think> token in rollout.")
    think_close_start = prompt_len + think_close_start_rollout
    first_after_think_close = think_close_start + len(think_close_ids)
    if first_after_think_close >= combined_len:
        raise ValueError("No token found after </think> in rollout.")

    token_points = {
        "im_end_token": im_end_start,
        "token_before_im_end": im_end_start - 1,
        "token_after_im_end": im_end_after,
        "trailing_im_start_token": im_start_start,
        "trailing_assistant_token": assistant_start,
        "last_prompt_token": prompt_len - 1,
        "first_rollout_token": prompt_len,
        "think_close_token": think_close_start,
        "first_token_after_think_close": first_after_think_close,
        "last_rollout_token": combined_len - 1,
    }
    return _build_combined_points_spec(
        combined_text=combined_text,
        prompt_len=prompt_len,
        combined_len=combined_len,
        token_points=token_points,
    )


TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME: dict[
    str,
    Callable[[AutoTokenizer, str, str], dict[str, Any]],
] = {
    "Qwen/Qwen3-8B": extract_token_points_combined_qwen,
}


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
    # Original segment mode compatibility ("segment")
    segment_start_idx: int = 0,
    segment_end_idx: int | None = None,
    # Original token range mode ("tokens")
    token_start_idx: int = 0,
    token_end_idx: int | None = None,
    # New fourth mode ("token_points")
    token_point_indices_by_target: list[list[int]] | None = None,
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
          * prompt-only: ["segment", "full_seq", "tokens"]
          * combined:    ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
      - If generation_kwargs is None:
          * oracle_repeats > 1: {"do_sample": True, "temperature": 1.0, "max_new_tokens": 128}
          * oracle_repeats == 1: {"do_sample": False, "temperature": 0.0, "max_new_tokens": 128}
      - If token_point_indices_by_target is None:
          * uses extractor-derived defaults per target from combined_specs
          * prompt-only targets use {"last_prompt_token": prompt_len - 1}

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
            oracle_input_types = ["segment", "full_seq", "tokens"]
        else:
            oracle_input_types = ["full_seq", "prompt_segment", "rollout_segment", "token_points"]
    if generation_kwargs is None:
        if oracle_repeats > 1:
            generation_kwargs = {"do_sample": True, "temperature": 1.0, "max_new_tokens": 128}
        else:
            generation_kwargs = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 128}

    combined_specs: list[dict[str, Any]] = []
    if target_responses is None:
        for prompt in formatted_target_prompts:
            prompt_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]
            prompt_len = int(prompt_ids.shape[0])
            if prompt_len <= 0:
                raise ValueError("Prompt has no tokens to probe.")
            token_points = {"last_prompt_token": prompt_len - 1}
            combined_specs.append({
                "combined_text": prompt,
                "prompt_len": prompt_len,
                "combined_len": prompt_len,
                "rollout_len": 0,
                "prompt_segment": (0, prompt_len),
                "rollout_segment": (prompt_len, prompt_len),
                "token_points": token_points,
                "token_point_indices": sorted(set(token_points.values())),
            })
    else:
        use_extractor_for_default_token_points = (
            "token_points" in oracle_input_types and token_point_indices_by_target is None
        )
        if use_extractor_for_default_token_points:
            extractor = TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME.get(
                model.config._name_or_path,
                extract_token_points_combined_default,
            )
            combined_specs = [
                extractor(tokenizer, prompt, response)
                for prompt, response in zip(formatted_target_prompts, target_responses, strict=True)
            ]
        else:
            for prompt, response in zip(formatted_target_prompts, target_responses, strict=True):
                combined_text, prompt_len, combined_len = _combined_text_and_lengths(
                    tokenizer=tokenizer,
                    formatted_target_prompt=prompt,
                    target_response=response,
                )
                combined_specs.append(
                    _build_combined_points_spec(
                        combined_text=combined_text,
                        prompt_len=prompt_len,
                        combined_len=combined_len,
                    )
                )
    combined_texts = [spec["combined_text"] for spec in combined_specs]

    if token_point_indices_by_target is None:
        token_point_indices_by_target = [spec["token_point_indices"] for spec in combined_specs]
    if len(token_point_indices_by_target) != len(combined_specs):
        raise ValueError("token_point_indices_by_target must match number of targets.")

    target_model_name = model.config._name_or_path
    oracle_model_name = model.config._name_or_path
    cache_paths: list[Path] = []
    cached_by_target: list[list[dict[str, Any]]] = []
    cached_counts_by_target: list[int] = []
    cache_status_by_target: list[tuple[int, int, int]] = []
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
        cache_paths.append(cache_path)
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
        cache_status_by_target.append((target_idx, cached_count, missing_count))

    full_hits = sum(1 for _, _, missing_count in cache_status_by_target if missing_count == 0)
    partial_hits = sum(1 for _, cached_count, missing_count in cache_status_by_target if cached_count > 0 and missing_count > 0)
    misses = sum(1 for _, cached_count, _ in cache_status_by_target if cached_count == 0)
    print(
        f"[oracle cache] targets={len(cache_status_by_target)} "
        f"requested_repeats={oracle_repeats} full_hits={full_hits} partial_hits={partial_hits} misses={misses}"
    )
    for target_idx, cached_count, missing_count in cache_status_by_target:
        if missing_count > 0:
            print(
                f"[oracle cache] target_idx={target_idx}: cached={cached_count}, missing={missing_count}"
            )

    missing_indices = [i for i, cached_count in enumerate(cached_counts_by_target) if cached_count < oracle_repeats]
    if not missing_indices:
        return [
            _aggregate_oracle_repeat_entries(cached_by_target[i][:oracle_repeats])
            for i in range(len(cached_by_target))
        ]

    model_name = model.config._name_or_path
    act_layer = layer_percent_to_layer(model_name, layer_percent)
    act_layers = [act_layer]

    if target_lora_path is not None:
        model.set_adapter(target_lora_path)
    else:
        model.set_adapter("default")

    missing_specs = [combined_specs[i] for i in missing_indices]
    missing_texts = [combined_texts[i] for i in missing_indices]
    missing_token_point_indices = [token_point_indices_by_target[i] for i in missing_indices]
    cached_counts_by_missing_target = [cached_counts_by_target[i] for i in missing_indices]
    total_missing_repeats = sum(oracle_repeats - cached_count for cached_count in cached_counts_by_missing_target)

    inputs_bl = _encode_formatted_prompts(tokenizer, missing_texts, device)
    submodules = {layer: get_hf_submodule(model, layer) for layer in act_layers}
    acts_by_layer = collect_activations_multiple_layers(
        model=model,
        submodules=submodules,
        inputs_BL=inputs_bl,
        min_offset=None,
        max_offset=None,
    )

    seq_len = int(inputs_bl["input_ids"].shape[1])
    attn = inputs_bl["attention_mask"]
    left_pads = [seq_len - int(attn[i].sum().item()) for i in range(attn.shape[0])]
    target_input_ids_by_target = [
        inputs_bl["input_ids"][i, left_pads[i] :].tolist()
        for i in range(attn.shape[0])
    ]

    injection_submodule = get_hf_submodule(model, injection_layer)
    updated_targets = 0
    for local_target_idx, spec in enumerate(missing_specs):
        target_input_ids = target_input_ids_by_target[local_target_idx]
        left_pad = left_pads[local_target_idx]
        prompt_start, prompt_end = spec["prompt_segment"]
        rollout_start, rollout_end = spec["rollout_segment"]
        total_tokens = len(target_input_ids)
        repeat_start = cached_counts_by_missing_target[local_target_idx]

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
                "target_idx": local_target_idx,
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
                seg_end = total_tokens if segment_end_idx is None else min(segment_end_idx, total_tokens)
                if seg_start < 0:
                    raise ValueError(f"segment_start_idx ({seg_start}) must be >= 0")
                if seg_start >= seg_end:
                    raise ValueError(f"segment_start_idx ({seg_start}) must be < segment_end_idx ({seg_end})")
                add_probe(list(range(seg_start, seg_end)), probe_kind="segment", repeat_idx=repeat_idx)
            if "prompt_segment" in oracle_input_types:
                add_probe(list(range(prompt_start, prompt_end)), probe_kind="prompt_segment", repeat_idx=repeat_idx)
            if "rollout_segment" in oracle_input_types:
                add_probe(list(range(rollout_start, rollout_end)), probe_kind="rollout_segment", repeat_idx=repeat_idx)
            if "tokens" in oracle_input_types:
                tok_start = token_start_idx
                tok_end = total_tokens if token_end_idx is None else min(token_end_idx, total_tokens)
                for token_idx in range(tok_start, tok_end):
                    add_probe([token_idx], probe_kind="tokens", repeat_idx=repeat_idx, token_index=token_idx)
            if "token_points" in oracle_input_types:
                for token_idx in sorted(set(missing_token_point_indices[local_target_idx])):
                    if token_idx < 0 or token_idx >= total_tokens:
                        continue
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

        global_idx = missing_indices[local_target_idx]
        append_entries = [target_outputs[idx] for idx in sorted(target_outputs.keys())]
        cached_by_target[global_idx].extend(append_entries)
        write_json(cache_paths[global_idx], cached_by_target[global_idx])
        updated_targets += 1
        print(
            f"[oracle cache] updated target_idx={global_idx} "
            f"(added_repeats={len(append_entries)}, total_cached={len(cached_by_target[global_idx])})"
        )
    print(f"[oracle cache] updated cache files for {updated_targets} target(s)")

    return [
        _aggregate_oracle_repeat_entries(cached_by_target[i][:oracle_repeats])
        for i in range(len(cached_by_target))
    ]
