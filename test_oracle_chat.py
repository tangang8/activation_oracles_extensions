#!/usr/bin/env python3
"""
Minimal oracle-as-target chat generation smoke test.

Loads Qwen3-8B + oracle LoRA, takes prompts from the harmful dataset,
and generates one stochastic completion per prompt in a single batched call.

Usage (from activation_oracles_extensions/):
  python test_oracle_chat.py
  python test_oracle_chat.py --num-prompts 3
  python test_oracle_chat.py --num-prompts 3
  # writes to activation_oracles_extensions/test_oracle_chat_output/oracle_thinking/ by default
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_loading_utils import AdapterSpec, load_model_stack
from prompt_utils import load_target_prompts_from_dataset, prompt_key
from rollout_utils import (
    THINKING_TAG_PATTERNS_BY_MODEL,
    format_user_target_prompt,
    validate_target_response_format,
)

MODEL_NAME = "Qwen/Qwen3-8B"
ORACLE_ADAPTER_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
ORACLE_ADAPTER_NAME = "oracle"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE_DIR = SCRIPT_DIR / "test_oracle_chat_output"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_BASE_DIR / "oracle_thinking"
WORKSPACE_ENV_PATH = SCRIPT_DIR.parent / ".env"
MAX_NEW_TOKENS = 10_000


def _load_workspace_env() -> None:
    load_dotenv(dotenv_path=str(WORKSPACE_ENV_PATH), override=False)


def _require_hf_token() -> str:
    _load_workspace_env()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(f"Set HF_TOKEN in {WORKSPACE_ENV_PATH}")
    return token


def _tag_counts(text: str, thinking_tag: dict[str, str] | None) -> dict[str, int]:
    counts: dict[str, int] = {
        "im_start_user": text.count("<|im_start|>user"),
        "im_start_assistant": text.count("<|im_start|>assistant"),
        "plain_user_line": len([line for line in text.splitlines() if line.strip() == "user"]),
        "plain_assistant_line": len([line for line in text.splitlines() if line.strip() == "assistant"]),
    }
    if thinking_tag is not None:
        counts["open_thinking"] = text.count(thinking_tag["open"])
        counts["close_thinking"] = text.count(thinking_tag["close"])
    return counts


def _print_format_validation(label: str, target_format: dict) -> bool:
    print(f"\n--- thinking format validation: {label} ---")
    print(f"valid_response_format: {target_format['valid_response_format']}")
    print(f"thinking chars: {len(target_format['thinking'])}")
    print(f"parsed_response chars: {len(target_format['parsed_response'])}")
    if not target_format["valid_response_format"]:
        print("Warning: malformed thinking tags (unclosed <think> block).")
    return bool(target_format["valid_response_format"])


def _check_validation(
    *,
    format_decoded: dict,
    format_special: dict,
    response: str,
    response_with_special: str,
    thinking_tag: dict[str, str] | None,
    require_parsed_response: bool,
    allow_missing_thinking_tags: bool,
) -> bool:
    valid_decoded = format_decoded["valid_response_format"]
    valid_special = format_special["valid_response_format"]
    validation_ok = valid_decoded and valid_special
    if require_parsed_response:
        if not format_decoded["parsed_response"].strip():
            print("Error: parsed response is empty (skip_special_tokens=True).", file=sys.stderr)
            validation_ok = False
        if not format_special["parsed_response"].strip():
            print("Error: parsed response is empty (skip_special_tokens=False).", file=sys.stderr)
            validation_ok = False
    if not allow_missing_thinking_tags and thinking_tag is not None:
        for label, text in (
            ("decoded", response),
            ("with_special", response_with_special),
        ):
            if _tag_counts(text, thinking_tag).get("open_thinking", 0) == 0:
                print(f"Error: no thinking tags found in {label} output.", file=sys.stderr)
                validation_ok = False
    return validation_ok


def _output_paths(output: Path | None, prompt_index: int, dataset_index: int, user_prompt: str) -> tuple[Path | None, Path | None]:
    if output is None:
        return None, None
    if prompt_index == 0 and output.suffix:
        base = output
    else:
        key = prompt_key(user_prompt, preview_len=24)
        base = output / f"prompt_{dataset_index:04d}_{key}.txt"
    special = base.with_suffix(base.suffix + ".with_special_tokens.txt")
    return base, special


def _batched_generate_dual_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    formatted_prompts: list[str],
    device: torch.device,
    lora_path: str | None,
    generation_kwargs: dict,
) -> tuple[list[str], list[str]]:
    """One model.generate over all prompts; decode with and without special tokens."""
    if lora_path is not None:
        model.set_adapter(lora_path)
    else:
        model.set_adapter("default")

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(device)
    tokenizer.padding_side = original_padding_side

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    with_special = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
    return decoded, with_special


def _process_prompt_result(
    *,
    prompt_index: int,
    dataset_index: int,
    user_prompt: str,
    formatted_prompt: str,
    response: str,
    response_with_special: str,
    thinking_tag: dict[str, str] | None,
    require_parsed_response: bool,
    allow_missing_thinking_tags: bool,
    output: Path | None,
) -> bool:
    print(f"\n{'=' * 72}")
    print(f"Prompt {prompt_index + 1} (dataset index {dataset_index}): {user_prompt!r}")
    print("\n--- formatted prompt (tail) ---")
    print(formatted_prompt[-500:])

    format_decoded = validate_target_response_format(response, thinking_tag)
    format_special = validate_target_response_format(response_with_special, thinking_tag)

    print(f"chars={len(response)}  lines={response.count(chr(10)) + 1}")
    print(f"tag counts (skip_special_tokens=True): {_tag_counts(response, thinking_tag)}")
    print(f"tag counts (skip_special_tokens=False): {_tag_counts(response_with_special, thinking_tag)}")

    _print_format_validation("decoded (skip_special_tokens=True)", format_decoded)
    _print_format_validation("decoded (skip_special_tokens=False)", format_special)

    validation_ok = _check_validation(
        format_decoded=format_decoded,
        format_special=format_special,
        response=response,
        response_with_special=response_with_special,
        thinking_tag=thinking_tag,
        require_parsed_response=require_parsed_response,
        allow_missing_thinking_tags=allow_missing_thinking_tags,
    )

    print("\n--- response preview (first 1200 chars, skip_special_tokens=True) ---")
    print(response[:1200])
    if len(response) > 1200:
        print(f"\n... [{len(response) - 1200} more chars] ...")
        print("\n--- response tail (last 800 chars) ---")
        print(response[-800:])

    out_path, special_path = _output_paths(output, prompt_index, dataset_index, user_prompt)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(response, encoding="utf-8")
        assert special_path is not None
        special_path.write_text(response_with_special, encoding="utf-8")
        print(f"\nWrote full response to: {out_path}")
        print(f"Wrote response with special tokens to: {special_path}")

    return validation_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-target chat generation smoke test.")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1,
        help="Number of dataset prompts to run in one batch (default: 1).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "Skip this many rows at the start of LLM-LAT/harmful-dataset (train split) "
            "before taking --num-prompts. Example: --offset 10 --num-prompts 3 uses "
            "dataset rows 10, 11, 12 (default: 0 = start at the first prompt)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory for response files (default: {DEFAULT_OUTPUT_DIR}). "
            "With --num-prompts 1 you may pass a single .txt file path instead."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write response files to disk.",
    )
    parser.add_argument(
        "--lora-adapter",
        "--target-lora",
        dest="lora_adapter",
        default=ORACLE_ADAPTER_NAME,
        help=(
            f"PEFT adapter active during generation (default: {ORACLE_ADAPTER_NAME!r} oracle LoRA). "
            "Pass 'default' for the empty base adapter only."
        ),
    )
    parser.add_argument(
        "--require-parsed-response",
        action="store_true",
        help="Fail validation if parsed content after thinking tags is empty.",
    )
    parser.add_argument(
        "--allow-missing-thinking-tags",
        action="store_true",
        help="Do not fail when no thinking tags are present.",
    )
    parser.add_argument(
        "--target-thinking",
        choices=("default", "off"),
        default="off",
        help=(
            "Thinking mode when formatting target prompts with apply_chat_template. "
            "'off' passes enable_thinking=False; 'default' leaves tokenizer defaults."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"Generation cap for each prompt completion (default: {MAX_NEW_TOKENS}).",
    )
    args = parser.parse_args()

    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be >= 1")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be >= 1")

    prompts = load_target_prompts_from_dataset(limit=args.num_prompts, offset=args.offset)
    if len(prompts) < args.num_prompts:
        raise RuntimeError(
            f"Requested {args.num_prompts} prompts at offset {args.offset}, "
            f"but only loaded {len(prompts)}."
        )

    output = None if args.no_save else args.output
    if output is not None:
        if args.num_prompts > 1 and output.suffix:
            raise ValueError(
                "With --num-prompts > 1, --output must be a directory (no file extension), "
                f"got: {output}"
            )
        if not output.suffix:
            output.mkdir(parents=True, exist_ok=True)
        print(f"Saving responses to: {output}")

    hf_token = _require_hf_token()
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map: str | dict[str, str] = "auto" if device.type == "cuda" else {"": str(device)}

    thinking_tag = THINKING_TAG_PATTERNS_BY_MODEL.get(MODEL_NAME)

    print(f"Running {args.num_prompts} prompt(s) in one batch (offset={args.offset})")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA adapter: {args.lora_adapter} (loaded from {ORACLE_ADAPTER_PATH})")
    print(f"Target thinking: {args.target_thinking}")
    print(f"Device: {device}  max_new_tokens: {args.max_new_tokens}")

    tokenizer, model = load_model_stack(
        model_name=MODEL_NAME,
        adapter_specs=[
            AdapterSpec(
                adapter_path=ORACLE_ADAPTER_PATH,
                adapter_name=ORACLE_ADAPTER_NAME,
            )
        ],
        torch_dtype=dtype,
        device_map=device_map,
        hf_token=hf_token,
    )
    model.set_adapter(args.lora_adapter)
    print(f"Active adapter after load: {args.lora_adapter}")

    generation_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }

    target_enable_thinking = False if args.target_thinking == "off" else None
    formatted_prompts = [
        format_user_target_prompt(tokenizer, p, enable_thinking=target_enable_thinking)
        for p in prompts
    ]

    t0 = perf_counter()
    responses, responses_with_special = _batched_generate_dual_decode(
        model=model,
        tokenizer=tokenizer,
        formatted_prompts=formatted_prompts,
        device=device,
        lora_path=args.lora_adapter,
        generation_kwargs=generation_kwargs,
    )
    elapsed = perf_counter() - t0
    print(f"\n--- batched generation finished in {elapsed:.2f}s for {len(prompts)} prompt(s) ---")

    if len(responses) != len(prompts):
        raise RuntimeError(f"Expected {len(prompts)} responses, got {len(responses)}")

    passed = 0
    for i, user_prompt in enumerate(prompts):
        ok = _process_prompt_result(
            prompt_index=i,
            dataset_index=args.offset + i,
            user_prompt=user_prompt,
            formatted_prompt=formatted_prompts[i],
            response=responses[i],
            response_with_special=responses_with_special[i],
            thinking_tag=thinking_tag,
            require_parsed_response=args.require_parsed_response,
            allow_missing_thinking_tags=args.allow_missing_thinking_tags,
            output=output,
        )
        if ok:
            passed += 1

    print(f"\n{'=' * 72}")
    print(f"Finished {args.num_prompts} prompt(s): {passed}/{args.num_prompts} passed validation")
    if passed < args.num_prompts:
        print("Thinking tag validation failed for one or more prompts.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
