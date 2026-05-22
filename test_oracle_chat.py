#!/usr/bin/env python3
"""
Minimal oracle-as-target chat generation smoke test.

Loads Qwen3-8B + oracle LoRA, takes the first harmful-dataset prompt (cyberbullying),
and generates one stochastic completion with max_new_tokens=10000.

Usage (from activation_oracles_extensions/):
  python test_oracle_chat.py
  python test_oracle_chat.py --output /tmp/oracle_chat_response.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import torch
from dotenv import load_dotenv

from model_loading_utils import AdapterSpec, load_model_stack
from prompt_utils import load_target_prompts_from_dataset
from rollout_utils import (
    THINKING_TAG_PATTERNS_BY_MODEL,
    format_user_target_prompt,
    validate_target_response_format,
)

MODEL_NAME = "Qwen/Qwen3-8B"
ORACLE_ADAPTER_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"
ORACLE_ADAPTER_NAME = "oracle"
WORKSPACE_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
MAX_NEW_TOKENS = 10_000


def _load_workspace_env() -> None:
    load_dotenv(dotenv_path=str(WORKSPACE_ENV_PATH), override=False)


def _require_hf_token() -> str:
    _load_workspace_env()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError(f"Set HF_TOKEN in {WORKSPACE_ENV_PATH}")
    return token


def _tag_counts(text: str) -> dict[str, int]:
    return {
        "open_thinking": text.count("<think>"),
        "close_thinking": text.count("</think>"),
        "im_start_user": text.count("<|im_start|>user"),
        "im_start_assistant": text.count("<|im_start|>assistant"),
        "plain_user_line": len([line for line in text.splitlines() if line.strip() == "user"]),
        "plain_assistant_line": len([line for line in text.splitlines() if line.strip() == "assistant"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single oracle-target chat generation smoke test.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full decoded response (skip_special_tokens=True).",
    )
    parser.add_argument(
        "--target-lora",
        default=ORACLE_ADAPTER_NAME,
        help=f"PEFT adapter name for generation (default: {ORACLE_ADAPTER_NAME}).",
    )
    args = parser.parse_args()

    prompts = load_target_prompts_from_dataset(limit=1, offset=0)
    if not prompts:
        raise RuntimeError("No prompts loaded from LLM-LAT/harmful-dataset (offset=0, limit=1).")
    user_prompt = prompts[0]

    hf_token = _require_hf_token()
    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_map: str | dict[str, str] = "auto" if device.type == "cuda" else {"": str(device)}

    print(f"Dataset prompt[0]: {user_prompt!r}")
    print(f"Model: {MODEL_NAME}")
    print(f"Adapter: {args.target_lora} ({ORACLE_ADAPTER_PATH})")
    print(f"Device: {device}  max_new_tokens: {MAX_NEW_TOKENS}")

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

    formatted_prompt = format_user_target_prompt(tokenizer, user_prompt)
    print("\n--- formatted prompt (tail) ---")
    print(formatted_prompt[-500:])

    generation_kwargs = {
        "do_sample": True,
        "temperature": 1.0,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    if args.target_lora is not None:
        model.set_adapter(args.target_lora)
    else:
        model.set_adapter("default")

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    t0 = perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
            num_return_sequences=1,
        )
    elapsed = perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    generated_tokens = output_ids[0, prompt_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response_with_special = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    thinking_tag = THINKING_TAG_PATTERNS_BY_MODEL.get(MODEL_NAME)
    target_format = validate_target_response_format(response, thinking_tag)

    print(f"\n--- generation finished in {elapsed:.2f}s ---")
    print(f"generated_tokens={generated_tokens.shape[0]}  chars={len(response)}  lines={response.count(chr(10)) + 1}")
    print(f"tag counts (skip_special_tokens=True): {_tag_counts(response)}")
    print(f"tag counts (skip_special_tokens=False): {_tag_counts(response_with_special)}")
    print(f"valid_response_format: {target_format['valid_response_format']}")
    print(f"parsed_response chars: {len(target_format['parsed_response'])}")
    print(f"thinking chars: {len(target_format['thinking'])}")

    print("\n--- response preview (first 1200 chars, skip_special_tokens=True) ---")
    print(response[:1200])
    if len(response) > 1200:
        print(f"\n... [{len(response) - 1200} more chars] ...")
        print("\n--- response tail (last 800 chars) ---")
        print(response[-800:])

    print("\n--- generated tokens with special tokens preserved (first 1200 chars) ---")
    print(response_with_special[:1200])

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(response, encoding="utf-8")
        special_path = args.output.with_suffix(args.output.suffix + ".with_special_tokens.txt")
        special_path.write_text(response_with_special, encoding="utf-8")
        print(f"\nWrote full response to: {args.output}")
        print(f"Wrote response with special tokens to: {special_path}")


if __name__ == "__main__":
    main()
