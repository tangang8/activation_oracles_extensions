import os
from contextlib import nullcontext
from time import perf_counter
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_loading_utils import AdapterSpec, load_model_stack
from distributed_utils import DistributedContext, broadcast_object, cleanup_distributed, init_distributed
from judge_instruction_utils import load_judge_instruction
from oracle_judge_utils import judge_deterministic_oracle_rollouts
from oracle_rollout_utils import generate_deterministic_oracle_rollouts
from perf_utils import PerfLogger, build_perf_logger
from prompt_utils import load_oracle_prompts_from_file, load_target_prompts_from_dataset, prompt_key
from report_pages import save_oracle_rollouts_html, save_rollouts_html
from rollout_utils import (
    display_rollout_results,
    format_user_target_prompt,
    generate_target_rollouts,
    judge_target_rollouts,
)
from wandb_utils import init_wandb_run, log_oracle_judge_metrics, log_oracle_metrics, log_rollout_metrics

dtype = torch.bfloat16

# Model configuration
QWEN_MODEL_NAME = "Qwen/Qwen3-8B"
QWEN_ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B"

# Configuration for the Llama EM model
LLAMA_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_ORACLE_LORA_PATH = "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct"
JUDGE_INSTRUCTION_PATH = "user_request_fulfillment.jinja2"
DEFAULT_ORACLE_PROMPTS_PATH = (
    Path(__file__).resolve().parent
    / "prompts"
    / "oracle_prompts"
    / "default_oracle_prompts.json"
)

# Load .env from workspace root (parent of this repo): /workspace/.env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=str(env_path))
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN in your <parent_dir>/.env file"


MAIN = __name__ == "__main__"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def run_pipeline_for_target_prompt(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ctx: DistributedContext,
    wandb_run: Any | None,
    perf: PerfLogger | None,
    target_prompt_str: str,
    target_prompt_index: int,
    oracle_prompts: list[str],
    num_rollouts: int,
    eval_batch_size: int,
    oracle_judge_batch_size: int,
    generation_kwargs_stochastic: dict[str, Any],
    oracle_generation_kwargs_deterministic: dict[str, Any],
    judge_instruction_file: str,
    judge_instruction_stem: str,
    judge_instruction_template: str,
) -> int:
    target_key = prompt_key(target_prompt_str)
    formatted_target_prompt = format_user_target_prompt(tokenizer, target_prompt_str)

    if ctx.is_main:
        print(f"[stage target rollout] target_prompt_index={target_prompt_index} key={target_key}")
    with (
        perf.track(
            "stage/target_rollouts_total",
            {
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "target_prompt_index": target_prompt_index,
                "target_prompt_key": target_key,
            },
        )
        if perf
        else nullcontext()
    ):
        target_rollout_entries, target_cache_file = generate_target_rollouts(
            model=model,
            tokenizer=tokenizer,
            formatted_target_prompt=formatted_target_prompt,
            user_prompt=target_prompt_str,
            generation_kwargs_stochastic=generation_kwargs_stochastic,
            num_rollouts=num_rollouts,
            device=ctx.device,
            target_lora_path="default",
            cache_root="cache",
            dist_ctx=ctx,
            perf=perf,
        )

    if ctx.is_main:
        print(f"[stage target judging] target_prompt_index={target_prompt_index} key={target_key}")
    with (
        perf.track(
            "stage/judge_rollouts_total",
            {
                "rank": ctx.rank,
                "world_size": ctx.world_size,
                "target_prompt_index": target_prompt_index,
                "target_prompt_key": target_key,
            },
        )
        if perf
        else nullcontext()
    ):
        judged_rollout_entries, judge_cache_file, compliance_results = judge_target_rollouts(
            judge_model=model,
            judge_tokenizer=tokenizer,
            user_prompt=target_prompt_str,
            target_rollout_entries=target_rollout_entries,
            judge_instruction_template=judge_instruction_template,
            judge_instruction_file=judge_instruction_file,
            judge_instruction_stem=judge_instruction_stem,
            device=ctx.device,
            target_model_name=model.config._name_or_path,
            target_lora_path="default",
            judge_lora_path="default",
            cache_root="cache",
            dist_ctx=ctx,
            perf=perf,
        )

    if ctx.is_main:
        with (
            perf.track(
                "stage/reporting_rollouts",
                {
                    "rank": ctx.rank,
                    "target_prompt_index": target_prompt_index,
                    "target_prompt_key": target_key,
                },
            )
            if perf
            else nullcontext()
        ):
            display_rollout_results(
                judged_rollout_entries,
                compliance_results,
                cache_file=judge_cache_file,
            )
            log_rollout_metrics(wandb_run, judged_rollout_entries, compliance_results)
            rollouts_report_path = save_rollouts_html(
                rollout_entries=judged_rollout_entries,
                compliance_results=compliance_results,
                output_path=f"rollouts_report_{target_key}.html",
            )
            print(f"[target {target_prompt_index}] Saved rollouts report: {rollouts_report_path}")
            print(f"[target {target_prompt_index}] Target rollouts cache: {target_cache_file}")
            print(f"[target {target_prompt_index}] Judge rollouts cache: {judge_cache_file}")

    if not judged_rollout_entries:
        if ctx.is_main:
            print(f"[target {target_prompt_index}] Skipping oracle stages: no judged rollouts.")
        return 0

    combinations_processed = 0
    for oracle_prompt_index, oracle_prompt in enumerate(oracle_prompts):
        oracle_key = prompt_key(oracle_prompt)
        t0 = perf_counter()
        if ctx.is_main:
            print(
                f"[stage oracle rollout] target_prompt_index={target_prompt_index} "
                f"oracle_prompt_index={oracle_prompt_index} target_key={target_key} oracle_key={oracle_key}"
            )
        with (
            perf.track(
                "stage/oracle_deterministic_total",
                {
                    "rank": ctx.rank,
                    "world_size": ctx.world_size,
                    "target_prompt_index": target_prompt_index,
                    "target_prompt_key": target_key,
                    "oracle_prompt_index": oracle_prompt_index,
                    "oracle_prompt_key": oracle_key,
                },
            )
            if perf
            else nullcontext()
        ):
            oracle_rollout_entries, oracle_cache_file, oracle_cache_stats = generate_deterministic_oracle_rollouts(
                model=model,
                tokenizer=tokenizer,
                device=ctx.device,
                oracle_prompt=oracle_prompt,
                target_rollout_entries=judged_rollout_entries,
                target_model_name=model.config._name_or_path,
                target_lora_path="default",
                oracle_lora_path="oracle",
                cache_root="cache",
                oracle_generation_kwargs=oracle_generation_kwargs_deterministic,
                eval_batch_size=eval_batch_size,
                dist_ctx=ctx,
                perf=perf,
            )
        if ctx.is_main:
            print(
                f"[stage oracle judging] target_prompt_index={target_prompt_index} "
                f"oracle_prompt_index={oracle_prompt_index} target_key={target_key} oracle_key={oracle_key}"
            )
        with (
            perf.track(
                "stage/judge_oracle_rollouts_total",
                {
                    "rank": ctx.rank,
                    "world_size": ctx.world_size,
                    "target_prompt_index": target_prompt_index,
                    "target_prompt_key": target_key,
                    "oracle_prompt_index": oracle_prompt_index,
                    "oracle_prompt_key": oracle_key,
                },
            )
            if perf
            else nullcontext()
        ):
            judged_oracle_entries, oracle_judge_cache_file, oracle_judge_summary = (
                judge_deterministic_oracle_rollouts(
                    judge_model=model,
                    judge_tokenizer=tokenizer,
                    oracle_rollout_entries=oracle_rollout_entries,
                    oracle_prompt=oracle_prompt,
                    judge_instruction_template=judge_instruction_template,
                    judge_instruction_file=judge_instruction_file,
                    judge_instruction_stem=judge_instruction_stem,
                    device=ctx.device,
                    target_model_name=model.config._name_or_path,
                    target_lora_path="default",
                    oracle_model_name=model.config._name_or_path,
                    oracle_lora_path="oracle",
                    oracle_generation_kwargs=oracle_generation_kwargs_deterministic,
                    judge_batch_size=oracle_judge_batch_size,
                    judge_lora_path="default",
                    cache_root="cache",
                    dist_ctx=ctx,
                    perf=perf,
                )
            )
        elapsed_s = perf_counter() - t0
        combinations_processed += 1

        if ctx.is_main:
            with (
                perf.track(
                    "stage/reporting_oracle",
                    {
                        "rank": ctx.rank,
                        "target_prompt_index": target_prompt_index,
                        "target_prompt_key": target_key,
                        "oracle_prompt_index": oracle_prompt_index,
                        "oracle_prompt_key": oracle_key,
                    },
                )
                if perf
                else nullcontext()
            ):
                oracle_report_path = save_oracle_rollouts_html(
                    oracle_results=judged_oracle_entries,
                    oracle_prompt=oracle_prompt,
                    tokenizer=tokenizer,
                    output_path=f"oracle_rollouts_report_{target_key}_{oracle_key}.html",
                )
                print(
                    f"[target {target_prompt_index} oracle {oracle_prompt_index}] "
                    f"Saved oracle report: {oracle_report_path}"
                )
                print(
                    f"[target {target_prompt_index} oracle {oracle_prompt_index}] "
                    f"Oracle rollouts cache: {oracle_cache_file}"
                )
                print(
                    f"[target {target_prompt_index} oracle {oracle_prompt_index}] "
                    f"Oracle judged cache: {oracle_judge_cache_file}"
                )
                log_oracle_metrics(
                    wandb_run,
                    judged_oracle_entries if isinstance(judged_oracle_entries, list) else [judged_oracle_entries],
                    elapsed_s,
                )
                log_oracle_judge_metrics(wandb_run, oracle_cache_stats)
                log_oracle_judge_metrics(wandb_run, oracle_judge_summary)
                log_oracle_judge_metrics(
                    wandb_run,
                    {
                        "pipeline/target_prompt_index": float(target_prompt_index),
                        "pipeline/oracle_prompt_index": float(oracle_prompt_index),
                    },
                )
    return combinations_processed


if MAIN:
    ctx = init_distributed()
    wandb_run = None
    perf: PerfLogger | None = None
    try:
        num_rollouts = _env_int("NUM_ROLLOUTS", 50)
        max_new_tokens = _env_int("MAX_NEW_TOKENS", 10000)
        oracle_max_new_tokens = _env_int("ORACLE_MAX_NEW_TOKENS", 1000)
        eval_batch_size = _env_int("ORACLE_EVAL_BATCH_SIZE", 32)
        oracle_judge_batch_size = _env_int("ORACLE_JUDGE_BATCH_SIZE", 8)
        target_prompt_limit = _env_int("TARGET_PROMPT_LIMIT", 100)
        oracle_prompts_path = os.getenv("ORACLE_PROMPTS_PATH", str(DEFAULT_ORACLE_PROMPTS_PATH)).strip()

        if ctx.is_main:
            target_prompts = load_target_prompts_from_dataset(limit=target_prompt_limit)
            oracle_prompts = load_oracle_prompts_from_file(oracle_prompts_path)
            if not target_prompts:
                raise ValueError("No target prompts were loaded from LLM-LAT/harmful-dataset.")
            if not oracle_prompts:
                raise ValueError("No oracle prompts were loaded from ORACLE_PROMPTS_PATH.")
        else:
            target_prompts = None
            oracle_prompts = None

        if ctx.enabled:
            target_prompts = broadcast_object(ctx, target_prompts, src=0)
            oracle_prompts = broadcast_object(ctx, oracle_prompts, src=0)

        if target_prompts is None:
            target_prompts = []
        if oracle_prompts is None:
            oracle_prompts = []
        if not target_prompts:
            raise ValueError("Target prompt list is empty.")
        if not oracle_prompts:
            raise ValueError("Oracle prompt list is empty.")

        if ctx.is_main:
            wandb_run = init_wandb_run({
                "model_name": QWEN_MODEL_NAME,
                "oracle_lora_path": QWEN_ORACLE_LORA_PATH,
                "num_rollouts": num_rollouts,
                "dtype": str(dtype),
                "device": str(ctx.device),
                "world_size": ctx.world_size,
                "rank_zero_device": str(ctx.device),
                "distributed_enabled": ctx.enabled,
                "max_new_tokens": max_new_tokens,
                "oracle_max_new_tokens": oracle_max_new_tokens,
                "oracle_eval_batch_size": eval_batch_size,
                "oracle_judge_batch_size": oracle_judge_batch_size,
                "judge_instruction_file": JUDGE_INSTRUCTION_PATH,
                "target_prompt_limit": target_prompt_limit,
                "target_prompts_total": len(target_prompts),
                "oracle_prompts_path": oracle_prompts_path,
                "oracle_prompts_total": len(oracle_prompts),
                "oracle_pipeline_mode": "deterministic",
            })
        perf = build_perf_logger(wandb_run=wandb_run, dist_ctx=ctx)

        device_map: str | dict[str, str] = "auto"
        if ctx.enabled and ctx.device.type == "cuda":
            device_map = {"": f"cuda:{ctx.local_rank}"}

        with perf.track("load/model_stack", {"rank": ctx.rank, "world_size": ctx.world_size}) if perf else nullcontext():
            tokenizer, model = load_model_stack(
                model_name=QWEN_MODEL_NAME,
                adapter_specs=[AdapterSpec(adapter_path=QWEN_ORACLE_LORA_PATH, adapter_name="oracle")],
                torch_dtype=dtype,
                device_map=device_map,
                hf_token=HF_TOKEN,
            )

        generation_kwargs_stochastic = {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": max_new_tokens,
        }
        oracle_generation_kwargs_deterministic = {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": oracle_max_new_tokens,
        }

        judge_instruction_file, judge_instruction_stem, judge_instruction_template = load_judge_instruction(
            JUDGE_INSTRUCTION_PATH
        )
        combinations_processed = 0
        combinations_total = len(target_prompts) * len(oracle_prompts)
        for target_prompt_index, target_prompt_str in enumerate(target_prompts):
            combinations_processed += run_pipeline_for_target_prompt(
                model=model,
                tokenizer=tokenizer,
                ctx=ctx,
                wandb_run=wandb_run,
                perf=perf,
                target_prompt_str=target_prompt_str,
                target_prompt_index=target_prompt_index,
                oracle_prompts=oracle_prompts,
                num_rollouts=num_rollouts,
                eval_batch_size=eval_batch_size,
                oracle_judge_batch_size=oracle_judge_batch_size,
                generation_kwargs_stochastic=generation_kwargs_stochastic,
                oracle_generation_kwargs_deterministic=oracle_generation_kwargs_deterministic,
                judge_instruction_file=judge_instruction_file,
                judge_instruction_stem=judge_instruction_stem,
                judge_instruction_template=judge_instruction_template,
            )

        if ctx.is_main:
            log_oracle_judge_metrics(
                wandb_run,
                {
                    "pipeline/target_prompts_total": float(len(target_prompts)),
                    "pipeline/oracle_prompts_total": float(len(oracle_prompts)),
                    "pipeline/combinations_total": float(combinations_total),
                    "pipeline/combinations_processed": float(combinations_processed),
                },
            )
            if perf:
                perf.flush_summary()
            if wandb_run is not None:
                wandb_run.finish()
    finally:
        cleanup_distributed(ctx)
