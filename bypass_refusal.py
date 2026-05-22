import os
from contextlib import nullcontext
from time import perf_counter
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from model_loading_utils import AdapterSpec, load_model_stack
from distributed_utils import DistributedContext, broadcast_object, cleanup_distributed, init_distributed
from judge_instruction_utils import load_judge_instruction
from oracle_judge_utils import judge_oracle_rollouts
from oracle_rollout_utils import (
    DEFAULT_ORACLE_ROLLOUT_MODE,
    OracleRolloutMode,
    generate_oracle_rollouts_for_mode,
    oracle_rollouts_dir_base_for_mode,
    parse_oracle_rollout_mode,
)
from perf_utils import PerfLogger, build_perf_logger
from prompt_utils import load_oracle_prompts_from_file, load_target_prompts_from_dataset, prompt_key
from rollout_utils import (
    display_rollout_results,
    format_user_target_prompt,
    generate_target_rollouts,
    judge_target_rollouts,
)
from wandb_utils import init_wandb_run, log_oracle_judge_metrics, log_oracle_metrics, log_rollout_metrics

dtype = torch.bfloat16
hf_logging.set_verbosity_error()

EXTENSION_ROOT = Path(__file__).resolve().parent


@dataclass
class ExperimentConfig:
    model_name: str
    oracle_adapter_path: str
    oracle_adapter_name: str
    oracle_prompts_path: str
    judge_instruction_path: str
    num_rollouts: int
    k_rollouts: int | None
    k_rollouts_raw: int
    num_oracle_rollouts: int
    oracle_rollout_mode: OracleRolloutMode
    max_new_tokens: int
    oracle_max_new_tokens: int
    oracle_eval_batch_size: int
    oracle_judge_batch_size: int
    target_judge_batch_size: int
    target_prompt_limit: int
    run_target_rollouts: bool
    run_target_judging: bool
    run_oracle_rollouts: bool
    run_oracle_judging: bool
    target_lora_path: str
    judge_lora_path: str
    oracle_lora_path: str
    judge_thinking_mode: str
    experiment_preset: str

    @staticmethod
    def resolve_relative_to_extension(raw_path: str) -> str:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return str(candidate)
        return str((EXTENSION_ROOT / candidate).resolve())

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        oracle_rollout_mode = parse_oracle_rollout_mode(
            _env_str("ORACLE_ROLLOUT_MODE", DEFAULT_ORACLE_ROLLOUT_MODE)
        )
        preset = _env_str("EXPERIMENT_PRESET", "")
        model_name = _env_str("MODEL_NAME", "Qwen/Qwen3-8B")
        oracle_adapter_path = _env_str("ORACLE_ADAPTER_PATH", "")
        oracle_adapter_name = _env_str("ORACLE_ADAPTER_NAME", "oracle")
        if not oracle_adapter_path:
            raise ValueError("ORACLE_ADAPTER_PATH must be set to a valid LoRA checkpoint path.")
        if not oracle_adapter_name:
            raise ValueError("ORACLE_ADAPTER_NAME must be set to a non-empty adapter name.")

        run_target_rollouts = _parse_bool(_env_str("RUN_TARGET_ROLLOUTS", "true"), field_name="RUN_TARGET_ROLLOUTS")
        run_target_judging = _parse_bool(_env_str("RUN_TARGET_JUDGING", "true"), field_name="RUN_TARGET_JUDGING")
        run_oracle_rollouts = _parse_bool(_env_str("RUN_ORACLE_ROLLOUTS", "true"), field_name="RUN_ORACLE_ROLLOUTS")
        run_oracle_judging = _parse_bool(_env_str("RUN_ORACLE_JUDGING", "true"), field_name="RUN_ORACLE_JUDGING")
        target_lora_path = _env_str("TARGET_LORA_PATH", "default")
        judge_lora_path = _env_str("JUDGE_LORA_PATH", "default")
        oracle_lora_path = _env_str("ORACLE_LORA_PATH", oracle_adapter_name)
        judge_thinking_mode = _env_str("JUDGE_THINKING", "off")

        if not any((run_target_rollouts, run_target_judging, run_oracle_rollouts, run_oracle_judging)):
            raise ValueError("At least one pipeline stage must be enabled.")
        if run_target_judging and not run_target_rollouts:
            raise ValueError("RUN_TARGET_JUDGING=true requires RUN_TARGET_ROLLOUTS=true.")
        if run_oracle_judging and not run_oracle_rollouts:
            raise ValueError("RUN_ORACLE_JUDGING=true requires RUN_ORACLE_ROLLOUTS=true.")
        if run_oracle_rollouts and not run_target_rollouts and oracle_rollout_mode != "prompt_only_repeats":
            raise ValueError(
                "RUN_ORACLE_ROLLOUTS=true with RUN_TARGET_ROLLOUTS=false is only supported for "
                "ORACLE_ROLLOUT_MODE=prompt_only_repeats."
            )
        if judge_thinking_mode not in {"default", "off"}:
            raise ValueError(
                f"Invalid JUDGE_THINKING={judge_thinking_mode!r}. Expected one of: default, off."
            )

        num_rollouts = _env_int("NUM_ROLLOUTS", 50)
        k_rollouts_raw = _env_int("K_ROLLOUTS", num_rollouts)
        k_rollouts = k_rollouts_raw if k_rollouts_raw > 0 else None
        num_oracle_rollouts = _env_int("NUM_ORACLE_ROLLOUTS", 1)
        oracle_prompts_raw = _env_str("ORACLE_PROMPTS_PATH", "prompts/oracle_prompts/default_oracle_prompts.json")
        judge_instruction_raw = _env_str("JUDGE_INSTRUCTION_PATH", "user_request_fulfillment.jinja2")

        return cls(
            model_name=model_name,
            oracle_adapter_path=oracle_adapter_path,
            oracle_adapter_name=oracle_adapter_name,
            oracle_prompts_path=cls.resolve_relative_to_extension(oracle_prompts_raw),
            judge_instruction_path=judge_instruction_raw,
            num_rollouts=num_rollouts,
            k_rollouts=k_rollouts,
            k_rollouts_raw=k_rollouts_raw,
            num_oracle_rollouts=num_oracle_rollouts,
            oracle_rollout_mode=oracle_rollout_mode,
            max_new_tokens=_env_int("MAX_NEW_TOKENS", 10000),
            oracle_max_new_tokens=_env_int("ORACLE_MAX_NEW_TOKENS", 1000),
            oracle_eval_batch_size=_env_int("ORACLE_EVAL_BATCH_SIZE", 32),
            oracle_judge_batch_size=_env_int("ORACLE_JUDGE_BATCH_SIZE", 8),
            target_judge_batch_size=_env_int("TARGET_JUDGE_BATCH_SIZE", 16),
            target_prompt_limit=_env_int("TARGET_PROMPT_LIMIT", 100),
            run_target_rollouts=run_target_rollouts,
            run_target_judging=run_target_judging,
            run_oracle_rollouts=run_oracle_rollouts,
            run_oracle_judging=run_oracle_judging,
            target_lora_path=target_lora_path,
            judge_lora_path=judge_lora_path,
            oracle_lora_path=oracle_lora_path,
            judge_thinking_mode=judge_thinking_mode,
            experiment_preset=preset,
        )

    def target_generation_kwargs(self) -> dict[str, Any]:
        return {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": self.max_new_tokens,
        }

    def oracle_generation_kwargs_deterministic(self) -> dict[str, Any]:
        return {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": self.oracle_max_new_tokens,
        }

    def oracle_generation_kwargs_sampled(self) -> dict[str, Any]:
        return {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": self.oracle_max_new_tokens,
        }

    def oracle_generation_kwargs_prompt_only(self) -> dict[str, Any]:
        return {
            "do_sample": True,
            "temperature": 1.0,
            "max_new_tokens": self.oracle_max_new_tokens,
        }

    def oracle_judge_generation_kwargs(self) -> dict[str, Any]:
        if self.oracle_rollout_mode == "prompt_only_repeats":
            return self.oracle_generation_kwargs_prompt_only()
        if self.oracle_rollout_mode == "sampled_target_repeats":
            return self.oracle_generation_kwargs_sampled()
        return self.oracle_generation_kwargs_deterministic()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _parse_bool(raw: str, *, field_name: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value for {field_name}: {raw!r}. "
        "Use one of: true/false, yes/no, on/off, 1/0."
    )


def run_pipeline_for_target_prompt(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    ctx: DistributedContext,
    wandb_run: Any | None,
    perf: PerfLogger | None,
    cfg: ExperimentConfig,
    target_prompt_str: str,
    target_prompt_index: int,
    oracle_prompts: list[str],
    judge_instruction_file: str,
    judge_instruction_stem: str,
    judge_instruction_template: str,
) -> int:
    target_key = prompt_key(target_prompt_str)
    formatted_target_prompt = format_user_target_prompt(tokenizer, target_prompt_str)
    target_rollout_entries: list[dict[str, Any]] = []
    judged_rollout_entries: list[dict[str, Any]] = []
    target_cache_file: str | Path | None = None
    judge_cache_file: str | Path | None = None
    compliance_results: dict[str, Any] = {"compliance_rate": 0.0, "partial_compliance_rate": 0.0, "total": 0}

    if cfg.run_target_rollouts:
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
                generation_kwargs_stochastic=cfg.target_generation_kwargs(),
                num_rollouts=cfg.num_rollouts,
                device=ctx.device,
                target_lora_path=cfg.target_lora_path,
                cache_root="cache",
                dist_ctx=ctx,
                perf=perf,
            )

    if cfg.run_target_judging:
        if not target_rollout_entries:
            if ctx.is_main:
                print(f"[target {target_prompt_index}] Skipping target judging: no target rollouts available.")
        else:
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
                    target_lora_path=cfg.target_lora_path,
                    judge_lora_path=cfg.judge_lora_path,
                    judge_thinking_mode=cfg.judge_thinking_mode,
                    target_judge_batch_size=cfg.target_judge_batch_size,
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
                    if target_cache_file is not None:
                        print(f"[target {target_prompt_index}] Target rollouts cache: {target_cache_file}")
                    if judge_cache_file is not None:
                        print(f"[target {target_prompt_index}] Judge rollouts cache: {judge_cache_file}")

    oracle_source_entries = judged_rollout_entries if cfg.run_target_judging else target_rollout_entries
    if not cfg.run_oracle_rollouts:
        return 0
    if not oracle_source_entries and cfg.oracle_rollout_mode != "prompt_only_repeats":
        if ctx.is_main:
            print(f"[target {target_prompt_index}] Skipping oracle stages: no target entries available.")
        return 0

    combinations_processed = 0
    for oracle_prompt_index, oracle_prompt in enumerate(oracle_prompts):
        oracle_key = prompt_key(oracle_prompt)
        oracle_input_source = "formatted_prompt_only" if cfg.oracle_rollout_mode == "prompt_only_repeats" else "target_rollouts"
        if ctx.is_main and cfg.oracle_rollout_mode == "prompt_only_repeats" and oracle_source_entries:
            print(
                "[oracle prompt-only] target rollout entries are present but will be ignored "
                f"(count={len(oracle_source_entries)} source={oracle_input_source})"
            )
        t0 = perf_counter()
        if ctx.is_main:
            print(
                f"[stage oracle rollout] mode={cfg.oracle_rollout_mode} input_source={oracle_input_source} "
                f"target_prompt_index={target_prompt_index} oracle_prompt_index={oracle_prompt_index} "
                f"target_key={target_key} oracle_key={oracle_key}"
            )
        with (
            perf.track(
                "stage/oracle_rollout_total",
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
            oracle_rollout_entries, oracle_cache_file, oracle_cache_stats = generate_oracle_rollouts_for_mode(
                mode=cfg.oracle_rollout_mode,
                model=model,
                tokenizer=tokenizer,
                device=ctx.device,
                oracle_prompt=oracle_prompt,
                target_prompt=target_prompt_str,
                target_rollout_entries=oracle_source_entries,
                target_model_name=model.config._name_or_path,
                target_lora_path=cfg.target_lora_path,
                oracle_lora_path=cfg.oracle_lora_path,
                cache_root="cache",
                k_rollouts=cfg.k_rollouts,
                num_oracle_rollouts=cfg.num_oracle_rollouts,
                oracle_generation_kwargs_deterministic=cfg.oracle_generation_kwargs_deterministic(),
                oracle_generation_kwargs_sampled=cfg.oracle_generation_kwargs_sampled(),
                oracle_generation_kwargs_prompt_only=cfg.oracle_generation_kwargs_prompt_only(),
                eval_batch_size=cfg.oracle_eval_batch_size,
                dist_ctx=ctx,
                perf=perf,
            )
        judged_oracle_entries: list[dict[str, Any]] = []
        oracle_judge_cache_file: str | Path | None = None
        oracle_judge_summary: dict[str, Any] = {}
        if cfg.run_oracle_judging:
            if ctx.is_main:
                print(
                    f"[stage oracle judging] mode={cfg.oracle_rollout_mode} input_source={oracle_input_source} "
                    f"target_prompt_index={target_prompt_index} oracle_prompt_index={oracle_prompt_index} "
                    f"target_key={target_key} oracle_key={oracle_key}"
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
                    judge_oracle_rollouts(
                        judge_model=model,
                        judge_tokenizer=tokenizer,
                        oracle_rollout_entries=oracle_rollout_entries,
                        oracle_prompt=oracle_prompt,
                        judge_instruction_template=judge_instruction_template,
                        judge_instruction_file=judge_instruction_file,
                        judge_instruction_stem=judge_instruction_stem,
                        device=ctx.device,
                        target_model_name=model.config._name_or_path,
                        target_lora_path=cfg.target_lora_path,
                        oracle_model_name=model.config._name_or_path,
                        oracle_lora_path=cfg.oracle_lora_path,
                        oracle_generation_kwargs=cfg.oracle_judge_generation_kwargs(),
                        oracle_rollouts_dir_base=oracle_rollouts_dir_base_for_mode(cfg.oracle_rollout_mode),
                        judge_batch_size=cfg.oracle_judge_batch_size,
                        judge_lora_path=cfg.judge_lora_path,
                        judge_thinking_mode=cfg.judge_thinking_mode,
                        cache_root="cache",
                        dist_ctx=ctx,
                        perf=perf,
                    )
                )
        elapsed_s = perf_counter() - t0
        combinations_processed += 1

        if ctx.is_main:
            report_entries = judged_oracle_entries if cfg.run_oracle_judging else oracle_rollout_entries
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
                print(
                    f"[oracle reporting] target_prompt_index={target_prompt_index} oracle_prompt_index={oracle_prompt_index} "
                    f"Oracle rollouts cache: {oracle_cache_file}"
                )
                log_oracle_metrics(
                    wandb_run,
                    report_entries if isinstance(report_entries, list) else [report_entries],
                    elapsed_s,
                )
                log_oracle_judge_metrics(wandb_run, oracle_cache_stats)
                if cfg.run_oracle_judging:
                    log_oracle_judge_metrics(wandb_run, oracle_judge_summary)
                log_oracle_judge_metrics(
                    wandb_run,
                    {
                        "pipeline/target_prompt_index": float(target_prompt_index),
                        "pipeline/oracle_prompt_index": float(oracle_prompt_index),
                    },
                )
                if oracle_judge_cache_file is not None:
                    print(
                        f"[oracle reporting] target_prompt_index={target_prompt_index} oracle_prompt_index={oracle_prompt_index} "
                        f"Oracle judged cache: {oracle_judge_cache_file}"
                    )
    return combinations_processed


def main(cfg: ExperimentConfig) -> None:
    ctx = init_distributed()
    wandb_run = None
    perf: PerfLogger | None = None
    try:
        if ctx.is_main:
            should_load_target_prompts = any(
                (
                    cfg.run_target_rollouts,
                    cfg.run_target_judging,
                    cfg.run_oracle_rollouts,
                )
            )
            target_prompts = (
                load_target_prompts_from_dataset(limit=cfg.target_prompt_limit)
                if should_load_target_prompts
                else []
            )
            oracle_prompts = (
                load_oracle_prompts_from_file(cfg.oracle_prompts_path)
                if cfg.run_oracle_rollouts
                else []
            )
            if should_load_target_prompts and not target_prompts:
                raise ValueError("No target prompts were loaded from LLM-LAT/harmful-dataset.")
            if cfg.run_oracle_rollouts and not oracle_prompts:
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
        if any((cfg.run_target_rollouts, cfg.run_target_judging, cfg.run_oracle_rollouts)):
            if not target_prompts:
                raise ValueError("Target prompt list is empty.")
        if cfg.run_oracle_rollouts and not oracle_prompts:
            raise ValueError("Oracle prompt list is empty.")

        if ctx.is_main:
            wandb_run = init_wandb_run({
                "model_name": cfg.model_name,
                "oracle_adapter_path": cfg.oracle_adapter_path,
                "oracle_adapter_name": cfg.oracle_adapter_name,
                "num_rollouts": cfg.num_rollouts,
                "k_rollouts": cfg.k_rollouts_raw,
                "num_oracle_rollouts": cfg.num_oracle_rollouts,
                "oracle_rollout_mode": cfg.oracle_rollout_mode,
                "dtype": str(dtype),
                "device": str(ctx.device),
                "world_size": ctx.world_size,
                "rank_zero_device": str(ctx.device),
                "distributed_enabled": ctx.enabled,
                "max_new_tokens": cfg.max_new_tokens,
                "oracle_max_new_tokens": cfg.oracle_max_new_tokens,
                "oracle_eval_batch_size": cfg.oracle_eval_batch_size,
                "oracle_judge_batch_size": cfg.oracle_judge_batch_size,
                "target_judge_batch_size": cfg.target_judge_batch_size,
                "judge_thinking_mode": cfg.judge_thinking_mode,
                "judge_instruction_file": cfg.judge_instruction_path,
                "target_prompt_limit": cfg.target_prompt_limit,
                "target_prompts_total": len(target_prompts),
                "oracle_prompts_path": cfg.oracle_prompts_path,
                "oracle_prompts_total": len(oracle_prompts),
                "experiment_preset": cfg.experiment_preset,
                "run_target_rollouts": float(cfg.run_target_rollouts),
                "run_target_judging": float(cfg.run_target_judging),
                "run_oracle_rollouts": float(cfg.run_oracle_rollouts),
                "run_oracle_judging": float(cfg.run_oracle_judging),
                "target_lora_path": cfg.target_lora_path,
                "judge_lora_path": cfg.judge_lora_path,
                "oracle_runtime_lora_path": cfg.oracle_lora_path,
            })
        perf = build_perf_logger(wandb_run=wandb_run, dist_ctx=ctx)

        device_map: str | dict[str, str] = "auto"
        if ctx.enabled and ctx.device.type == "cuda":
            device_map = {"": f"cuda:{ctx.local_rank}"}

        with perf.track("load/model_stack", {"rank": ctx.rank, "world_size": ctx.world_size}) if perf else nullcontext():
            tokenizer, model = load_model_stack(
                model_name=cfg.model_name,
                adapter_specs=[AdapterSpec(adapter_path=cfg.oracle_adapter_path, adapter_name=cfg.oracle_adapter_name)],
                torch_dtype=dtype,
                device_map=device_map,
                hf_token=_require_hf_token(),
            )

        if cfg.run_target_judging or cfg.run_oracle_judging:
            judge_instruction_file, judge_instruction_stem, judge_instruction_template = load_judge_instruction(
                cfg.judge_instruction_path
            )
        else:
            judge_instruction_file = cfg.judge_instruction_path
            judge_instruction_stem = Path(cfg.judge_instruction_path).stem
            judge_instruction_template = ""
        combinations_processed = 0
        combinations_total = len(target_prompts) * len(oracle_prompts)
        for target_prompt_index, target_prompt_str in enumerate(target_prompts):
            combinations_processed += run_pipeline_for_target_prompt(
                model=model,
                tokenizer=tokenizer,
                ctx=ctx,
                wandb_run=wandb_run,
                perf=perf,
                cfg=cfg,
                target_prompt_str=target_prompt_str,
                target_prompt_index=target_prompt_index,
                oracle_prompts=oracle_prompts,
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


def _require_hf_token() -> str:
    # Load .env from workspace root (parent of this repo): /workspace/.env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=str(env_path))
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Please set HF_TOKEN in your <parent_dir>/.env file")
    return token


if __name__ == "__main__":
    main(ExperimentConfig.from_env())
