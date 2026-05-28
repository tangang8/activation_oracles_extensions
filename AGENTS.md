# AGENTS.md

This file provides guidance to agents (Claude Code, Cursor, etc.) when working in this repository. `CLAUDE.md` is a symlink to this file (`ln -s AGENTS.md CLAUDE.md`).

## Coding Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. These bias toward caution over speed; for trivial tasks, use judgment.

### Think Before Coding

Do not assume. Do not hide confusion. Surface tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — do not pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that was not requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes

Touch only what you must. Clean up only your own mess.

When editing existing code:

- Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it — do not delete it.

When your changes create orphans:

- Remove imports, variables, and functions that your changes made unused.
- Do not remove pre-existing dead code unless asked.
- If you make a change, make sure you understand how it impacts other code and update all dependencies to be compatible. 

Every changed line should trace directly to the user's request.

### Goal-Driven Execution

Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

- "Add validation" → write tests for invalid inputs, then make them pass
- "Fix the bug" → write a test that reproduces it, then make it pass
- "Refactor X" → ensure tests pass before and after

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria allow independent verification. Weak criteria ("make it work") require constant clarification.

**These guidelines are working if:** diffs have fewer unnecessary changes, fewer rewrites from overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Overview

**activation_oracles_bypass_refusal** extends the upstream [activation_oracles](https://github.com/adamkarvonen/activation_oracles) repository with experimental utilities for oracle rollout modes and evaluation pipelines. The project runs multi-stage experiments that generate target model responses, judge them with classification models, and optionally run oracle-guided rollouts using activation probes.

**Key capability**: Oracle rollout modes allow generating responses based on different oracle input strategies (deterministic, sampled with repeats, or prompt-only), with flexible stage gates to skip target generation, target judging, oracle generation, or oracle judging as needed.

## Directory Structure & Sibling Dependency

This repo requires a specific parent-folder layout:

```
<parent-folder>/
├── activation_oracles_bypass_refusal/  (this repo)
└── activation_oracles/              (upstream repo)
```

Both repos should be cloned with exact names. The activation_oracles repo is imported via sys.path manipulation in `oracle_pipeline.py` to access the activation probing utilities.

## Core Modules

### Pipeline Orchestration
- **bypass_refusal.py**: Main entry point. Loads `ExperimentConfig` from environment variables, orchestrates the four-stage pipeline (target rollout → target judge → oracle rollout → oracle judge), and logs results to W&B.
- **run_oracle_experiment.sh**: Bash wrapper with editable defaults and CLI overrides for common presets. Handles model/adapter mappings and stage gate logic.
- **run_parallel_strongreject_v5.sh**: Multi-GPU scheduler for full StrongReject v5 sweeps (target shards, prompt-only oracle, oracle target control, rollout-post-prompt deterministic shards). Logs under `logs/parallel_<timestamp>/` (default `RUN_LABEL=parallel_<timestamp>`).
- **run_overnight_strongreject_v5.sh**: Simpler two-stage overnight driver (prompt-only oracle, then target judging only). Logs under `logs/overnight_<timestamp>/`.

### Oracle Rollout Modes
- **oracle_rollout_utils.py**: Implements three oracle rollout modes controlled by `ORACLE_ROLLOUT_MODE` env var:
  - `all_target_deterministic`: Uses all judged target rollouts; runs one oracle rollout per target at temperature=0.0
  - `sampled_target_repeats`: Samples K target rollouts; runs NUM_ORACLE_ROLLOUTS repeats at temperature=1.0
  - `prompt_only_repeats`: Ignores target responses; runs NUM_ORACLE_ROLLOUTS repeats at temperature=1.0

### Model & Adapter Loading
- **model_loading_utils.py**: Loads base models via transformers, applies LoRA adapters (from peft), handles dtype casting (bfloat16).
- **oracle_pipeline.py**: Collects activation probes from specified layers, evaluates oracle inputs using activation-based classifiers, returns predictions used to guide generation.

### Response Generation & Judging
- **rollout_utils.py**: Generates target responses, extracts thinking tags (model-specific), applies stopping criteria, caches results. Provides batch judgment via compliance scoring.
- **oracle_judge_utils.py**: Judges oracle-generated responses; flattens oracle output structure, applies compliance scoring, aggregates scores across rollouts.

### Prompt & Cache Management
- **prompt_utils.py**: Loads oracle prompts and target prompts from files (JSON, JSONL, txt); generates stable cache keys via preview+hash.
- **cache_utils.py**: Manages cache directory structure for target rollouts, oracle rollouts, and judge outputs; provides sanitized path construction.

### Utilities & Logging
- **oracle_token_points.py**: Extracts token-level activation points for probing; validates prompt/response tokenization boundaries.
- **judge_instruction_utils.py**: Loads Jinja2 judge instruction templates from `prompts/judge_classification_instructions/`.
- **perf_utils.py**: Performance logging with context managers.
- **wandb_utils.py**: W&B integration for metrics logging across pipeline stages.
- **distributed_utils.py**: Distributed (multi-GPU) support via torch.distributed; broadcast/gather across ranks.

### Compilation & Reporting
- **compile_strongreject_results.py**: Aggregates judge outputs into structured JSON for later analysis.
- **report_pages.py** & **generate_reports.py**: HTML report generation from compiled results.

## Key Commands

### Run Tests
```bash
# From repo root, run all tests
PYTHONPATH=".:results" python -m unittest discover -v -s tests

# Run a specific test file
PYTHONPATH=".:results" python -m unittest tests.test_prompt_utils -v

# Run a specific test class or method
PYTHONPATH=".:results" python -m unittest tests.test_prompt_utils.PromptUtilsTests.test_prompt_key_hash_length -v
```

### Bash Script Syntax Check
```bash
bash -n run_oracle_experiment.sh
bash -n run_parallel_strongreject_v5.sh
bash -n run_overnight_strongreject_v5.sh
```

### Run Experiments

**Easy preset-based runs** (recommended):
```bash
./run_oracle_experiment.sh --preset full_deterministic_oracle
./run_oracle_experiment.sh --preset sampled_target_repeats --k-rollouts 5 --num-oracle-rollouts 2
./run_oracle_experiment.sh --preset prompt_only_oracle --num-oracle-rollouts 4
```

**Small validation run** (one prompt per mode):
```bash
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=3 NUM_ORACLE_ROLLOUTS=1 ORACLE_ROLLOUT_MODE=all_target_deterministic python bypass_refusal.py
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=5 K_ROLLOUTS=2 NUM_ORACLE_ROLLOUTS=2 ORACLE_ROLLOUT_MODE=sampled_target_repeats python bypass_refusal.py
TARGET_PROMPT_LIMIT=1 NUM_ROLLOUTS=3 NUM_ORACLE_ROLLOUTS=3 ORACLE_ROLLOUT_MODE=prompt_only_repeats python bypass_refusal.py
```

**Direct Python entry** (if not using the bash wrapper):
```bash
ORACLE_ROLLOUT_MODE=prompt_only_repeats NUM_ORACLE_ROLLOUTS=3 TARGET_PROMPT_LIMIT=1 python bypass_refusal.py
```

**Parallel StrongReject v5** (multi-GPU scheduler; default preset for deterministic shards is `rollout_post_prompt_oracle`):
```bash
./run_parallel_strongreject_v5.sh
# Optional: DRY_RUN=1 GPU_IDS=0,1 ./run_parallel_strongreject_v5.sh
# Logs: logs/parallel_<timestamp>/parallel_driver.log
# Override label: RUN_LABEL=my_run LOG_ROOT=logs/my_run ./run_parallel_strongreject_v5.sh
```

**Overnight StrongReject v5** (sequential prompt-only, then target judging):
```bash
./run_overnight_strongreject_v5.sh
```

Older runs may still have logs under `logs/parallel_h200_<timestamp>/` from before the script rename; new runs use `logs/parallel_<timestamp>/`.

## Environment & Dependencies

- **Python version**: 3.10+ (depends on upstream activation_oracles)
- **Shared venv**: Create a single `.venv` at the parent folder and apply upstream lock:
  ```bash
  cd <parent-folder>
  python3 -m venv .venv
  source .venv/bin/activate
  uv sync --project activation_oracles --active
  ```
- **HuggingFace login**: Required for model access:
  ```bash
  huggingface-cli login --token <your_token>
  ```
- **Runtime deps** (from activation_oracles): torch, transformers, peft, tqdm, dotenv, wandb, bitsandbytes (GPU-only)

## Configuration via Environment Variables

All experiment parameters are env-var driven (see `ExperimentConfig.from_env()` in bypass_refusal.py):

**Core settings**:
- `MODEL_NAME`: Base model identifier (default: "Qwen/Qwen3-8B")
- `ORACLE_ADAPTER_PATH`: LoRA checkpoint path (required)
- `ORACLE_ADAPTER_NAME`: Adapter name for set_adapter (required)
- `ORACLE_ROLLOUT_MODE`: One of {all_target_deterministic, sampled_target_repeats, prompt_only_repeats}

**Target rollout**:
- `NUM_ROLLOUTS`: Number of target responses to generate (default: 50)
- `MAX_NEW_TOKENS`: Generation cap for targets (default: 10000)
- `TARGET_THINKING`: "default" or "off" (controls thinking tag extraction)

**Oracle rollout**:
- `K_ROLLOUTS`: Max target rollouts to select for oracle stage (used by sampled_target_repeats)
- `NUM_ORACLE_ROLLOUTS`: Oracle repeats per target or prompt-only (default: 1)
- `ORACLE_MAX_NEW_TOKENS`: Generation cap for oracle (default: 1000)
- `ORACLE_INPUT_TYPES`: Comma-separated activation probe types (defaults per mode)
- `ORACLE_TOKEN_POINT_FILTER`: "all" or "post_prompt" (default: "all")

**Judging**:
- `JUDGE_INSTRUCTION_PATH`: Jinja2 template filename (default: "strongReject_v5.jinja2")
- `JUDGE_THINKING`: "default" or "off"
- `TARGET_JUDGE_BATCH_SIZE`: Batch size for target judging (default: 16)
- `ORACLE_JUDGE_BATCH_SIZE`: Batch size for oracle judging (default: 16)

**Stage gates**:
- `RUN_TARGET_ROLLOUTS`: true/false (default: true)
- `RUN_TARGET_JUDGING`: true/false (default: true; requires RUN_TARGET_ROLLOUTS)
- `RUN_ORACLE_ROLLOUTS`: true/false (default: true)
- `RUN_ORACLE_JUDGING`: true/false (default: true; requires RUN_ORACLE_ROLLOUTS)

**Logging & prompts**:
- `ORACLE_PROMPTS_PATH`: Path to oracle prompt file (default: "prompts/oracle_prompts/default_oracle_prompts.json")
- `TARGET_PROMPT_OFFSET` & `TARGET_PROMPT_LIMIT`: Dataset slice control
- `WANDB_RUN_NAME`: Optional run display name
- `WANDB_SETTING`: "on" or "off"

## Testing Patterns

Tests use Python's `unittest` framework and mock dependencies when necessary. Key patterns:

- Skip tests if dependencies unavailable: `@unittest.skipIf(condition, reason)`
- Use `patch()` to mock external calls (transformers, torch, etc.)
- Test isolation: temporary directories for file I/O, mocked models for pipeline tests
- Integration tests validate cache schema, stage output, and environment variable parsing

Test files in `tests/` correspond to modules:
- `test_bypass_refusal_pipeline.py`: ExperimentConfig parsing, pipeline stage orchestration
- `test_oracle_modes.py`: Oracle rollout mode selection and cache path generation
- `test_cache_utils.py`: Cache directory structure
- `test_prompt_utils.py`: Prompt loading and key generation
- `test_compile_strongreject_results.py`: Result aggregation
- `test_run_parallel_sh.py`: `run_parallel_strongreject_v5.sh` scheduler dry-run and failure handling
- `test_run_oracle_experiment_sh.py`: `run_oracle_experiment.sh` CLI and preset flags

## Data & Cache Structure

The pipeline generates output under cache directories with structure like:
```
target_rollouts_<model>_lora-<adapter>/
├── prompt_key/
│   ├── target_rollout_metadata.json
│   ├── target_rollout_output_*.json
│   └── judge_*.json
oracle_rollouts_<mode>_<model>_lora-<adapter>/
├── prompt_key/
│   ├── oracle_rollout_metadata.json
│   ├── oracle_rollout_output_*.json
│   └── judge_*.json
```

Cache keys are stable hashes (preview + SHA256) to enable reproducible runs.

## Design Notes

1. **Modular stage gates**: Each pipeline stage (target generation, target judging, oracle generation, oracle judging) can be independently enabled/disabled, enabling flexible experiment designs.

2. **Oracle input flexibility**: Three rollout modes support different oracle use cases—deterministic baselines, statistical repeats, and prompt-only oracle responses.

3. **Distributed support**: Rank-aware caching and broadcast/gather patterns allow scaling across multi-GPU setups.

4. **Deterministic outputs**: Stable cache keys and seed management (where applicable) enable reproducible experiment runs.

5. **Adapter flexibility**: LoRA adapters can be swapped at each stage (target, judge, oracle), enabling composition of multiple fine-tuned models.
