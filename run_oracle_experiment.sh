#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User-editable defaults
# -----------------------------
# Mode/knob behavior (important):
# - all_target_deterministic:
#     * uses NUM_ROLLOUTS for target rollout generation/judging
#     * ignores K_ROLLOUTS
#     * ignores NUM_ORACLE_ROLLOUTS (always 1 oracle rollout per target)
# - sampled_target_repeats:
#     * uses NUM_ROLLOUTS for target rollout generation/judging
#     * uses K_ROLLOUTS to select target rollouts for oracle stage
#     * uses NUM_ORACLE_ROLLOUTS repeats for each selected target rollout
# - prompt_only_repeats:
#     * currently still uses NUM_ROLLOUTS for target rollout generation/judging
#       because the shared pipeline always runs those stages first
#     * ignores K_ROLLOUTS
#     * uses NUM_ORACLE_ROLLOUTS prompt-only oracle repeats
#
# Stage switches:
# - RUN_TARGET_ROLLOUTS: generate target responses
# - RUN_TARGET_JUDGING: classify target responses
# - RUN_ORACLE_ROLLOUTS: generate oracle responses
# - RUN_ORACLE_JUDGING: classify oracle responses
#
# Common commands:
# - Full deterministic oracle pipeline:
#   ./run_oracle_experiment.sh --preset full_deterministic_oracle
# - Sampled target repeats:
#   ./run_oracle_experiment.sh --preset sampled_target_repeats --k-rollouts 5 --num-oracle-rollouts 2
# - Prompt-only oracle:
#   ./run_oracle_experiment.sh --preset prompt_only_oracle --num-oracle-rollouts 4
# - Oracle target control:
#   ./run_oracle_experiment.sh --preset oracle_target_control
# - Target judging only:
#   ./run_oracle_experiment.sh --preset target_judging_only

# Available base models + oracle adapters (mapping):
MODEL_ORACLE_ADAPTER_MAPPINGS=(
  "Qwen/Qwen3-8B|adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3-8B|oracle"
  "meta-llama/Llama-3.1-8B-Instruct|adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct|oracle"
)

lookup_oracle_adapter_for_model() {
  local model_name="$1"
  local mapping_entry=""
  local mapped_model=""
  local mapped_path=""
  local mapped_name=""

  for mapping_entry in "${MODEL_ORACLE_ADAPTER_MAPPINGS[@]}"; do
    IFS='|' read -r mapped_model mapped_path mapped_name <<< "$mapping_entry"
    if [[ "$mapped_model" == "$model_name" ]]; then
      printf '%s|%s\n' "$mapped_path" "$mapped_name"
      return 0
    fi
  done
  return 1
}

model_name="Qwen/Qwen3-8B"                   # Base model identifier passed to MODEL_NAME.
model_mapping="$(lookup_oracle_adapter_for_model "$model_name")"  # Lookup default oracle adapter for model_name.
oracle_mode="all_target_deterministic"       # ORACLE_ROLLOUT_MODE: all_target_deterministic | sampled_target_repeats | prompt_only_repeats
num_rollouts=50                              # NUM_ROLLOUTS: number of target rollouts to generate.
k_rollouts=10                                # K_ROLLOUTS: max target rollouts selected for sampled oracle mode.
num_oracle_rollouts=1                        # NUM_ORACLE_ROLLOUTS: oracle repeats per selected target (or prompt-only repeats).
target_prompt_limit=1                        # TARGET_PROMPT_LIMIT: how many target prompts to load.
max_new_tokens=10000                         # MAX_NEW_TOKENS: generation cap for target rollout stage.
oracle_max_new_tokens=1000                   # ORACLE_MAX_NEW_TOKENS: generation cap for oracle rollout stage.
oracle_eval_batch_size=32                    # ORACLE_EVAL_BATCH_SIZE: batch size for oracle rollout generation.
oracle_judge_batch_size=8                    # ORACLE_JUDGE_BATCH_SIZE: batch size for oracle judging stage.
judge_thinking="off"                         # JUDGE_THINKING: default | off
judge_instruction_path="strongReject.jinja2"  # JUDGE_INSTRUCTION_PATH: judge prompt template file.
oracle_prompts_path="prompts/oracle_prompts/default_oracle_prompts.json"  # ORACLE_PROMPTS_PATH: oracle prompt list file.
wandb_run_name=""                            # WANDB_RUN_NAME: optional run display name.
wandb="off"                                  # WANDB_SETTING: on | off
oracle_adapter_path="${model_mapping%%|*}"   # ORACLE_ADAPTER_PATH: adapter checkpoint/path to load.
oracle_adapter_name="${model_mapping##*|}"   # ORACLE_ADAPTER_NAME: adapter name used with set_adapter.

# Experiment config defaults:
# - EXPERIMENT_PRESET can set a bundle of stage/adapter defaults
# - RUN_* flags gate pipeline stages (target rollout/judge, oracle rollout/judge)
# - ORACLE_ADAPTER_PATH / ORACLE_ADAPTER_NAME configure which LoRA gets loaded
# - *_LORA_PATH defaults set adapter names selected at each stage (must match loaded adapter names)
# - You can edit these defaults here or override with CLI flags below
experiment_preset=""
run_target_rollouts="true"
run_target_judging="true"
run_oracle_rollouts="true"
run_oracle_judging="true"
target_lora_path="default"
judge_lora_path="default"
oracle_lora_path="$oracle_adapter_name"

usage() {
  cat <<'EOF'
Run activation oracle experiment with readable defaults + CLI overrides.

Usage:
  ./run_oracle_experiment.sh [options]

Options:
  --mode MODE                    Oracle rollout mode:
                                 all_target_deterministic | sampled_target_repeats | prompt_only_repeats
  --model-name NAME              MODEL_NAME (base model identifier)
  --num-rollouts N               NUM_ROLLOUTS
  --k-rollouts N                 K_ROLLOUTS
  --num-oracle-rollouts N        NUM_ORACLE_ROLLOUTS
  --target-prompt-limit N        TARGET_PROMPT_LIMIT
  --max-new-tokens N             MAX_NEW_TOKENS
  --oracle-max-new-tokens N      ORACLE_MAX_NEW_TOKENS
  --oracle-eval-batch-size N     ORACLE_EVAL_BATCH_SIZE
  --oracle-judge-batch-size N    ORACLE_JUDGE_BATCH_SIZE
  --judge-instruction-path PATH  JUDGE_INSTRUCTION_PATH
  --oracle-prompts-path PATH     ORACLE_PROMPTS_PATH
  --oracle-adapter-path PATH     ORACLE_ADAPTER_PATH (LoRA checkpoint/path to load)
  --oracle-adapter-name NAME     ORACLE_ADAPTER_NAME (adapter name used with set_adapter)
  --preset NAME                  EXPERIMENT_PRESET (see preset section below)
  --run-target-rollouts true|false
  --run-target-judging true|false
  --run-oracle-rollouts true|false
  --run-oracle-judging true|false
  --target-lora-path NAME        TARGET_LORA_PATH (adapter name for target stages)
  --judge-lora-path NAME         JUDGE_LORA_PATH (adapter name for judge stages)
  --oracle-lora-path NAME        ORACLE_LORA_PATH (adapter name for oracle rollout stage)
  --judge-thinking MODE          JUDGE_THINKING: default | off
  --wandb on|off                 Enable/disable Weights & Biases logging
  --wandb-run-name NAME          WANDB_RUN_NAME
  --set KEY=VALUE                Additional env var (repeatable)
  -h, --help                     Show this help

Mode-specific knob usage:
  all_target_deterministic: uses --num-rollouts; ignores --k-rollouts and --num-oracle-rollouts
  sampled_target_repeats:   uses --num-rollouts, --k-rollouts, --num-oracle-rollouts
  prompt_only_repeats:      uses --num-rollouts and --num-oracle-rollouts; ignores --k-rollouts
  NOTE: prompt_only_repeats still runs target rollout/judge stages in current pipeline.

Preset behavior:
  full_deterministic_oracle:
    mode=all_target_deterministic, all 4 stages enabled
  sampled_target_repeats:
    mode=sampled_target_repeats, all 4 stages enabled
  prompt_only_oracle:
    mode=prompt_only_repeats, target stages off, oracle rollout on, oracle judging on
  oracle_target_control:
    target stages on, oracle stages off, target adapter=oracle adapter name
    oracle rollout mode is unused because oracle stages are off
  target_judging_only:
    target stages on, oracle stages off
    oracle rollout mode is unused because oracle stages are off

Preset examples:
  ./run_oracle_experiment.sh --preset full_deterministic_oracle
  ./run_oracle_experiment.sh --preset sampled_target_repeats --k-rollouts 8 --num-oracle-rollouts 3
  ./run_oracle_experiment.sh --preset prompt_only_oracle --num-oracle-rollouts 4
  ./run_oracle_experiment.sh --preset oracle_target_control --num-rollouts 50 --target-prompt-limit 100
  ./run_oracle_experiment.sh --preset target_judging_only

Override via env vars is also supported, e.g.:
  ORACLE_ROLLOUT_MODE=prompt_only_repeats ./run_oracle_experiment.sh
EOF
}

MODE_FROM_ENV="${ORACLE_ROLLOUT_MODE+x}"
MODEL_NAME_FROM_ENV="${MODEL_NAME+x}"
NUM_ROLLOUTS_FROM_ENV="${NUM_ROLLOUTS+x}"
K_ROLLOUTS_FROM_ENV="${K_ROLLOUTS+x}"
NUM_ORACLE_ROLLOUTS_FROM_ENV="${NUM_ORACLE_ROLLOUTS+x}"
TARGET_PROMPT_LIMIT_FROM_ENV="${TARGET_PROMPT_LIMIT+x}"
MAX_NEW_TOKENS_FROM_ENV="${MAX_NEW_TOKENS+x}"
ORACLE_MAX_NEW_TOKENS_FROM_ENV="${ORACLE_MAX_NEW_TOKENS+x}"
ORACLE_EVAL_BATCH_SIZE_FROM_ENV="${ORACLE_EVAL_BATCH_SIZE+x}"
ORACLE_JUDGE_BATCH_SIZE_FROM_ENV="${ORACLE_JUDGE_BATCH_SIZE+x}"
JUDGE_THINKING_FROM_ENV="${JUDGE_THINKING+x}"
JUDGE_INSTRUCTION_PATH_FROM_ENV="${JUDGE_INSTRUCTION_PATH+x}"
ORACLE_PROMPTS_PATH_FROM_ENV="${ORACLE_PROMPTS_PATH+x}"
ORACLE_ADAPTER_PATH_FROM_ENV="${ORACLE_ADAPTER_PATH+x}"
ORACLE_ADAPTER_NAME_FROM_ENV="${ORACLE_ADAPTER_NAME+x}"
ORACLE_LORA_PATH_FROM_ENV="${ORACLE_LORA_PATH+x}"
EXPERIMENT_PRESET_FROM_ENV="${EXPERIMENT_PRESET+x}"
RUN_TARGET_ROLLOUTS_FROM_ENV="${RUN_TARGET_ROLLOUTS+x}"
RUN_TARGET_JUDGING_FROM_ENV="${RUN_TARGET_JUDGING+x}"
RUN_ORACLE_ROLLOUTS_FROM_ENV="${RUN_ORACLE_ROLLOUTS+x}"
RUN_ORACLE_JUDGING_FROM_ENV="${RUN_ORACLE_JUDGING+x}"
TARGET_LORA_PATH_FROM_ENV="${TARGET_LORA_PATH+x}"
JUDGE_LORA_PATH_FROM_ENV="${JUDGE_LORA_PATH+x}"

MODE="${ORACLE_ROLLOUT_MODE:-$oracle_mode}"
MODEL_NAME="${MODEL_NAME:-$model_name}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-$num_rollouts}"
K_ROLLOUTS="${K_ROLLOUTS:-$k_rollouts}"
NUM_ORACLE_ROLLOUTS="${NUM_ORACLE_ROLLOUTS:-$num_oracle_rollouts}"
TARGET_PROMPT_LIMIT="${TARGET_PROMPT_LIMIT:-$target_prompt_limit}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-$max_new_tokens}"
ORACLE_MAX_NEW_TOKENS="${ORACLE_MAX_NEW_TOKENS:-$oracle_max_new_tokens}"
ORACLE_EVAL_BATCH_SIZE="${ORACLE_EVAL_BATCH_SIZE:-$oracle_eval_batch_size}"
ORACLE_JUDGE_BATCH_SIZE="${ORACLE_JUDGE_BATCH_SIZE:-$oracle_judge_batch_size}"
JUDGE_THINKING="${JUDGE_THINKING:-$judge_thinking}"
JUDGE_INSTRUCTION_PATH="${JUDGE_INSTRUCTION_PATH:-$judge_instruction_path}"
ORACLE_PROMPTS_PATH="${ORACLE_PROMPTS_PATH:-$oracle_prompts_path}"
ORACLE_ADAPTER_PATH="${ORACLE_ADAPTER_PATH:-}"
ORACLE_ADAPTER_NAME="${ORACLE_ADAPTER_NAME:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$wandb_run_name}"
WANDB_SETTING="${WANDB_SETTING:-$wandb}"
EXPERIMENT_PRESET="${EXPERIMENT_PRESET:-$experiment_preset}"
RUN_TARGET_ROLLOUTS="${RUN_TARGET_ROLLOUTS:-$run_target_rollouts}"
RUN_TARGET_JUDGING="${RUN_TARGET_JUDGING:-$run_target_judging}"
RUN_ORACLE_ROLLOUTS="${RUN_ORACLE_ROLLOUTS:-$run_oracle_rollouts}"
RUN_ORACLE_JUDGING="${RUN_ORACLE_JUDGING:-$run_oracle_judging}"
TARGET_LORA_PATH="${TARGET_LORA_PATH:-$target_lora_path}"
JUDGE_LORA_PATH="${JUDGE_LORA_PATH:-$judge_lora_path}"
ORACLE_LORA_PATH="${ORACLE_LORA_PATH:-$oracle_lora_path}"
ORACLE_ADAPTER_PATH_SET="false"
ORACLE_ADAPTER_NAME_SET="false"
ORACLE_LORA_PATH_SET="false"
MODE_SET="false"
K_ROLLOUTS_SET="false"
NUM_ORACLE_ROLLOUTS_SET="false"
RUN_TARGET_ROLLOUTS_SET="false"
RUN_TARGET_JUDGING_SET="false"
RUN_ORACLE_ROLLOUTS_SET="false"
RUN_ORACLE_JUDGING_SET="false"
TARGET_LORA_PATH_SET="false"
JUDGE_LORA_PATH_SET="false"
EXPERIMENT_PRESET_SET="false"

EXTRA_ENV_VARS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; MODE_SET="true"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --num-rollouts) NUM_ROLLOUTS="$2"; shift 2 ;;
    --k-rollouts) K_ROLLOUTS="$2"; K_ROLLOUTS_SET="true"; shift 2 ;;
    --num-oracle-rollouts) NUM_ORACLE_ROLLOUTS="$2"; NUM_ORACLE_ROLLOUTS_SET="true"; shift 2 ;;
    --target-prompt-limit) TARGET_PROMPT_LIMIT="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --oracle-max-new-tokens) ORACLE_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --oracle-eval-batch-size) ORACLE_EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --oracle-judge-batch-size) ORACLE_JUDGE_BATCH_SIZE="$2"; shift 2 ;;
    --judge-thinking) JUDGE_THINKING="$2"; shift 2 ;;
    --judge-instruction-path) JUDGE_INSTRUCTION_PATH="$2"; shift 2 ;;
    --oracle-prompts-path) ORACLE_PROMPTS_PATH="$2"; shift 2 ;;
    --oracle-adapter-path) ORACLE_ADAPTER_PATH="$2"; ORACLE_ADAPTER_PATH_SET="true"; shift 2 ;;
    --oracle-adapter-name) ORACLE_ADAPTER_NAME="$2"; ORACLE_ADAPTER_NAME_SET="true"; shift 2 ;;
    --preset) EXPERIMENT_PRESET="$2"; EXPERIMENT_PRESET_SET="true"; shift 2 ;;
    --run-target-rollouts) RUN_TARGET_ROLLOUTS="$2"; RUN_TARGET_ROLLOUTS_SET="true"; shift 2 ;;
    --run-target-judging) RUN_TARGET_JUDGING="$2"; RUN_TARGET_JUDGING_SET="true"; shift 2 ;;
    --run-oracle-rollouts) RUN_ORACLE_ROLLOUTS="$2"; RUN_ORACLE_ROLLOUTS_SET="true"; shift 2 ;;
    --run-oracle-judging) RUN_ORACLE_JUDGING="$2"; RUN_ORACLE_JUDGING_SET="true"; shift 2 ;;
    --target-lora-path) TARGET_LORA_PATH="$2"; TARGET_LORA_PATH_SET="true"; shift 2 ;;
    --judge-lora-path) JUDGE_LORA_PATH="$2"; JUDGE_LORA_PATH_SET="true"; shift 2 ;;
    --oracle-lora-path) ORACLE_LORA_PATH="$2"; ORACLE_LORA_PATH_SET="true"; shift 2 ;;
    --wandb) WANDB_SETTING="$2"; shift 2 ;;
    --wandb-run-name) WANDB_RUN_NAME="$2"; shift 2 ;;
    --set) EXTRA_ENV_VARS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

set_preset_if_unset() {
  local var_name="$1"
  local value="$2"
  local cli_set="$3"
  local from_env="$4"
  if [[ "$cli_set" == "true" || -n "$from_env" ]]; then
    return
  fi
  printf -v "$var_name" '%s' "$value"
}

MAPPED_MODEL_VALUES_PRESET="$(lookup_oracle_adapter_for_model "$MODEL_NAME" || true)"
MAPPED_ORACLE_ADAPTER_NAME_PRESET="${MAPPED_MODEL_VALUES_PRESET##*|}"
if [[ "$ORACLE_ADAPTER_NAME_SET" != "true" && -z "${ORACLE_ADAPTER_NAME_FROM_ENV}" && -n "${MAPPED_MODEL_VALUES_PRESET}" ]]; then
  ORACLE_ADAPTER_NAME="$MAPPED_ORACLE_ADAPTER_NAME_PRESET"
fi

case "$EXPERIMENT_PRESET" in
  "")
    ;;
  full_deterministic_oracle)
    set_preset_if_unset MODE "all_target_deterministic" "$MODE_SET" "$MODE_FROM_ENV"
    set_preset_if_unset RUN_TARGET_ROLLOUTS "true" "$RUN_TARGET_ROLLOUTS_SET" "$RUN_TARGET_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_TARGET_JUDGING "true" "$RUN_TARGET_JUDGING_SET" "$RUN_TARGET_JUDGING_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_ROLLOUTS "true" "$RUN_ORACLE_ROLLOUTS_SET" "$RUN_ORACLE_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_JUDGING "true" "$RUN_ORACLE_JUDGING_SET" "$RUN_ORACLE_JUDGING_FROM_ENV"
    ;;
  sampled_target_repeats)
    set_preset_if_unset MODE "sampled_target_repeats" "$MODE_SET" "$MODE_FROM_ENV"
    set_preset_if_unset RUN_TARGET_ROLLOUTS "true" "$RUN_TARGET_ROLLOUTS_SET" "$RUN_TARGET_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_TARGET_JUDGING "true" "$RUN_TARGET_JUDGING_SET" "$RUN_TARGET_JUDGING_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_ROLLOUTS "true" "$RUN_ORACLE_ROLLOUTS_SET" "$RUN_ORACLE_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_JUDGING "true" "$RUN_ORACLE_JUDGING_SET" "$RUN_ORACLE_JUDGING_FROM_ENV"
    set_preset_if_unset K_ROLLOUTS "10" "$K_ROLLOUTS_SET" "$K_ROLLOUTS_FROM_ENV"
    set_preset_if_unset NUM_ORACLE_ROLLOUTS "2" "$NUM_ORACLE_ROLLOUTS_SET" "$NUM_ORACLE_ROLLOUTS_FROM_ENV"
    ;;
  prompt_only_oracle)
    set_preset_if_unset MODE "prompt_only_repeats" "$MODE_SET" "$MODE_FROM_ENV"
    set_preset_if_unset RUN_TARGET_ROLLOUTS "false" "$RUN_TARGET_ROLLOUTS_SET" "$RUN_TARGET_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_TARGET_JUDGING "false" "$RUN_TARGET_JUDGING_SET" "$RUN_TARGET_JUDGING_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_ROLLOUTS "true" "$RUN_ORACLE_ROLLOUTS_SET" "$RUN_ORACLE_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_JUDGING "true" "$RUN_ORACLE_JUDGING_SET" "$RUN_ORACLE_JUDGING_FROM_ENV"
    set_preset_if_unset NUM_ORACLE_ROLLOUTS "4" "$NUM_ORACLE_ROLLOUTS_SET" "$NUM_ORACLE_ROLLOUTS_FROM_ENV"
    ;;
  oracle_target_control)
    set_preset_if_unset MODE "all_target_deterministic" "$MODE_SET" "$MODE_FROM_ENV"
    set_preset_if_unset RUN_TARGET_ROLLOUTS "true" "$RUN_TARGET_ROLLOUTS_SET" "$RUN_TARGET_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_TARGET_JUDGING "true" "$RUN_TARGET_JUDGING_SET" "$RUN_TARGET_JUDGING_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_ROLLOUTS "false" "$RUN_ORACLE_ROLLOUTS_SET" "$RUN_ORACLE_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_JUDGING "false" "$RUN_ORACLE_JUDGING_SET" "$RUN_ORACLE_JUDGING_FROM_ENV"
    set_preset_if_unset TARGET_LORA_PATH "$ORACLE_ADAPTER_NAME" "$TARGET_LORA_PATH_SET" "$TARGET_LORA_PATH_FROM_ENV"
    set_preset_if_unset ORACLE_LORA_PATH "$ORACLE_ADAPTER_NAME" "$ORACLE_LORA_PATH_SET" "$ORACLE_LORA_PATH_FROM_ENV"
    ;;
  target_judging_only)
    set_preset_if_unset MODE "all_target_deterministic" "$MODE_SET" "$MODE_FROM_ENV"
    set_preset_if_unset RUN_TARGET_ROLLOUTS "true" "$RUN_TARGET_ROLLOUTS_SET" "$RUN_TARGET_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_TARGET_JUDGING "true" "$RUN_TARGET_JUDGING_SET" "$RUN_TARGET_JUDGING_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_ROLLOUTS "false" "$RUN_ORACLE_ROLLOUTS_SET" "$RUN_ORACLE_ROLLOUTS_FROM_ENV"
    set_preset_if_unset RUN_ORACLE_JUDGING "false" "$RUN_ORACLE_JUDGING_SET" "$RUN_ORACLE_JUDGING_FROM_ENV"
    ;;
esac

MAPPED_MODEL_VALUES="$(lookup_oracle_adapter_for_model "$MODEL_NAME" || true)"
MAPPED_ORACLE_ADAPTER_PATH="${MAPPED_MODEL_VALUES%%|*}"
MAPPED_ORACLE_ADAPTER_NAME="${MAPPED_MODEL_VALUES##*|}"

if [[ "$ORACLE_ADAPTER_PATH_SET" != "true" && -z "${ORACLE_ADAPTER_PATH_FROM_ENV}" ]]; then
  if [[ -n "${MAPPED_MODEL_VALUES}" ]]; then
    ORACLE_ADAPTER_PATH="$MAPPED_ORACLE_ADAPTER_PATH"
  else
    echo "No mapped oracle adapter path for MODEL_NAME=$MODEL_NAME. Set --oracle-adapter-path explicitly." >&2
    exit 1
  fi
fi

if [[ "$ORACLE_ADAPTER_NAME_SET" != "true" && -z "${ORACLE_ADAPTER_NAME_FROM_ENV}" ]]; then
  if [[ -n "${MAPPED_MODEL_VALUES}" ]]; then
    ORACLE_ADAPTER_NAME="$MAPPED_ORACLE_ADAPTER_NAME"
  else
    ORACLE_ADAPTER_NAME="$oracle_adapter_name"
  fi
fi

if [[ "$ORACLE_LORA_PATH_SET" != "true" && -z "${ORACLE_LORA_PATH_FROM_ENV}" ]]; then
  ORACLE_LORA_PATH="$ORACLE_ADAPTER_NAME"
fi

if [[ -z "${ORACLE_ADAPTER_PATH:-}" ]]; then
  echo "ORACLE_ADAPTER_PATH cannot be empty." >&2
  exit 1
fi
if [[ -z "${ORACLE_ADAPTER_NAME:-}" ]]; then
  echo "ORACLE_ADAPTER_NAME cannot be empty." >&2
  exit 1
fi

case "$MODE" in
  all_target_deterministic|sampled_target_repeats|prompt_only_repeats) ;;
  *)
    echo "Invalid --mode: $MODE" >&2
    usage
    exit 1
    ;;
esac

case "$WANDB_SETTING" in
  on|off) ;;
  *)
    echo "Invalid --wandb setting: $WANDB_SETTING (expected 'on' or 'off')" >&2
    usage
    exit 1
    ;;
esac

case "$EXPERIMENT_PRESET" in
  ""|full_deterministic_oracle|sampled_target_repeats|prompt_only_oracle|oracle_target_control|target_judging_only) ;;
  *)
    echo "Invalid --preset: $EXPERIMENT_PRESET (supported: full_deterministic_oracle, sampled_target_repeats, prompt_only_oracle, oracle_target_control, target_judging_only)" >&2
    usage
    exit 1
    ;;
esac

case "$RUN_TARGET_ROLLOUTS" in
  true|false) ;;
  *)
    echo "Invalid --run-target-rollouts setting: $RUN_TARGET_ROLLOUTS (expected 'true' or 'false')" >&2
    usage
    exit 1
    ;;
esac
case "$RUN_TARGET_JUDGING" in
  true|false) ;;
  *)
    echo "Invalid --run-target-judging setting: $RUN_TARGET_JUDGING (expected 'true' or 'false')" >&2
    usage
    exit 1
    ;;
esac
case "$RUN_ORACLE_ROLLOUTS" in
  true|false) ;;
  *)
    echo "Invalid --run-oracle-rollouts setting: $RUN_ORACLE_ROLLOUTS (expected 'true' or 'false')" >&2
    usage
    exit 1
    ;;
esac
case "$RUN_ORACLE_JUDGING" in
  true|false) ;;
  *)
    echo "Invalid --run-oracle-judging setting: $RUN_ORACLE_JUDGING (expected 'true' or 'false')" >&2
    usage
    exit 1
    ;;
esac
case "$JUDGE_THINKING" in
  default|off) ;;
  *)
    echo "Invalid --judge-thinking setting: $JUDGE_THINKING (expected 'default' or 'off')" >&2
    usage
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if (( ${#EXTRA_ENV_VARS[@]} > 0 )); then
  for kv in "${EXTRA_ENV_VARS[@]}"; do
    if [[ "$kv" != *=* ]]; then
      echo "Invalid --set value '$kv' (expected KEY=VALUE)" >&2
      exit 1
    fi
    export "$kv"
  done
fi

export ORACLE_ROLLOUT_MODE="$MODE"
export MODEL_NAME="$MODEL_NAME"
export NUM_ROLLOUTS="$NUM_ROLLOUTS"
export K_ROLLOUTS="$K_ROLLOUTS"
export NUM_ORACLE_ROLLOUTS="$NUM_ORACLE_ROLLOUTS"
export TARGET_PROMPT_LIMIT="$TARGET_PROMPT_LIMIT"
export MAX_NEW_TOKENS="$MAX_NEW_TOKENS"
export ORACLE_MAX_NEW_TOKENS="$ORACLE_MAX_NEW_TOKENS"
export ORACLE_EVAL_BATCH_SIZE="$ORACLE_EVAL_BATCH_SIZE"
export ORACLE_JUDGE_BATCH_SIZE="$ORACLE_JUDGE_BATCH_SIZE"
export JUDGE_THINKING="$JUDGE_THINKING"
export JUDGE_INSTRUCTION_PATH="$JUDGE_INSTRUCTION_PATH"
export ORACLE_ADAPTER_PATH="$ORACLE_ADAPTER_PATH"
export ORACLE_ADAPTER_NAME="$ORACLE_ADAPTER_NAME"
export EXPERIMENT_PRESET="$EXPERIMENT_PRESET"
export RUN_TARGET_ROLLOUTS="$RUN_TARGET_ROLLOUTS"
export RUN_TARGET_JUDGING="$RUN_TARGET_JUDGING"
export RUN_ORACLE_ROLLOUTS="$RUN_ORACLE_ROLLOUTS"
export RUN_ORACLE_JUDGING="$RUN_ORACLE_JUDGING"
export TARGET_LORA_PATH="$TARGET_LORA_PATH"
export JUDGE_LORA_PATH="$JUDGE_LORA_PATH"
export ORACLE_LORA_PATH="$ORACLE_LORA_PATH"

if [[ -n "$ORACLE_PROMPTS_PATH" ]]; then
  export ORACLE_PROMPTS_PATH="$ORACLE_PROMPTS_PATH"
fi
if [[ -n "$WANDB_RUN_NAME" ]]; then
  export WANDB_RUN_NAME="$WANDB_RUN_NAME"
fi

if [[ "$WANDB_SETTING" == "off" ]]; then
  export WANDB_MODE="disabled"
  unset WANDB_API_KEY || true
else
  unset WANDB_MODE || true
fi

cat <<EOF
Running bypass_refusal.py with:
  MODEL_NAME=$MODEL_NAME
  ORACLE_ROLLOUT_MODE=$ORACLE_ROLLOUT_MODE
  NUM_ROLLOUTS=$NUM_ROLLOUTS
  K_ROLLOUTS=$K_ROLLOUTS
  NUM_ORACLE_ROLLOUTS=$NUM_ORACLE_ROLLOUTS
  TARGET_PROMPT_LIMIT=$TARGET_PROMPT_LIMIT
  MAX_NEW_TOKENS=$MAX_NEW_TOKENS
  ORACLE_MAX_NEW_TOKENS=$ORACLE_MAX_NEW_TOKENS
  ORACLE_EVAL_BATCH_SIZE=$ORACLE_EVAL_BATCH_SIZE
  ORACLE_JUDGE_BATCH_SIZE=$ORACLE_JUDGE_BATCH_SIZE
  JUDGE_THINKING=$JUDGE_THINKING
  JUDGE_INSTRUCTION_PATH=$JUDGE_INSTRUCTION_PATH
  ORACLE_PROMPTS_PATH=${ORACLE_PROMPTS_PATH:-<default>}
  ORACLE_ADAPTER_PATH=$ORACLE_ADAPTER_PATH
  ORACLE_ADAPTER_NAME=$ORACLE_ADAPTER_NAME
  EXPERIMENT_PRESET=${EXPERIMENT_PRESET:-<none>}
  RUN_TARGET_ROLLOUTS=$RUN_TARGET_ROLLOUTS
  RUN_TARGET_JUDGING=$RUN_TARGET_JUDGING
  RUN_ORACLE_ROLLOUTS=$RUN_ORACLE_ROLLOUTS
  RUN_ORACLE_JUDGING=$RUN_ORACLE_JUDGING
  TARGET_LORA_PATH=$TARGET_LORA_PATH
  JUDGE_LORA_PATH=$JUDGE_LORA_PATH
  ORACLE_LORA_PATH=$ORACLE_LORA_PATH
  WANDB=$WANDB_SETTING
  WANDB_MODE=${WANDB_MODE:-<unset>}
  WANDB_RUN_NAME=${WANDB_RUN_NAME:-<unset>}
EOF

python bypass_refusal.py
