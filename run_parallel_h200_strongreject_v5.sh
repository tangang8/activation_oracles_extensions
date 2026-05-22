#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"

TARGET_PROMPT_TOTAL="${TARGET_PROMPT_TOTAL:-100}"
TARGET_PROMPT_SPLIT="${TARGET_PROMPT_SPLIT:-50}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
NUM_ORACLE_ROLLOUTS="${NUM_ORACLE_ROLLOUTS:-50}"

DEFAULT_ORACLE_PROMPTS_PATH="${DEFAULT_ORACLE_PROMPTS_PATH:-prompts/oracle_prompts/default_oracle_prompts.json}"
SECOND_ORACLE_PROMPTS_PATH="${SECOND_ORACLE_PROMPTS_PATH:-prompts/oracle_prompts/model_answer_min_200_words.json}"
JUDGE_INSTRUCTION_PATH="${JUDGE_INSTRUCTION_PATH:-strongReject_v5.jinja2}"
JUDGE_THINKING="${JUDGE_THINKING:-off}"
LOG_ROOT="${LOG_ROOT:-logs/parallel_h200_${timestamp}}"
WANDB_SETTING="${WANDB_SETTING:-on}"

GPU_TARGET_A="${GPU_TARGET_A:-0}"
GPU_PROMPT_ONLY="${GPU_PROMPT_ONLY:-1}"
GPU_ORACLE_TARGET_CONTROL="${GPU_ORACLE_TARGET_CONTROL:-2}"
GPU_TARGET_B="${GPU_TARGET_B:-3}"

TARGET_JUDGE_BATCH_SIZE="${TARGET_JUDGE_BATCH_SIZE:-64}"
ORACLE_EVAL_BATCH_SIZE="${ORACLE_EVAL_BATCH_SIZE:-128}"
ORACLE_JUDGE_BATCH_SIZE="${ORACLE_JUDGE_BATCH_SIZE:-64}"

FULL_DETERMINISTIC_PRESET="${FULL_DETERMINISTIC_PRESET:-rollout_post_prompt_oracle}"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$LOG_ROOT"
DRIVER_LOG="$LOG_ROOT/parallel_driver.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$DRIVER_LOG"
}

die() {
  log "ERROR: $*"
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Required file does not exist: $path"
}

ensure_distinct_gpus() {
  local seen=" "
  local gpu=""
  for gpu in "$@"; do
    if [[ "$seen" == *" $gpu "* ]]; then
      die "GPU assignments must be distinct; duplicate GPU=$gpu"
    fi
    seen+="$gpu "
  done
}

is_oom_log() {
  local log_file="$1"
  grep -Eiq 'out of memory|cuda.*oom|cuda.*out.*memory|cublas.*alloc|cannot allocate memory' "$log_file"
}

quote_cmd() {
  printf '%q ' "$@"
}

run_attempt() {
  local gpu="$1"
  local log_file="$2"
  shift 2

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN GPU $gpu: CUDA_VISIBLE_DEVICES=$gpu $(quote_cmd "$@")"
    return 0
  fi

  log "Starting GPU $gpu attempt: $(quote_cmd "$@")"
  log "Attempt log: $log_file"
  if CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$log_file" 2>&1; then
    log "Finished GPU $gpu attempt successfully: $(basename "$log_file")"
    return 0
  fi
  local status=$?
  log "GPU $gpu attempt failed with exit code $status: $(basename "$log_file")"
  return "$status"
}

run_target_job() {
  local job_name="$1"
  local gpu="$2"
  local offset="$3"
  local limit="$4"
  local batch_sizes=("$TARGET_JUDGE_BATCH_SIZE" 32 16 8)
  local batch=""
  local attempt=0
  local log_file=""

  for batch in "${batch_sizes[@]}"; do
    attempt=$((attempt + 1))
    log_file="$LOG_ROOT/${job_name}_attempt-${attempt}_target-judge-batch-${batch}.log"
    if run_attempt "$gpu" "$log_file" ./run_oracle_experiment.sh \
      --preset target_judging_only \
      --target-prompt-offset "$offset" \
      --target-prompt-limit "$limit" \
      --num-rollouts "$NUM_ROLLOUTS" \
      --target-judge-batch-size "$batch" \
      --judge-instruction-path "$JUDGE_INSTRUCTION_PATH" \
      --judge-thinking "$JUDGE_THINKING" \
      --wandb "$WANDB_SETTING" \
      --wandb-run-name "$job_name"; then
      return 0
    fi
    if [[ "$DRY_RUN" == "1" ]] || ! is_oom_log "$log_file"; then
      return 1
    fi
    log "$job_name hit OOM; retrying with next target judge batch size."
  done

  log "$job_name exhausted target judge OOM retry ladder."
  return 1
}

run_oracle_job() {
  local job_name="$1"
  local gpu="$2"
  shift 2
  local base_args=("$@")
  local eval_batches=( "$ORACLE_EVAL_BATCH_SIZE" 128 64 64 32 32 )
  local judge_batches=( "$ORACLE_JUDGE_BATCH_SIZE" 32 32 16 16 8 )
  local seen_pairs=" "
  local attempt=0
  local idx=""
  local eval_batch=""
  local judge_batch=""
  local pair=""
  local log_file=""

  for idx in "${!eval_batches[@]}"; do
    eval_batch="${eval_batches[$idx]}"
    judge_batch="${judge_batches[$idx]}"
    pair="${eval_batch}/${judge_batch}"
    if [[ "$seen_pairs" == *" $pair "* ]]; then
      continue
    fi
    seen_pairs+="$pair "
    attempt=$((attempt + 1))
    log_file="$LOG_ROOT/${job_name}_attempt-${attempt}_oracle-eval-${eval_batch}_oracle-judge-${judge_batch}.log"
    if run_attempt "$gpu" "$log_file" ./run_oracle_experiment.sh \
      "${base_args[@]}" \
      --oracle-eval-batch-size "$eval_batch" \
      --oracle-judge-batch-size "$judge_batch" \
      --judge-instruction-path "$JUDGE_INSTRUCTION_PATH" \
      --judge-thinking "$JUDGE_THINKING" \
      --wandb "$WANDB_SETTING" \
      --wandb-run-name "$job_name"; then
      return 0
    fi
    if [[ "$DRY_RUN" == "1" ]] || ! is_oom_log "$log_file"; then
      return 1
    fi
    log "$job_name hit OOM; retrying with next oracle eval/judge batch pair."
  done

  log "$job_name exhausted oracle OOM retry ladder."
  return 1
}

run_prompt_only_sequence() {
  local gpu="$GPU_PROMPT_ONLY"
  run_oracle_job "gpu${gpu}_prompt_only_default" "$gpu" \
    --preset prompt_only_oracle \
    --target-prompt-offset 0 \
    --target-prompt-limit "$TARGET_PROMPT_TOTAL" \
    --num-oracle-rollouts "$NUM_ORACLE_ROLLOUTS" \
    --oracle-prompts-path "$DEFAULT_ORACLE_PROMPTS_PATH" || return 1

  run_oracle_job "gpu${gpu}_prompt_only_min_200" "$gpu" \
    --preset prompt_only_oracle \
    --target-prompt-offset 0 \
    --target-prompt-limit "$TARGET_PROMPT_TOTAL" \
    --num-oracle-rollouts "$NUM_ORACLE_ROLLOUTS" \
    --oracle-prompts-path "$SECOND_ORACLE_PROMPTS_PATH"
}

run_target_then_full_sequence() {
  local label="$1"
  local gpu="$2"
  local offset="$3"
  local limit="$4"

  run_target_job "gpu${gpu}_target_judging_shard_${label}" "$gpu" "$offset" "$limit" || return 1

  run_oracle_job "gpu${gpu}_full_deterministic_shard_${label}_default" "$gpu" \
    --preset "$FULL_DETERMINISTIC_PRESET" \
    --target-prompt-offset "$offset" \
    --target-prompt-limit "$limit" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --oracle-prompts-path "$DEFAULT_ORACLE_PROMPTS_PATH" || return 1

  run_oracle_job "gpu${gpu}_full_deterministic_shard_${label}_min_200" "$gpu" \
    --preset "$FULL_DETERMINISTIC_PRESET" \
    --target-prompt-offset "$offset" \
    --target-prompt-limit "$limit" \
    --num-rollouts "$NUM_ROLLOUTS" \
    --oracle-prompts-path "$SECOND_ORACLE_PROMPTS_PATH"
}

run_oracle_target_control_once() {
  local gpu="$GPU_ORACLE_TARGET_CONTROL"
  local batch_sizes=("$TARGET_JUDGE_BATCH_SIZE" 32 16 8)
  local batch=""
  local attempt=0
  local log_file=""

  for batch in "${batch_sizes[@]}"; do
    attempt=$((attempt + 1))
    log_file="$LOG_ROOT/gpu${gpu}_oracle_target_control_attempt-${attempt}_target-judge-batch-${batch}.log"
    if run_attempt "$gpu" "$log_file" ./run_oracle_experiment.sh \
      --preset oracle_target_control \
      --target-prompt-offset 0 \
      --target-prompt-limit "$TARGET_PROMPT_TOTAL" \
      --num-rollouts "$NUM_ROLLOUTS" \
      --target-judge-batch-size "$batch" \
      --judge-instruction-path "$JUDGE_INSTRUCTION_PATH" \
      --judge-thinking "$JUDGE_THINKING" \
      --wandb "$WANDB_SETTING" \
      --wandb-run-name "gpu${gpu}_oracle_target_control"; then
      return 0
    fi
    if [[ "$DRY_RUN" == "1" ]] || ! is_oom_log "$log_file"; then
      return 1
    fi
    log "oracle_target_control hit OOM; retrying with next target judge batch size."
  done

  log "oracle_target_control exhausted target judge OOM retry ladder."
  return 1
}

require_file "$DEFAULT_ORACLE_PROMPTS_PATH"
require_file "$SECOND_ORACLE_PROMPTS_PATH"
ensure_distinct_gpus "$GPU_TARGET_A" "$GPU_PROMPT_ONLY" "$GPU_ORACLE_TARGET_CONTROL" "$GPU_TARGET_B"

shard_a_offset=0
shard_a_limit="$TARGET_PROMPT_SPLIT"
shard_b_offset="$TARGET_PROMPT_SPLIT"
shard_b_limit=$((TARGET_PROMPT_TOTAL - TARGET_PROMPT_SPLIT))

if (( shard_b_limit < 0 )); then
  die "TARGET_PROMPT_SPLIT ($TARGET_PROMPT_SPLIT) cannot exceed TARGET_PROMPT_TOTAL ($TARGET_PROMPT_TOTAL)."
fi

log "Parallel H200 run starting. Logs: $LOG_ROOT"
log "Shard A offset=$shard_a_offset limit=$shard_a_limit on GPU $GPU_TARGET_A"
log "Shard B offset=$shard_b_offset limit=$shard_b_limit on GPU $GPU_TARGET_B"
log "Prompt-only default then min-200 on GPU $GPU_PROMPT_ONLY"
log "Oracle target control once on GPU $GPU_ORACLE_TARGET_CONTROL"
log "Full deterministic oracle preset: $FULL_DETERMINISTIC_PRESET"

pids=()
names=()

run_target_then_full_sequence "A" "$GPU_TARGET_A" "$shard_a_offset" "$shard_a_limit" &
pids+=("$!")
names+=("target/full shard A")

run_target_then_full_sequence "B" "$GPU_TARGET_B" "$shard_b_offset" "$shard_b_limit" &
pids+=("$!")
names+=("target/full shard B")

run_prompt_only_sequence &
pids+=("$!")
names+=("prompt-only wave 1/2")

run_oracle_target_control_once &
pids+=("$!")
names+=("oracle-target-control")

failures=0
for idx in "${!pids[@]}"; do
  if wait "${pids[$idx]}"; then
    log "Sequence completed: ${names[$idx]}"
  else
    log "Sequence failed: ${names[$idx]}"
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  die "$failures sequence(s) failed. Check logs under $LOG_ROOT."
fi

log "All parallel H200 sequences completed successfully."
