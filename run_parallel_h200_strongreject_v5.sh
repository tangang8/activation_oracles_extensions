#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_LABEL="${RUN_LABEL:-parallel_h200_${timestamp}}"
RUN_ORACLE_EXPERIMENT="${RUN_ORACLE_EXPERIMENT:-./run_oracle_experiment.sh}"

TARGET_PROMPT_TOTAL="${TARGET_PROMPT_TOTAL:-100}"
TARGET_PROMPT_SPLIT="${TARGET_PROMPT_SPLIT:-50}"
DETERMINISTIC_SHARD_COUNT="${DETERMINISTIC_SHARD_COUNT:-10}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
NUM_ORACLE_ROLLOUTS="${NUM_ORACLE_ROLLOUTS:-50}"

DEFAULT_ORACLE_PROMPTS_PATH="${DEFAULT_ORACLE_PROMPTS_PATH:-prompts/oracle_prompts/default_oracle_prompts.json}"
SECOND_ORACLE_PROMPTS_PATH="${SECOND_ORACLE_PROMPTS_PATH:-prompts/oracle_prompts/model_answer_min_200_words.json}"
ORACLE_PROMPTS_PATHS="${ORACLE_PROMPTS_PATHS:-$DEFAULT_ORACLE_PROMPTS_PATH,$SECOND_ORACLE_PROMPTS_PATH}"
JUDGE_INSTRUCTION_PATH="${JUDGE_INSTRUCTION_PATH:-strongReject_v5.jinja2}"
JUDGE_THINKING="${JUDGE_THINKING:-off}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_LABEL}}"
WANDB_SETTING="${WANDB_SETTING:-on}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"

TARGET_JUDGE_BATCH_SIZE="${TARGET_JUDGE_BATCH_SIZE:-64}"
ORACLE_EVAL_BATCH_SIZE="${ORACLE_EVAL_BATCH_SIZE:-128}"
ORACLE_JUDGE_BATCH_SIZE="${ORACLE_JUDGE_BATCH_SIZE:-64}"

FULL_DETERMINISTIC_PRESET="${FULL_DETERMINISTIC_PRESET:-rollout_post_prompt_oracle}"

DRY_RUN="${DRY_RUN:-0}"
SCHEDULER_POLL_SECONDS="${SCHEDULER_POLL_SECONDS:-5}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

is_oom_log() {
  local log_file="$1"
  grep -Eiq 'torch\.OutOfMemoryError|CUDA out of memory|out of memory|cuda.*oom|cuda.*out.*memory|cublas.*alloc|cannot allocate memory' "$log_file"
}

quote_cmd() {
  printf '%q ' "$@"
}

contains_csv_item() {
  local csv="$1"
  local item="$2"
  local part=""
  [[ -z "$csv" ]] && return 1
  IFS=',' read -r -a parts <<< "$csv"
  for part in "${parts[@]}"; do
    [[ "$part" == "$item" ]] && return 0
  done
  return 1
}

validate_unique_array() {
  local label="$1"
  shift
  local item=""
  local seen=" "
  if (( $# == 0 )); then
    die "$label must include at least one value."
  fi
  for item in "$@"; do
    if [[ -z "$item" ]]; then
      die "$label contains an empty value."
    fi
    if [[ "$seen" == *" $item "* ]]; then
      die "$label contains duplicate value: $item"
    fi
    seen+="$item "
  done
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
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" "$@" >"$log_file" 2>&1
  local status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    log "Finished GPU $gpu attempt successfully: $(basename "$log_file")"
    return 0
  fi
  log "GPU $gpu attempt failed with exit code $status: $(basename "$log_file")"
  return "$status"
}

run_target_ladder() {
  local job_idx="$1"
  local gpu="$2"
  local job_id="${JOB_ID[$job_idx]}"
  local preset="${JOB_PRESET[$job_idx]}"
  local offset="${JOB_OFFSET[$job_idx]}"
  local limit="${JOB_LIMIT[$job_idx]}"
  local batch_sizes=("$TARGET_JUDGE_BATCH_SIZE" 32 16 8)
  local batch=""
  local attempt=0
  local log_file=""

  for batch in "${batch_sizes[@]}"; do
    attempt=$((attempt + 1))
    log_file="$LOG_ROOT/${job_id}_attempt-${attempt}_target-judge-batch-${batch}.log"
    if [[ "${JOB_TYPE[$job_idx]}" == "target_control" ]]; then
      if run_attempt "$gpu" "$log_file" "$RUN_ORACLE_EXPERIMENT" \
        --preset "$preset" \
        --target-prompt-offset "$offset" \
        --target-prompt-limit "$limit" \
        --num-rollouts "$NUM_ROLLOUTS" \
        --target-judge-batch-size "$batch" \
        --target-thinking off \
        --judge-instruction-path "$JUDGE_INSTRUCTION_PATH" \
        --judge-thinking "$JUDGE_THINKING" \
        --wandb "$WANDB_SETTING" \
        --wandb-run-name "${RUN_LABEL}__${job_id}" \
        --set "WANDB_GROUP=$RUN_LABEL" \
        --set "WANDB_JOB_TYPE=$job_id"; then
        return 0
      fi
    else
      if run_attempt "$gpu" "$log_file" "$RUN_ORACLE_EXPERIMENT" \
        --preset "$preset" \
        --target-prompt-offset "$offset" \
        --target-prompt-limit "$limit" \
        --num-rollouts "$NUM_ROLLOUTS" \
        --target-judge-batch-size "$batch" \
        --judge-instruction-path "$JUDGE_INSTRUCTION_PATH" \
        --judge-thinking "$JUDGE_THINKING" \
        --wandb "$WANDB_SETTING" \
        --wandb-run-name "${RUN_LABEL}__${job_id}" \
        --set "WANDB_GROUP=$RUN_LABEL" \
        --set "WANDB_JOB_TYPE=$job_id"; then
        return 0
      fi
    fi
    if [[ "$DRY_RUN" == "1" ]] || ! is_oom_log "$log_file"; then
      return 1
    fi
    log "$job_id hit OOM; retrying with next target judge batch size."
  done

  log "$job_id exhausted target judge OOM retry ladder."
  return 1
}

run_oracle_ladder() {
  local job_idx="$1"
  local gpu="$2"
  local job_id="${JOB_ID[$job_idx]}"
  local preset="${JOB_PRESET[$job_idx]}"
  local offset="${JOB_OFFSET[$job_idx]}"
  local limit="${JOB_LIMIT[$job_idx]}"
  local oracle_prompts_path="${JOB_ORACLE_PROMPTS_PATH[$job_idx]}"
  local eval_batches=( "$ORACLE_EVAL_BATCH_SIZE" 128 64 64 32 32 )
  local judge_batches=( "$ORACLE_JUDGE_BATCH_SIZE" 32 32 16 16 8 )
  local seen_pairs=" "
  local attempt=0
  local idx=""
  local eval_batch=""
  local judge_batch=""
  local pair=""
  local log_file=""
  local mode_args=()

  if [[ "${JOB_TYPE[$job_idx]}" == "prompt_only" ]]; then
    mode_args=(--num-oracle-rollouts "$NUM_ORACLE_ROLLOUTS")
  else
    mode_args=(--num-rollouts "$NUM_ROLLOUTS")
  fi

  for idx in "${!eval_batches[@]}"; do
    eval_batch="${eval_batches[$idx]}"
    judge_batch="${judge_batches[$idx]}"
    pair="${eval_batch}/${judge_batch}"
    if [[ "$seen_pairs" == *" $pair "* ]]; then
      continue
    fi
    seen_pairs+="$pair "
    attempt=$((attempt + 1))
    log_file="$LOG_ROOT/${job_id}_attempt-${attempt}_oracle-eval-${eval_batch}_oracle-judge-${judge_batch}.log"
    if run_attempt "$gpu" "$log_file" "$RUN_ORACLE_EXPERIMENT" \
      --preset "$preset" \
      --target-prompt-offset "$offset" \
      --target-prompt-limit "$limit" \
      "${mode_args[@]}" \
      --oracle-prompts-path "$oracle_prompts_path" \
      --oracle-eval-batch-size "$eval_batch" \
      --oracle-judge-batch-size "$judge_batch" \
      --judge-instruction-path "$JUDGE_INSTRUCTION_PATH" \
      --judge-thinking "$JUDGE_THINKING" \
      --wandb "$WANDB_SETTING" \
      --wandb-run-name "${RUN_LABEL}__${job_id}" \
      --set "WANDB_GROUP=$RUN_LABEL" \
      --set "WANDB_JOB_TYPE=$job_id"; then
      return 0
    fi
    if [[ "$DRY_RUN" == "1" ]] || ! is_oom_log "$log_file"; then
      return 1
    fi
    log "$job_id hit OOM; retrying with next oracle eval/judge batch pair."
  done

  log "$job_id exhausted oracle OOM retry ladder."
  return 1
}

run_job_with_retries() {
  local job_idx="$1"
  local gpu="$2"
  case "${JOB_LADDER[$job_idx]}" in
    target_judge) run_target_ladder "$job_idx" "$gpu" ;;
    oracle_eval_judge) run_oracle_ladder "$job_idx" "$gpu" ;;
    *) log "ERROR: Unknown ladder for ${JOB_ID[$job_idx]}: ${JOB_LADDER[$job_idx]}"; return 1 ;;
  esac
}

declare -a GPU_ID_ARRAY=()
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
validate_unique_array "GPU_IDS" "${GPU_ID_ARRAY[@]}"

declare -a ORACLE_PROMPT_PATH_ARRAY=()
IFS=',' read -r -a ORACLE_PROMPT_PATH_ARRAY <<< "$ORACLE_PROMPTS_PATHS"
validate_unique_array "ORACLE_PROMPTS_PATHS" "${ORACLE_PROMPT_PATH_ARRAY[@]}"
for prompt_path in "${ORACLE_PROMPT_PATH_ARRAY[@]}"; do
  require_file "$prompt_path"
done

shard_a_offset=0
shard_a_limit="$TARGET_PROMPT_SPLIT"
shard_b_offset="$TARGET_PROMPT_SPLIT"
shard_b_limit=$((TARGET_PROMPT_TOTAL - TARGET_PROMPT_SPLIT))

if (( shard_a_limit < 0 || shard_b_limit < 0 )); then
  die "TARGET_PROMPT_SPLIT ($TARGET_PROMPT_SPLIT) must be between 0 and TARGET_PROMPT_TOTAL ($TARGET_PROMPT_TOTAL)."
fi
if (( DETERMINISTIC_SHARD_COUNT <= 0 )); then
  die "DETERMINISTIC_SHARD_COUNT must be >= 1."
fi

declare -a JOB_ID=()
declare -a JOB_TYPE=()
declare -a JOB_PRESET=()
declare -a JOB_OFFSET=()
declare -a JOB_LIMIT=()
declare -a JOB_ORACLE_PROMPTS_PATH=()
declare -a JOB_DEPENDS_ON=()
declare -a JOB_LADDER=()
declare -a JOB_STATE=()
declare -a JOB_PID=()
declare -a JOB_GPU=()

add_job() {
  local job_id="$1"
  local job_type="$2"
  local preset="$3"
  local offset="$4"
  local limit="$5"
  local oracle_prompts_path="$6"
  local depends_on="$7"
  local ladder="$8"
  local idx="${#JOB_ID[@]}"

  JOB_ID[$idx]="$job_id"
  JOB_TYPE[$idx]="$job_type"
  JOB_PRESET[$idx]="$preset"
  JOB_OFFSET[$idx]="$offset"
  JOB_LIMIT[$idx]="$limit"
  JOB_ORACLE_PROMPTS_PATH[$idx]="$oracle_prompts_path"
  JOB_DEPENDS_ON[$idx]="$depends_on"
  JOB_LADDER[$idx]="$ladder"
  JOB_STATE[$idx]="pending"
  JOB_PID[$idx]=""
  JOB_GPU[$idx]=""
}

find_job_index_by_id() {
  local wanted="$1"
  local idx=""
  for idx in "${!JOB_ID[@]}"; do
    if [[ "${JOB_ID[$idx]}" == "$wanted" ]]; then
      printf '%s\n' "$idx"
      return 0
    fi
  done
  return 1
}

job_dependencies_satisfied() {
  local job_idx="$1"
  local deps="${JOB_DEPENDS_ON[$job_idx]}"
  local dep=""
  local dep_idx=""
  [[ -z "$deps" ]] && return 0
  IFS=',' read -r -a dep_array <<< "$deps"
  for dep in "${dep_array[@]}"; do
    dep_idx="$(find_job_index_by_id "$dep")" || return 1
    [[ "${JOB_STATE[$dep_idx]}" == "done" ]] || return 1
  done
  return 0
}

next_ready_job() {
  local idx=""
  for idx in "${!JOB_ID[@]}"; do
    if [[ "${JOB_STATE[$idx]}" == "pending" ]] && job_dependencies_satisfied "$idx"; then
      printf '%s\n' "$idx"
      return 0
    fi
  done
  return 1
}

declare -a GPU_BUSY_JOB=()
for _gpu in "${GPU_ID_ARRAY[@]}"; do
  GPU_BUSY_JOB+=("")
done

next_free_gpu_slot() {
  local slot=""
  for slot in "${!GPU_ID_ARRAY[@]}"; do
    if [[ -z "${GPU_BUSY_JOB[$slot]}" ]]; then
      printf '%s\n' "$slot"
      return 0
    fi
  done
  return 1
}

find_gpu_slot_by_id() {
  local gpu="$1"
  local slot=""
  for slot in "${!GPU_ID_ARRAY[@]}"; do
    if [[ "${GPU_ID_ARRAY[$slot]}" == "$gpu" ]]; then
      printf '%s\n' "$slot"
      return 0
    fi
  done
  return 1
}

mark_dependents_blocked() {
  local failed_job_id="$1"
  local idx=""
  for idx in "${!JOB_ID[@]}"; do
    if [[ "${JOB_STATE[$idx]}" == "pending" ]] && contains_csv_item "${JOB_DEPENDS_ON[$idx]}" "$failed_job_id"; then
      JOB_STATE[$idx]="blocked"
      log "Job blocked: ${JOB_ID[$idx]} depends_on=$failed_job_id"
      mark_dependents_blocked "${JOB_ID[$idx]}"
    fi
  done
}

launch_job() {
  local job_idx="$1"
  local gpu_slot="$2"
  local gpu="${GPU_ID_ARRAY[$gpu_slot]}"
  local job_id="${JOB_ID[$job_idx]}"

  JOB_STATE[$job_idx]="running"
  JOB_GPU[$job_idx]="$gpu"
  GPU_BUSY_JOB[$gpu_slot]="$job_idx"
  log "Launching job=$job_id gpu=$gpu type=${JOB_TYPE[$job_idx]} depends_on=${JOB_DEPENDS_ON[$job_idx]:-none}"
  (
    run_job_with_retries "$job_idx" "$gpu"
  ) &
  JOB_PID[$job_idx]="$!"
}

poll_running_jobs() {
  local idx=""
  local pid=""
  local status=0
  local gpu=""
  local gpu_slot=""
  for idx in "${!JOB_ID[@]}"; do
    [[ "${JOB_STATE[$idx]}" == "running" ]] || continue
    pid="${JOB_PID[$idx]}"
    if kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    set +e
    wait "$pid"
    status=$?
    set -e
    gpu="${JOB_GPU[$idx]}"
    gpu_slot="$(find_gpu_slot_by_id "$gpu")"
    GPU_BUSY_JOB[$gpu_slot]=""
    if [[ "$status" -eq 0 ]]; then
      JOB_STATE[$idx]="done"
      log "Job completed: ${JOB_ID[$idx]} gpu=$gpu"
    else
      JOB_STATE[$idx]="failed"
      log "Job failed: ${JOB_ID[$idx]} gpu=$gpu status=$status"
      mark_dependents_blocked "${JOB_ID[$idx]}"
    fi
  done
}

any_running_jobs() {
  local idx=""
  for idx in "${!JOB_ID[@]}"; do
    [[ "${JOB_STATE[$idx]}" == "running" ]] && return 0
  done
  return 1
}

all_jobs_terminal() {
  local idx=""
  for idx in "${!JOB_ID[@]}"; do
    case "${JOB_STATE[$idx]}" in
      done|failed|blocked) ;;
      *) return 1 ;;
    esac
  done
  return 0
}

log_job_table() {
  local idx=""
  log "Job table:"
  for idx in "${!JOB_ID[@]}"; do
    log "  job=${JOB_ID[$idx]} type=${JOB_TYPE[$idx]} preset=${JOB_PRESET[$idx]} offset=${JOB_OFFSET[$idx]} limit=${JOB_LIMIT[$idx]} oracle_prompts_path=${JOB_ORACLE_PROMPTS_PATH[$idx]:-none} depends_on=${JOB_DEPENDS_ON[$idx]:-none} ladder=${JOB_LADDER[$idx]}"
  done
}

log_summary() {
  local state=""
  local idx=""
  for state in done failed blocked pending running; do
    local items=()
    for idx in "${!JOB_ID[@]}"; do
      if [[ "${JOB_STATE[$idx]}" == "$state" ]]; then
        items+=("${JOB_ID[$idx]}")
      fi
    done
    if (( ${#items[@]} > 0 )); then
      log "Summary $state: ${items[*]}"
    else
      log "Summary $state: <none>"
    fi
  done
}

add_job "target_shard_A" "target" "target_judging_only" "$shard_a_offset" "$shard_a_limit" "" "" "target_judge"
if (( shard_b_limit > 0 )); then
  add_job "target_shard_B" "target" "target_judging_only" "$shard_b_offset" "$shard_b_limit" "" "" "target_judge"
fi

add_job "prompt_only_prompt_0" "prompt_only" "prompt_only_oracle" 0 "$TARGET_PROMPT_TOTAL" "${ORACLE_PROMPT_PATH_ARRAY[0]}" "" "oracle_eval_judge"
add_job "oracle_target_control" "target_control" "oracle_target_control" 0 "$TARGET_PROMPT_TOTAL" "" "" "target_judge"
for prompt_idx in "${!ORACLE_PROMPT_PATH_ARRAY[@]}"; do
  if (( prompt_idx == 0 )); then
    continue
  fi
  add_job "prompt_only_prompt_${prompt_idx}" "prompt_only" "prompt_only_oracle" 0 "$TARGET_PROMPT_TOTAL" "${ORACLE_PROMPT_PATH_ARRAY[$prompt_idx]}" "" "oracle_eval_judge"
done

for prompt_idx in "${!ORACLE_PROMPT_PATH_ARRAY[@]}"; do
  for ((det_shard_idx=0; det_shard_idx<DETERMINISTIC_SHARD_COUNT; det_shard_idx++)); do
    det_offset=$(( (det_shard_idx * TARGET_PROMPT_TOTAL) / DETERMINISTIC_SHARD_COUNT ))
    det_next_offset=$(( ((det_shard_idx + 1) * TARGET_PROMPT_TOTAL) / DETERMINISTIC_SHARD_COUNT ))
    det_limit=$(( det_next_offset - det_offset ))
    if (( det_limit <= 0 )); then
      continue
    fi

    det_dep="target_shard_A"
    if (( det_offset >= TARGET_PROMPT_SPLIT )); then
      if (( shard_b_limit <= 0 )); then
        die "Deterministic shard offset ${det_offset} requires target_shard_B, but target_shard_B is not configured."
      fi
      det_dep="target_shard_B"
    fi

    add_job "deterministic_shard_${det_shard_idx}_prompt_${prompt_idx}" "deterministic" "$FULL_DETERMINISTIC_PRESET" "$det_offset" "$det_limit" "${ORACLE_PROMPT_PATH_ARRAY[$prompt_idx]}" "$det_dep" "oracle_eval_judge"
  done
done

log "Parallel H200 run starting. run_label=$RUN_LABEL logs=$LOG_ROOT"
log "GPU pool: ${GPU_ID_ARRAY[*]}"
log "Target shard A offset=$shard_a_offset limit=$shard_a_limit"
if (( shard_b_limit > 0 )); then
  log "Target shard B offset=$shard_b_offset limit=$shard_b_limit"
else
  log "Target shard B omitted because limit=0"
fi
log "Full deterministic oracle preset: $FULL_DETERMINISTIC_PRESET"
log "Deterministic shard count: $DETERMINISTIC_SHARD_COUNT"
log "Oracle prompt paths: $ORACLE_PROMPTS_PATHS"
log_job_table

while true; do
  while true; do
    ready_job="$(next_ready_job || true)"
    free_gpu_slot="$(next_free_gpu_slot || true)"
    if [[ -z "$ready_job" || -z "$free_gpu_slot" ]]; then
      break
    fi
    launch_job "$ready_job" "$free_gpu_slot"
  done

  poll_running_jobs

  if all_jobs_terminal; then
    break
  fi

  if ! any_running_jobs; then
    ready_job="$(next_ready_job || true)"
    if [[ -n "$ready_job" ]]; then
      continue
    fi
    log "No running jobs and no ready jobs remain; marking unresolved pending jobs as blocked."
    for idx in "${!JOB_ID[@]}"; do
      if [[ "${JOB_STATE[$idx]}" == "pending" ]]; then
        JOB_STATE[$idx]="blocked"
        log "Job blocked: ${JOB_ID[$idx]} unresolved dependencies=${JOB_DEPENDS_ON[$idx]:-none}"
      fi
    done
    break
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    sleep 0.1
  else
    sleep "$SCHEDULER_POLL_SECONDS"
  fi
done

log_summary

failures=0
for idx in "${!JOB_ID[@]}"; do
  case "${JOB_STATE[$idx]}" in
    failed|blocked) failures=$((failures + 1)) ;;
  esac
done

if (( failures > 0 )); then
  die "$failures job(s) failed or were blocked. Check logs under $LOG_ROOT."
fi

log "All parallel H200 jobs completed successfully."
