import os
from typing import Any


def init_wandb_run(config: dict[str, Any]) -> Any | None:
    try:
        import wandb  # type: ignore
    except Exception:
        print("[wandb] wandb not installed; skipping logging.")
        return None

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("[wandb] WANDB_API_KEY not set; skipping logging.")
        return None

    os.environ["WANDB_API_KEY"] = api_key
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT", "activation-oracles-extensions"),
        entity=os.getenv("WANDB_ENTITY"),
        name=os.getenv("WANDB_RUN_NAME"),
        config=config,
        reinit=True,
    )
    print(f"[wandb] logging to run: {run.name} ({run.id})")
    return run


def log_rollout_metrics(run: Any, rollout_entries: list[dict[str, Any]], compliance_results: dict[str, Any]) -> None:
    if run is None:
        return
    scores = [
        entry.get("compliance", {}).get("score")
        for entry in rollout_entries
        if entry.get("compliance", {}).get("score") is not None
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    payload = {
        "rollouts/valid_count": len(rollout_entries),
        "rollouts/compliance_rate": compliance_results.get("compliance_rate", 0.0),
        "rollouts/partial_compliance_rate": compliance_results.get("partial_compliance_rate", 0.0),
        "rollouts/avg_score": avg_score,
    }
    if rollout_entries:
        payload["rollouts/judge_instruction_file"] = rollout_entries[0].get("compliance", {}).get(
            "judge_instruction_file", ""
        )
    run.log(payload)


def log_oracle_metrics(run: Any, oracle_results: list[dict[str, Any]], elapsed_s: float) -> None:
    if run is None:
        return
    deterministic_schema = bool(oracle_results) and isinstance(oracle_results[0], dict) and "oracle_response" in oracle_results[0]
    if deterministic_schema:
        repeats_observed = [1 for _ in oracle_results]
    else:
        repeats_observed = [
            int(result.get("oracle_repeats", len(result.get("full_seq", [])) or 0))
            for result in oracle_results
        ]
    avg_repeats = (sum(repeats_observed) / len(repeats_observed)) if repeats_observed else 0.0
    run.log({
        "oracle/elapsed_seconds": elapsed_s,
        "oracle/targets_evaluated": len(oracle_results),
        "oracle/avg_repeats_observed": avg_repeats,
        "oracle/deterministic_cache_schema": 1.0 if deterministic_schema else 0.0,
    })


def log_timing_metrics(run: Any, timings: dict[str, float]) -> None:
    if run is None:
        return
    run.log({f"timing/{name}": value for name, value in timings.items()})


def log_oracle_judge_metrics(run: Any, summary: dict[str, Any]) -> None:
    if run is None:
        return
    payload: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            payload[str(key)] = float(value)
    if payload:
        run.log(payload)
