from __future__ import annotations

import subprocess
import tempfile
import unittest
import os
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "run_oracle_experiment.sh"


class RunScriptTests(unittest.TestCase):
    def _run_with_fake_python(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            shim = Path(tmpdir) / "python"
            shim.write_text(
                "#!/usr/bin/env bash\n"
                "echo \"FAKE_PYTHON\"\n"
                "echo \"MODEL_NAME=${MODEL_NAME:-}\"\n"
                "echo \"ORACLE_ROLLOUT_MODE=${ORACLE_ROLLOUT_MODE:-}\"\n"
                "echo \"K_ROLLOUTS=${K_ROLLOUTS:-}\"\n"
                "echo \"NUM_ORACLE_ROLLOUTS=${NUM_ORACLE_ROLLOUTS:-}\"\n"
                "echo \"TARGET_PROMPT_OFFSET=${TARGET_PROMPT_OFFSET:-}\"\n"
                "echo \"EXPERIMENT_PRESET=${EXPERIMENT_PRESET:-}\"\n"
                "echo \"RUN_TARGET_ROLLOUTS=${RUN_TARGET_ROLLOUTS:-}\"\n"
                "echo \"RUN_TARGET_JUDGING=${RUN_TARGET_JUDGING:-}\"\n"
                "echo \"RUN_ORACLE_ROLLOUTS=${RUN_ORACLE_ROLLOUTS:-}\"\n"
                "echo \"RUN_ORACLE_JUDGING=${RUN_ORACLE_JUDGING:-}\"\n"
                "echo \"TARGET_LORA_PATH=${TARGET_LORA_PATH:-}\"\n"
                "echo \"JUDGE_LORA_PATH=${JUDGE_LORA_PATH:-}\"\n"
                "echo \"ORACLE_LORA_PATH=${ORACLE_LORA_PATH:-}\"\n"
                "echo \"JUDGE_THINKING=${JUDGE_THINKING:-}\"\n"
                "echo \"TARGET_JUDGE_BATCH_SIZE=${TARGET_JUDGE_BATCH_SIZE:-}\"\n"
                "echo \"ORACLE_INPUT_TYPES=${ORACLE_INPUT_TYPES:-}\"\n"
                "echo \"ORACLE_TOKEN_POINT_FILTER=${ORACLE_TOKEN_POINT_FILTER:-}\"\n"
                "echo \"ORACLE_ADAPTER_PATH=${ORACLE_ADAPTER_PATH:-}\"\n"
                "echo \"ORACLE_ADAPTER_NAME=${ORACLE_ADAPTER_NAME:-}\"\n"
                "echo \"ORACLE_PROMPTS_PATH=${ORACLE_PROMPTS_PATH:-}\"\n"
                "echo \"JUDGE_INSTRUCTION_PATH=${JUDGE_INSTRUCTION_PATH:-}\"\n",
                encoding="utf-8",
            )
            shim.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{tmpdir}:{env.get('PATH', '')}"
            return subprocess.run([str(SCRIPT), *args], capture_output=True, text=True, check=False, env=env)

    def test_help(self) -> None:
        proc = subprocess.run([str(SCRIPT), "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--mode MODE", proc.stdout)
        self.assertIn("--model-name NAME", proc.stdout)
        self.assertIn("--preset NAME", proc.stdout)
        self.assertIn("--oracle-adapter-path PATH", proc.stdout)
        self.assertIn("--oracle-adapter-name NAME", proc.stdout)
        self.assertIn("--target-judge-batch-size N", proc.stdout)
        self.assertIn("--target-prompt-offset N", proc.stdout)
        self.assertIn("--oracle-input-types CSV", proc.stdout)
        self.assertIn("--oracle-token-point-filter F", proc.stdout)
        self.assertIn("--wandb on|off", proc.stdout)
        self.assertIn("full_deterministic_oracle", proc.stdout)
        self.assertIn("rollout_post_prompt_oracle", proc.stdout)
        self.assertIn("sampled_target_repeats", proc.stdout)
        self.assertIn("prompt_only_oracle", proc.stdout)
        self.assertIn("target_judging_only", proc.stdout)
        self.assertIn("--judge-instruction-path PATH", proc.stdout)
        self.assertIn("--oracle-prompts-path PATH", proc.stdout)
        self.assertIn("--judge-thinking MODE", proc.stdout)

    def test_invalid_mode(self) -> None:
        proc = subprocess.run([str(SCRIPT), "--mode", "bad"], capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --mode", proc.stderr)

    def test_invalid_wandb(self) -> None:
        proc = subprocess.run([str(SCRIPT), "--wandb", "maybe"], capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --wandb setting", proc.stderr)

    def test_invalid_set(self) -> None:
        proc = subprocess.run([str(SCRIPT), "--set", "NOTKV"], capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --set value", proc.stderr)

    def test_invalid_preset(self) -> None:
        proc = subprocess.run([str(SCRIPT), "--preset", "bad"], capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --preset", proc.stderr)

    def test_invalid_stage_bool_flag(self) -> None:
        proc = subprocess.run(
            [str(SCRIPT), "--run-oracle-rollouts", "maybe"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --run-oracle-rollouts setting", proc.stderr)

    def test_invalid_judge_thinking(self) -> None:
        proc = subprocess.run([str(SCRIPT), "--judge-thinking", "maybe"], capture_output=True, text=True, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --judge-thinking setting", proc.stderr)

    def test_invalid_oracle_token_point_filter(self) -> None:
        proc = subprocess.run(
            [str(SCRIPT), "--oracle-token-point-filter", "before_prompt"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Invalid --oracle-token-point-filter setting", proc.stderr)

    def test_preset_oracle_target_control_exports_expected_flags(self) -> None:
        proc = self._run_with_fake_python("--preset", "oracle_target_control", "--mode", "all_target_deterministic")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("EXPERIMENT_PRESET=oracle_target_control", proc.stdout)
        self.assertIn("RUN_TARGET_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_TARGET_JUDGING=true", proc.stdout)
        self.assertIn("RUN_ORACLE_ROLLOUTS=false", proc.stdout)
        self.assertIn("RUN_ORACLE_JUDGING=false", proc.stdout)
        self.assertIn("ORACLE_PROMPTS_PATH=prompts/oracle_prompts/default_oracle_prompts.json", proc.stdout)
        self.assertIn("JUDGE_INSTRUCTION_PATH=strongReject_v5.jinja2", proc.stdout)

    def test_preset_prompt_only_oracle_exports_expected_flags(self) -> None:
        proc = self._run_with_fake_python("--preset", "prompt_only_oracle")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("ORACLE_ROLLOUT_MODE=prompt_only_repeats", proc.stdout)
        self.assertIn("RUN_TARGET_ROLLOUTS=false", proc.stdout)
        self.assertIn("RUN_TARGET_JUDGING=false", proc.stdout)
        self.assertIn("RUN_ORACLE_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_ORACLE_JUDGING=true", proc.stdout)
        self.assertIn("JUDGE_THINKING=off", proc.stdout)
        self.assertIn("TARGET_JUDGE_BATCH_SIZE=16", proc.stdout)

    def test_oracle_probe_overrides_export_expected_env(self) -> None:
        proc = self._run_with_fake_python(
            "--preset",
            "full_deterministic_oracle",
            "--target-prompt-offset",
            "50",
            "--target-prompt-limit",
            "50",
            "--oracle-input-types",
            "rollout_segment,token_points",
            "--oracle-token-point-filter",
            "post_prompt",
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("TARGET_PROMPT_OFFSET=50", proc.stdout)
        self.assertIn("ORACLE_INPUT_TYPES=rollout_segment,token_points", proc.stdout)
        self.assertIn("ORACLE_TOKEN_POINT_FILTER=post_prompt", proc.stdout)

    def test_prompt_only_allows_raw_oracle_input_export_for_python_validation(self) -> None:
        proc = self._run_with_fake_python(
            "--preset",
            "prompt_only_oracle",
            "--oracle-input-types",
            "rollout_segment,token_points",
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("ORACLE_INPUT_TYPES=rollout_segment,token_points", proc.stdout)

    def test_preset_sampled_target_repeats_sets_sampled_mode(self) -> None:
        proc = self._run_with_fake_python("--preset", "sampled_target_repeats")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("ORACLE_ROLLOUT_MODE=sampled_target_repeats", proc.stdout)
        self.assertIn("RUN_TARGET_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_ORACLE_ROLLOUTS=true", proc.stdout)

    def test_preset_rollout_post_prompt_oracle_sets_probe_variant(self) -> None:
        proc = self._run_with_fake_python("--preset", "rollout_post_prompt_oracle")
        self.assertEqual(proc.returncode, 0)
        self.assertIn("ORACLE_ROLLOUT_MODE=all_target_deterministic", proc.stdout)
        self.assertIn("RUN_TARGET_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_TARGET_JUDGING=true", proc.stdout)
        self.assertIn("RUN_ORACLE_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_ORACLE_JUDGING=true", proc.stdout)
        self.assertIn("ORACLE_INPUT_TYPES=rollout_segment,token_points", proc.stdout)
        self.assertIn("ORACLE_TOKEN_POINT_FILTER=post_prompt", proc.stdout)

    def test_explicit_stage_flags_override_defaults(self) -> None:
        proc = self._run_with_fake_python(
            "--preset",
            "prompt_only_oracle",
            "--mode",
            "prompt_only_repeats",
            "--model-name",
            "Qwen/Qwen3-8B",
            "--run-target-rollouts",
            "true",
            "--run-target-judging",
            "true",
            "--run-oracle-rollouts",
            "true",
            "--run-oracle-judging",
            "true",
            "--target-lora-path",
            "oracle",
            "--judge-lora-path",
            "default",
            "--oracle-lora-path",
            "oracle",
            "--oracle-adapter-path",
            "myorg/checkpoints/custom-adapter",
            "--oracle-adapter-name",
            "custom_oracle",
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("MODEL_NAME=Qwen/Qwen3-8B", proc.stdout)
        self.assertIn("RUN_TARGET_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_TARGET_JUDGING=true", proc.stdout)
        self.assertIn("RUN_ORACLE_ROLLOUTS=true", proc.stdout)
        self.assertIn("RUN_ORACLE_JUDGING=true", proc.stdout)
        self.assertIn("TARGET_LORA_PATH=oracle", proc.stdout)
        self.assertIn("ORACLE_ADAPTER_PATH=myorg/checkpoints/custom-adapter", proc.stdout)
        self.assertIn("ORACLE_ADAPTER_NAME=custom_oracle", proc.stdout)


if __name__ == "__main__":
    unittest.main()
