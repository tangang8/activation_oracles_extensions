from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / "run_parallel_strongreject_v5.sh"


class RunParallelScriptTests(unittest.TestCase):
    def _write_fake_runner(self, tmpdir: str, mode: str) -> Path:
        fake = Path(tmpdir) / "fake_oracle_runner.sh"
        if mode == "success":
            body = (
                "#!/usr/bin/env bash\n"
                "echo \"FAKE_OK\"\n"
                "echo \"WANDB_RUN_NAME=${WANDB_RUN_NAME:-}\"\n"
                "echo \"WANDB_GROUP=${WANDB_GROUP:-}\"\n"
                "echo \"WANDB_JOB_TYPE=${WANDB_JOB_TYPE:-}\"\n"
                "echo \"TARGET_THINKING=${TARGET_THINKING:-}\"\n"
                "exit 0\n"
            )
        elif mode == "fail_non_oom":
            body = (
                "#!/usr/bin/env bash\n"
                "echo \"regular failure\" >&2\n"
                "exit 7\n"
            )
        elif mode == "oom_once_per_ladder":
            body = (
                "#!/usr/bin/env bash\n"
                "args=\"$*\"\n"
                "if [[ \"$args\" == *\"--target-judge-batch-size 64\"* || \"$args\" == *\"--oracle-judge-batch-size 64\"* ]]; then\n"
                "  echo \"CUDA out of memory\" >&2\n"
                "  exit 1\n"
                "fi\n"
                "echo \"FAKE_OK\"\n"
                "exit 0\n"
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        fake.write_text(body, encoding="utf-8")
        fake.chmod(0o755)
        return fake

    def test_dry_run_uses_run_label_in_log_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["RUN_LABEL"] = "unit_parallel_label"
            env["LOG_ROOT"] = str(Path(tmpdir) / "logs")
            env["DRY_RUN"] = "1"
            proc = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, env=env)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("run_label=unit_parallel_label", proc.stdout)
            self.assertIn("GPU pool: 0 1 2 3", proc.stdout)
            self.assertIn("Job table:", proc.stdout)
            self.assertIn("job=deterministic_shard_0_prompt_0", proc.stdout)
            self.assertIn("depends_on=target_shard_A", proc.stdout)
            self.assertIn("Summary done:", proc.stdout)
            self.assertIn("Summary failed: <none>", proc.stdout)

    def test_dry_run_supports_variable_gpu_counts(self) -> None:
        for gpu_ids in ("0", "0,1"):
            with self.subTest(gpu_ids=gpu_ids), tempfile.TemporaryDirectory() as tmpdir:
                env = os.environ.copy()
                env["RUN_LABEL"] = f"unit_parallel_gpus_{gpu_ids.replace(',', '_')}"
                env["LOG_ROOT"] = str(Path(tmpdir) / "logs")
                env["DRY_RUN"] = "1"
                env["GPU_IDS"] = gpu_ids
                env["TARGET_PROMPT_TOTAL"] = "2"
                env["TARGET_PROMPT_SPLIT"] = "1"
                env["NUM_ROLLOUTS"] = "1"
                env["NUM_ORACLE_ROLLOUTS"] = "1"
                proc = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, env=env)
                self.assertEqual(proc.returncode, 0)
                self.assertIn(f"GPU pool: {gpu_ids.replace(',', ' ')}", proc.stdout)
                self.assertIn("All parallel jobs completed successfully.", proc.stdout)
                self.assertEqual(proc.stdout.count(" type=deterministic preset="), 4)
                self.assertIn("job=deterministic_shard_4_prompt_0 type=", proc.stdout)
                self.assertIn("job=deterministic_shard_9_prompt_0 type=", proc.stdout)
                self.assertIn("job=deterministic_shard_4_prompt_1 type=", proc.stdout)
                self.assertIn("job=deterministic_shard_9_prompt_1 type=", proc.stdout)

    def test_non_oom_failure_fails_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_runner = self._write_fake_runner(tmpdir, "fail_non_oom")
            env = os.environ.copy()
            env["RUN_LABEL"] = "unit_parallel_fail"
            env["LOG_ROOT"] = str(Path(tmpdir) / "logs")
            env["TARGET_PROMPT_TOTAL"] = "1"
            env["TARGET_PROMPT_SPLIT"] = "1"
            env["NUM_ROLLOUTS"] = "1"
            env["NUM_ORACLE_ROLLOUTS"] = "1"
            env["RUN_ORACLE_EXPERIMENT"] = str(fake_runner)
            proc = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, env=env)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("Job failed", proc.stdout)
            self.assertIn("Job blocked", proc.stdout)
            self.assertIn("failed with exit code 7", proc.stdout)

    def test_oom_retries_are_per_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_runner = self._write_fake_runner(tmpdir, "oom_once_per_ladder")
            env = os.environ.copy()
            env["RUN_LABEL"] = "unit_parallel_oom"
            env["LOG_ROOT"] = str(Path(tmpdir) / "logs")
            env["TARGET_PROMPT_TOTAL"] = "1"
            env["TARGET_PROMPT_SPLIT"] = "1"
            env["NUM_ROLLOUTS"] = "1"
            env["NUM_ORACLE_ROLLOUTS"] = "1"
            env["RUN_ORACLE_EXPERIMENT"] = str(fake_runner)
            proc = subprocess.run([str(SCRIPT)], capture_output=True, text=True, check=False, env=env)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("hit OOM; retrying with next target judge batch size", proc.stdout)
            self.assertIn("hit OOM; retrying with next oracle eval/judge batch pair", proc.stdout)
            self.assertIn("Summary failed: <none>", proc.stdout)


if __name__ == "__main__":
    unittest.main()
