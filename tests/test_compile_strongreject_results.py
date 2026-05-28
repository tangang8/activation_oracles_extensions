from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cache_utils import deterministic_oracle_judge_cache_file_path, judge_cache_file_path
from compile_strongreject_results import (
    ROLLOUT_POST_PROMPT_VARIANT,
    StrongRejectCompileConfig,
    compile_strongreject_results,
)


class CompileStrongRejectResultsTests(unittest.TestCase):
    def _cfg(self, root: Path) -> StrongRejectCompileConfig:
        return StrongRejectCompileConfig(
            cache_root=root / "cache",
            output_dir=root / "compiled",
            target_model_name="ModelA",
            judge_model_name="JudgeA",
            oracle_model_name="OracleA",
            oracle_lora_path="oracle",
            expected_target_prompts=1,
            expected_target_rollouts=2,
            expected_oracle_rollouts=2,
            oracle_prompts_paths=("oracle_a.json",),
            thresholds=(0.2, 0.5, 0.8, 1.0),
        )

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _target_path(
        self,
        cfg: StrongRejectCompileConfig,
        prompt: str,
        *,
        target_lora_path: str,
        target_thinking_mode: str,
    ) -> Path:
        return judge_cache_file_path(
            cache_root=str(cfg.cache_root),
            target_model_name=cfg.target_model_name,
            target_lora_path=target_lora_path,
            judge_model_name=cfg.judge_model_name,
            judge_lora_path="default",
            generation_kwargs={"temperature": 0.0},
            judge_thinking_mode="off",
            target_thinking_mode=target_thinking_mode,
            judge_instruction_stem=cfg.judge_instruction_stem,
            user_prompt=prompt,
        )

    def _oracle_path(
        self,
        cfg: StrongRejectCompileConfig,
        prompt: str,
        oracle_prompt: str,
        *,
        dir_base: str,
        temp: float,
        variant_key: str | None = None,
    ) -> Path:
        return deterministic_oracle_judge_cache_file_path(
            cache_root=str(cfg.cache_root),
            target_model_name=cfg.target_model_name,
            target_lora_path="default",
            judge_model_name=cfg.judge_model_name,
            judge_lora_path="default",
            judge_generation_kwargs={"temperature": 0.0},
            judge_thinking_mode="off",
            judge_instruction_stem=cfg.judge_instruction_stem,
            oracle_model_name=cfg.oracle_model_name,
            oracle_lora_path=cfg.oracle_lora_path,
            oracle_generation_kwargs={"temperature": temp},
            target_prompt=prompt,
            oracle_prompt=oracle_prompt,
            oracle_rollouts_dir_base=dir_base,
            cache_variant_key=variant_key,
        )

    def test_workflow_traced_compilation_and_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root)
            target_prompt = "harmful prompt"
            oracle_prompt = "oracle prompt"

            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="default", target_thinking_mode="default"),
                [
                    {"rollout_index": 0, "compliance": {"score": 0.0, "score_scale": "strongreject_0_1"}},
                    {"rollout_index": 1, "compliance": {"score": 1.0, "score_scale": "strongreject_0_1"}},
                ],
            )
            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="oracle", target_thinking_mode="off"),
                [
                    {"rollout_index": 0, "compliance": {"score": 0.25, "score_scale": "strongreject_0_1"}},
                    {"rollout_index": 1, "compliance": {"score": 0.75, "score_scale": "strongreject_0_1"}},
                ],
            )
            self._write_json(
                self._oracle_path(
                    cfg,
                    target_prompt,
                    oracle_prompt,
                    dir_base="oracle_prompt_rollouts",
                    temp=1.0,
                ),
                [
                    {
                        "oracle_rollout_index": 0,
                        "compliance": {
                            "full_seq": {"score": 1.0, "score_scale": "strongreject_0_1"},
                            "token_points": {"last_prompt_token": {"score": 0.4, "score_scale": "strongreject_0_1"}},
                        },
                    },
                    {
                        "oracle_rollout_index": 1,
                        "compliance": {
                            "full_seq": {"score": 0.0, "score_scale": "strongreject_0_1"},
                            "token_points": {"last_prompt_token": {"score": 0.6, "score_scale": "strongreject_0_1"}},
                        },
                    },
                ],
            )
            self._write_json(
                self._oracle_path(
                    cfg,
                    target_prompt,
                    oracle_prompt,
                    dir_base="oracle_rollouts",
                    temp=0.0,
                    variant_key=ROLLOUT_POST_PROMPT_VARIANT,
                ),
                [
                    {
                        "rollout_index": 0,
                        "target_rollout_index": 0,
                        "oracle_rollout_index": 0,
                        "compliance": {
                            "rollout_segment": {"score": 0.8, "score_scale": "strongreject_0_1"},
                            "token_points": {"first_rollout_token": {"score": 0.1, "score_scale": "strongreject_0_1"}},
                        },
                    },
                    {
                        "rollout_index": 1,
                        "target_rollout_index": 1,
                        "oracle_rollout_index": 0,
                        "compliance": {
                            "rollout_segment": {"score": 0.2, "score_scale": "strongreject_0_1"},
                            "token_points": {"first_rollout_token": {"score": 0.3, "score_scale": "strongreject_0_1"}},
                        },
                    }
                ],
            )

            unrelated = cfg.cache_root / "target_ModelA" / "judge_JudgeA_temp-0.0" / "other" / "target_rollouts_judged" / "x.json"
            self._write_json(unrelated, [{"rollout_index": 0, "compliance": {"score": 1.0}}])

            manifest = compile_strongreject_results(
                cfg,
                target_prompts=[target_prompt],
                oracle_prompts_by_file={"oracle_a.json": [oracle_prompt]},
            )

            self.assertEqual(manifest["missing_files"], [])
            self.assertEqual(manifest["loaded_files"]["target_baseline"], 1)
            self.assertEqual(manifest["loaded_files"]["oracle_rollout_control"], 1)
            self.assertEqual(manifest["loaded_files"]["user_prompt_oracle"], 1)
            self.assertEqual(manifest["loaded_files"]["target_rollout_oracle"], 1)

            with (cfg.output_dir / "strongreject_summary.csv").open("r", encoding="utf-8") as f:
                summary = list(csv.DictReader(f))
            baseline = [row for row in summary if row["condition"] == "target_baseline"][0]
            self.assertEqual(baseline["mean_score"], "0.5")
            self.assertEqual(baseline["asr_1"], "0.5")

            rollout = [
                row
                for row in summary
                if row["condition"] == "target_rollout_oracle" and row["probe_name"] == "rollout_segment"
            ][0]
            self.assertEqual(rollout["mean_score"], "0.5")
            self.assertEqual(rollout["asr_0_8"], "0.5")

            with (cfg.output_dir / "strongreject_prompt_level.csv").open("r", encoding="utf-8") as f:
                prompt_level = list(csv.DictReader(f))
            rollout_prompt = [
                row
                for row in prompt_level
                if row["condition"] == "target_rollout_oracle" and row["probe_name"] == "rollout_segment"
            ][0]
            self.assertEqual(rollout_prompt["n_scored"], "2")
            self.assertNotEqual(rollout_prompt["sd_within_prompt_target_rollouts"], "")

            self.assertFalse(
                any(
                    warning.get("condition") == "target_rollout_oracle"
                    and warning.get("reason") == "more than one target_rollout_index found"
                    for warning in manifest.get("coverage_warnings", [])
                )
            )

            with (cfg.output_dir / "strongreject_details.csv").open("r", encoding="utf-8") as f:
                details = list(csv.DictReader(f))
            self.assertFalse(any("other" in row["cache_path"] for row in details))

    def test_missing_and_invalid_scores_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root)
            target_prompt = "harmful prompt"
            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="default", target_thinking_mode="default"),
                [
                    {"rollout_index": 0, "compliance": {"score": 2.0, "score_scale": "strongreject_0_1"}},
                    {"rollout_index": 1, "compliance": {"score": 0.5, "score_scale": "legacy_1_5"}},
                ],
            )

            manifest = compile_strongreject_results(
                cfg,
                target_prompts=[target_prompt],
                oracle_prompts_by_file={"oracle_a.json": ["oracle prompt"]},
            )
            self.assertGreater(len(manifest["missing_files"]), 0)
            self.assertEqual(manifest["detail_row_count"], 0)
            self.assertEqual(len(manifest["skipped_score_leaves"]), 2)

    def test_strict_raises_on_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = StrongRejectCompileConfig(
                cache_root=root / "cache",
                output_dir=root / "compiled",
                expected_target_prompts=1,
                expected_target_rollouts=1,
                expected_oracle_rollouts=1,
                oracle_prompts_paths=("oracle_a.json",),
                strict=True,
            )
            with self.assertRaises(RuntimeError):
                compile_strongreject_results(
                    cfg,
                    target_prompts=["prompt"],
                    oracle_prompts_by_file={"oracle_a.json": ["oracle"]},
                )

    def test_target_path_selection_requires_target_thinking_suffix_for_control(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root)
            target_prompt = "harmful prompt"

            # Baseline at expected path.
            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="default", target_thinking_mode="default"),
                [{"rollout_index": 0, "compliance": {"score": 0.1, "score_scale": "strongreject_0_1"}}],
            )
            # Oracle control written to a near-miss path (wrong thinking mode in path).
            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="oracle", target_thinking_mode="default"),
                [{"rollout_index": 0, "compliance": {"score": 0.9, "score_scale": "strongreject_0_1"}}],
            )

            manifest = compile_strongreject_results(
                cfg,
                target_prompts=[target_prompt],
                oracle_prompts_by_file={"oracle_a.json": ["oracle prompt"]},
            )
            self.assertEqual(manifest["loaded_files"]["target_baseline"], 1)
            self.assertEqual(manifest["loaded_files"].get("oracle_rollout_control", 0), 0)
            self.assertGreater(len(manifest["missing_files"]), 0)

    def test_oracle_path_selection_requires_exact_mode_and_variant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root)
            target_prompt = "harmful prompt"
            oracle_prompt = "oracle prompt"

            # Provide required target files so failure is only on oracle path matching.
            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="default", target_thinking_mode="default"),
                [{"rollout_index": 0, "compliance": {"score": 0.1, "score_scale": "strongreject_0_1"}}],
            )
            self._write_json(
                self._target_path(cfg, target_prompt, target_lora_path="oracle", target_thinking_mode="off"),
                [{"rollout_index": 0, "compliance": {"score": 0.1, "score_scale": "strongreject_0_1"}}],
            )

            # Near miss 1: user-prompt oracle written to oracle_rollouts dir instead of oracle_prompt_rollouts.
            self._write_json(
                self._oracle_path(
                    cfg,
                    target_prompt,
                    oracle_prompt,
                    dir_base="oracle_rollouts",
                    temp=1.0,
                ),
                [{"oracle_rollout_index": 0, "compliance": {"full_seq": {"score": 0.4, "score_scale": "strongreject_0_1"}}}],
            )

            # Near miss 2: target-rollout oracle written without required rollout-post-prompt variant key.
            self._write_json(
                self._oracle_path(
                    cfg,
                    target_prompt,
                    oracle_prompt,
                    dir_base="oracle_rollouts",
                    temp=0.0,
                    variant_key=None,
                ),
                [{"rollout_index": 0, "compliance": {"rollout_segment": {"score": 0.8, "score_scale": "strongreject_0_1"}}}],
            )
            # Near miss 3: the abandoned one-rollout variant should not be treated as
            # the corrected all-target-rollouts deterministic result.
            one_rollout_variant = json.dumps(
                {
                    "k_rollouts": 1,
                    "oracle_input_types": ["rollout_segment", "token_points"],
                    "oracle_token_point_filter": "post_prompt",
                },
                sort_keys=True,
                ensure_ascii=True,
            )
            self._write_json(
                self._oracle_path(
                    cfg,
                    target_prompt,
                    oracle_prompt,
                    dir_base="oracle_rollouts",
                    temp=0.0,
                    variant_key=one_rollout_variant,
                ),
                [{"rollout_index": 0, "compliance": {"rollout_segment": {"score": 0.9, "score_scale": "strongreject_0_1"}}}],
            )

            manifest = compile_strongreject_results(
                cfg,
                target_prompts=[target_prompt],
                oracle_prompts_by_file={"oracle_a.json": [oracle_prompt]},
            )
            self.assertEqual(manifest["loaded_files"].get("user_prompt_oracle", 0), 0)
            self.assertEqual(manifest["loaded_files"].get("target_rollout_oracle", 0), 0)
            self.assertGreaterEqual(len(manifest["missing_files"]), 2)

    def test_manifest_warns_when_actual_target_prompt_count_differs_from_expected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = self._cfg(root)
            # expected_target_prompts stays 1 from _cfg; pass two prompts to force mismatch.
            prompts = ["p0", "p1"]
            manifest = compile_strongreject_results(
                cfg,
                target_prompts=prompts,
                oracle_prompts_by_file={"oracle_a.json": ["oracle"]},
            )
            self.assertEqual(manifest["actual_target_prompts"], 2)
            self.assertEqual(manifest["expected_target_prompts"], 1)
            warnings = manifest.get("coverage_warnings", [])
            self.assertTrue(
                any(
                    warning.get("condition") == "target_prompt_set"
                    and warning.get("actual_target_prompts") == 2
                    and warning.get("expected_target_prompts") == 1
                    for warning in warnings
                )
            )


if __name__ == "__main__":
    unittest.main()
