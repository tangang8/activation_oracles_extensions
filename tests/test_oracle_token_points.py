from __future__ import annotations

import unittest

try:
    import oracle_token_points as otp
except Exception:
    otp = None


class _FakeTensor:
    def __init__(self, values):
        self._values = list(values)
        self.shape = (len(self._values),)

    def tolist(self):
        return list(self._values)


class _FakeTokenizer:
    def __call__(self, text, return_tensors, add_special_tokens):
        del return_tensors, add_special_tokens
        if text == "prompt":
            ids = [11, 12, 13]
        elif text == "promptrollout":
            ids = [11, 12, 13, 21, 22]
        elif text == "qwen_prompt":
            ids = [31, 10, 41, 20, 30, 99]
        elif text == "qwen_promptqwen_response":
            ids = [31, 10, 41, 20, 30, 99, 50, 77, 88]
        else:
            ids = [31, 10, 41, 20, 30, 99]
        return {"input_ids": [_FakeTensor(ids)]}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "<|im_end|>": [10],
            "<|im_start|>": [20],
            "assistant": [30],
            "</think>": [50],
        }
        return mapping[text]


class _BoundaryUnstableTokenizer(_FakeTokenizer):
    def __call__(self, text, return_tensors, add_special_tokens):
        if text == "promptrollout":
            return {"input_ids": [_FakeTensor([11, 12, 99, 21, 22])]}
        if text == "qwen_promptqwen_response":
            return {"input_ids": [_FakeTensor([31, 10, 41, 20, 99, 50, 77, 88])]}
        return super().__call__(text, return_tensors, add_special_tokens)


@unittest.skipIf(otp is None, "oracle_token_points dependencies unavailable")
class OracleTokenPointsTests(unittest.TestCase):
    def test_preview_combined_default(self):
        tok = _FakeTokenizer()
        spec = otp.extract_token_points_combined_default(tok, "prompt", "rollout")
        self.assertEqual(spec["prompt_segment"], (0, 3))
        self.assertEqual(spec["rollout_segment"], (3, 5))
        self.assertEqual(set(spec["token_points"].keys()), {"last_prompt_token", "first_rollout_token", "last_rollout_token"})

    def test_combined_default_rejects_unstable_boundary(self):
        tok = _BoundaryUnstableTokenizer()
        with self.assertRaisesRegex(ValueError, "prompt/response boundary is unstable"):
            otp.extract_token_points_combined_default(tok, "prompt", "rollout")

    def test_combined_spec_rejects_unstable_boundary(self):
        tok = _BoundaryUnstableTokenizer()
        with self.assertRaisesRegex(ValueError, "prompt/response boundary is unstable"):
            otp.build_combined_points_spec(tok, "prompt", "rollout")

    def test_combined_qwen_rejects_unstable_boundary(self):
        tok = _BoundaryUnstableTokenizer()
        with self.assertRaisesRegex(ValueError, "prompt/response boundary is unstable"):
            otp.extract_token_points_combined_qwen(tok, "qwen_prompt", "qwen_response")

    def test_preview_combined_qwen(self):
        tok = _FakeTokenizer()
        spec = otp.extract_token_points_combined_qwen(tok, "qwen_prompt", "qwen_response")
        self.assertEqual(spec["prompt_segment"], (0, 6))
        self.assertEqual(spec["rollout_segment"], (6, 9))
        self.assertEqual(spec["token_points"]["first_rollout_token"], 6)
        self.assertEqual(spec["token_points"]["first_token_after_think_close"], 7)

    def test_prompt_only_default(self):
        tok = _FakeTokenizer()
        spec = otp.extract_token_points_prompt_default(tok, "prompt")
        self.assertEqual(spec["rollout_len"], 0)
        self.assertEqual(spec["token_points"]["last_prompt_token"], 2)

    def test_prompt_only_qwen_points(self):
        tok = _FakeTokenizer()
        spec = otp.extract_token_points_prompt_qwen(tok, "whatever")
        self.assertEqual(
            set(spec["token_points"].keys()),
            {
                "im_end_token",
                "token_before_im_end",
                "token_after_im_end",
                "trailing_im_start_token",
                "trailing_assistant_token",
                "last_prompt_token",
            },
        )


if __name__ == "__main__":
    unittest.main()
