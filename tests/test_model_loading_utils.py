from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import model_loading_utils as mlu
except Exception:
    mlu = None


@unittest.skipIf(mlu is None, "model_loading_utils dependencies unavailable")
class ModelLoadingUtilsTests(unittest.TestCase):
    def test_load_tokenizer_preserves_zero_pad_token_id(self):
        tokenizer = SimpleNamespace(padding_side="right", pad_token_id=0, eos_token_id=2)
        with patch("model_loading_utils.AutoTokenizer.from_pretrained", return_value=tokenizer):
            result = mlu.load_tokenizer("model")

        self.assertIs(result, tokenizer)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.pad_token_id, 0)

    def test_load_tokenizer_sets_missing_pad_token_to_eos(self):
        tokenizer = SimpleNamespace(padding_side="right", pad_token_id=None, eos_token_id=2)
        with patch("model_loading_utils.AutoTokenizer.from_pretrained", return_value=tokenizer):
            result = mlu.load_tokenizer("model")

        self.assertIs(result, tokenizer)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.pad_token_id, 2)


if __name__ == "__main__":
    unittest.main()
