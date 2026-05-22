from __future__ import annotations

from typing import Any, Callable

from transformers import AutoTokenizer


def _find_last_subsequence_start(token_ids: list[int], pattern_ids: list[int]) -> int | None:
    if not pattern_ids or len(pattern_ids) > len(token_ids):
        return None
    width = len(pattern_ids)
    for i in range(len(token_ids) - width, -1, -1):
        if token_ids[i : i + width] == pattern_ids:
            return i
    return None


def _tokenize_ids(tokenizer: AutoTokenizer, text: str) -> list[int]:
    return tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()


def _validate_prompt_response_boundary(prompt_ids: list[int], combined_ids: list[int]) -> None:
    prompt_len = len(prompt_ids)
    if combined_ids[:prompt_len] != prompt_ids:
        raise ValueError(
            "Tokenized prompt/response boundary is unstable: tokenizing "
            "formatted_target_prompt + target_response does not preserve "
            "formatted_target_prompt as an exact token prefix. Cannot safely "
            f"compute prompt/rollout token boundaries (prompt_len={prompt_len}, "
            f"combined_len={len(combined_ids)})."
        )


def build_combined_points_spec(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    target_response: str,
    token_points: dict[str, int] | None = None,
) -> dict[str, Any]:
    prompt_ids = _tokenize_ids(tokenizer, formatted_target_prompt)
    combined_text = formatted_target_prompt + target_response
    combined_ids = _tokenize_ids(tokenizer, combined_text)
    _validate_prompt_response_boundary(prompt_ids, combined_ids)

    prompt_len = len(prompt_ids)
    combined_len = len(combined_ids)
    rollout_len = combined_len - prompt_len
    if rollout_len <= 0:
        raise ValueError("Combined sequence has no rollout tokens to probe.")

    points = token_points or {}
    return {
        "combined_text": combined_text,
        "prompt_len": prompt_len,
        "combined_len": combined_len,
        "rollout_len": rollout_len,
        "prompt_segment": (0, prompt_len),
        "rollout_segment": (prompt_len, combined_len),
        "token_points": points,
        "token_point_indices": sorted(set(points.values())),
    }


def build_prompt_only_points_spec(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    token_points: dict[str, int] | None = None,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        formatted_target_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])
    if prompt_len <= 0:
        raise ValueError("Prompt has no tokens to probe.")

    points = token_points or {}
    return {
        "combined_text": formatted_target_prompt,
        "prompt_len": prompt_len,
        "combined_len": prompt_len,
        "rollout_len": 0,
        "prompt_segment": (0, prompt_len),
        "rollout_segment": (prompt_len, prompt_len),
        "token_points": points,
        "token_point_indices": sorted(set(points.values())),
    }


def extract_token_points_combined_default(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    target_response: str,
) -> dict[str, Any]:
    prompt_ids = _tokenize_ids(tokenizer, formatted_target_prompt)
    combined_ids = _tokenize_ids(tokenizer, formatted_target_prompt + target_response)
    _validate_prompt_response_boundary(prompt_ids, combined_ids)

    prompt_len = len(prompt_ids)
    combined_len = len(combined_ids)
    token_points = {
        "last_prompt_token": prompt_len - 1,
        "first_rollout_token": prompt_len,
        "last_rollout_token": combined_len - 1,
    }
    return build_combined_points_spec(
        tokenizer=tokenizer,
        formatted_target_prompt=formatted_target_prompt,
        target_response=target_response,
        token_points=token_points,
    )


def extract_token_points_combined_qwen(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
    target_response: str,
) -> dict[str, Any]:
    prompt_ids = _tokenize_ids(tokenizer, formatted_target_prompt)
    combined_text = formatted_target_prompt + target_response
    combined_ids = _tokenize_ids(tokenizer, combined_text)
    _validate_prompt_response_boundary(prompt_ids, combined_ids)

    prompt_len = len(prompt_ids)
    combined_len = len(combined_ids)

    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)

    im_end_start = _find_last_subsequence_start(prompt_ids, im_end_ids)
    if im_end_start is None:
        raise ValueError("Could not find <|im_end|> token in formatted prompt.")
    im_end_after = im_end_start + len(im_end_ids)
    if im_end_start - 1 < 0 or im_end_after >= prompt_len:
        raise ValueError("Prompt too short to index around <|im_end|> token.")

    im_start_start = _find_last_subsequence_start(prompt_ids, im_start_ids)
    if im_start_start is None:
        raise ValueError("Could not find trailing <|im_start|> token in formatted prompt.")

    assistant_start = _find_last_subsequence_start(prompt_ids, assistant_ids)
    if assistant_start is None:
        raise ValueError("Could not find trailing assistant token in formatted prompt.")

    rollout_ids = combined_ids[prompt_len:]
    think_close_start_rollout = _find_last_subsequence_start(rollout_ids, think_close_ids)
    if think_close_start_rollout is None:
        raise ValueError("Could not find </think> token in rollout.")
    think_close_start = prompt_len + think_close_start_rollout
    first_after_think_close = think_close_start + len(think_close_ids)
    if first_after_think_close >= combined_len:
        raise ValueError("No token found after </think> in rollout.")

    token_points = {
        "im_end_token": im_end_start,
        "token_before_im_end": im_end_start - 1,
        "token_after_im_end": im_end_after,
        "trailing_im_start_token": im_start_start,
        "trailing_assistant_token": assistant_start,
        "last_prompt_token": prompt_len - 1,
        "first_rollout_token": prompt_len,
        "think_close_token": think_close_start,
        "first_token_after_think_close": first_after_think_close,
        "last_rollout_token": combined_len - 1,
    }
    return build_combined_points_spec(
        tokenizer=tokenizer,
        formatted_target_prompt=formatted_target_prompt,
        target_response=target_response,
        token_points=token_points,
    )


def extract_token_points_prompt_default(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        formatted_target_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])
    token_points = {"last_prompt_token": prompt_len - 1}
    return build_prompt_only_points_spec(
        tokenizer=tokenizer,
        formatted_target_prompt=formatted_target_prompt,
        token_points=token_points,
    )


def extract_token_points_prompt_qwen(
    tokenizer: AutoTokenizer,
    formatted_target_prompt: str,
) -> dict[str, Any]:
    prompt_ids = tokenizer(
        formatted_target_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0].tolist()
    prompt_len = len(prompt_ids)

    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)

    im_end_start = _find_last_subsequence_start(prompt_ids, im_end_ids)
    if im_end_start is None:
        raise ValueError("Could not find <|im_end|> token in formatted prompt.")
    im_end_after = im_end_start + len(im_end_ids)
    if im_end_start - 1 < 0 or im_end_after >= prompt_len:
        raise ValueError("Prompt too short to index around <|im_end|> token.")

    im_start_start = _find_last_subsequence_start(prompt_ids, im_start_ids)
    if im_start_start is None:
        raise ValueError("Could not find trailing <|im_start|> token in formatted prompt.")

    assistant_start = _find_last_subsequence_start(prompt_ids, assistant_ids)
    if assistant_start is None:
        raise ValueError("Could not find trailing assistant token in formatted prompt.")

    token_points = {
        "im_end_token": im_end_start,
        "token_before_im_end": im_end_start - 1,
        "token_after_im_end": im_end_after,
        "trailing_im_start_token": im_start_start,
        "trailing_assistant_token": assistant_start,
        "last_prompt_token": prompt_len - 1,
    }
    return build_prompt_only_points_spec(
        tokenizer=tokenizer,
        formatted_target_prompt=formatted_target_prompt,
        token_points=token_points,
    )


CombinedTokenPointExtractor = Callable[[AutoTokenizer, str, str], dict[str, Any]]
PromptTokenPointExtractor = Callable[[AutoTokenizer, str], dict[str, Any]]


COMBINED_TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME: dict[str, CombinedTokenPointExtractor] = {
    "Qwen/Qwen3-8B": extract_token_points_combined_qwen,
}

PROMPT_TOKEN_POINT_EXTRACTORS_BY_MODEL_NAME: dict[str, PromptTokenPointExtractor] = {
    "Qwen/Qwen3-8B": extract_token_points_prompt_qwen,
}
