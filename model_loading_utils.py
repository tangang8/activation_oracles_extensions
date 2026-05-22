from dataclasses import dataclass

import pandas as pd
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class AdapterSpec:
    adapter_path: str
    adapter_name: str
    is_trainable: bool = False


def load_tokenizer(model_name: str, hf_token: str | None = None):
    kwargs = {"token": hf_token} if hf_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_causal_lm(
    model_name: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, str] | None = "auto",
    hf_token: str | None = None,
):
    kwargs = {"device_map": device_map, "torch_dtype": torch_dtype}
    if hf_token:
        kwargs["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model


def ensure_default_adapter(model):
    model.add_adapter(LoraConfig(), adapter_name="default")


def load_adapters(model, adapter_specs: list[AdapterSpec]):
    for spec in adapter_specs:
        print(f"Loading adapter '{spec.adapter_name}' from: {spec.adapter_path}")
        model.load_adapter(
            spec.adapter_path,
            adapter_name=spec.adapter_name,
            is_trainable=spec.is_trainable,
        )


def load_model_stack(
    model_name: str,
    adapter_specs: list[AdapterSpec] | None = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | dict[str, str] | None = "auto",
    hf_token: str | None = None,
):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = load_tokenizer(model_name, hf_token=hf_token)
    print(f"Loading model: {model_name}")
    model = load_causal_lm(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        hf_token=hf_token,
    )
    ensure_default_adapter(model)
    if adapter_specs:
        load_adapters(model, adapter_specs)
    print("Model loaded successfully!")
    return tokenizer, model


def get_adapter_config_df(model, adapter_name: str) -> pd.DataFrame:
    if adapter_name not in model.peft_config:
        available = ", ".join(model.peft_config.keys())
        raise KeyError(
            f"Adapter '{adapter_name}' not found in model.peft_config. "
            f"Available adapters: {available}"
        )
    config_dict = model.peft_config[adapter_name].to_dict()
    return pd.DataFrame(list(config_dict.items()), columns=["Parameter", "Value"])
