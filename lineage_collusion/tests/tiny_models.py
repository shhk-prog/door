"""
テスト用の極小ランダム初期化モデル（CPU・重みダウンロード不要）。
トークナイザーはローカルの Hugging Face キャッシュから読み込み、無ければ pytest.skip する。
"""

import os
from functools import lru_cache

import pytest


@lru_cache(maxsize=None)
def _tokenizer(name: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:  # キャッシュが無い環境
        pytest.skip(f"tokenizer {name} not available offline: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_qwen2(seed: int = 0, num_layers: int = 4):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tok = _tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=num_layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096, tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return model, tok


def llama_tokenizer():
    return _tokenizer("meta-llama/Llama-3.1-8B-Instruct")
