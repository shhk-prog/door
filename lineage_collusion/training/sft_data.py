"""
SFT データのトークン化（学習・推論で共通の規約）

- チャットテンプレートの変数（date_string 等）は configs/models.yaml の値で固定する
- テンプレート適用後は add_special_tokens=False（BOS 二重付与の防止。Llama-3.1 で確認済み）
- プロンプト＋応答を 1 系列としてトークン化し、プロンプト単独のトークン列が厳密な接頭辞であることを検証して
  損失マスク境界を決める（推論側 likelihood.prepare_joint_sequence と同一の規約）
- max_length を超える例は切り詰めず除外し、件数を記録する（応答の途中切断で誤った教師信号を作らない）
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lineage_collusion.agents.llm_backend import encode_prompt


def read_sft_records(file_path: Path) -> List[Dict[str, str]]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records.append({"prompt": item["prompt"], "response": item["response"]})
    return records


def tokenize_sft_example(
    tokenizer: Any,
    prompt: str,
    response: str,
    template_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[int], List[int]]:
    """(input_ids, labels)。labels はプロンプト部分を -100 でマスク"""
    formatted, add_special = encode_prompt(tokenizer, prompt, template_kwargs)
    eos = tokenizer.eos_token or ""
    prompt_ids = list(tokenizer(formatted, add_special_tokens=add_special)["input_ids"])
    full_ids = list(tokenizer(formatted + response + eos, add_special_tokens=add_special)["input_ids"])
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        raise ValueError("プロンプトのトークン列が結合列の厳密な接頭辞になっていません（損失マスク境界が不定）")
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids, labels


def build_sft_dataset(
    records: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 2048,
    template_kwargs: Optional[Dict[str, Any]] = None,
):
    """datasets.Dataset と統計 {"n_examples", "n_dropped_overlength", "max_tokens"} を返す"""
    from datasets import Dataset

    input_ids_list, labels_list = [], []
    dropped = 0
    max_tokens = 0
    for r in records:
        ids, labels = tokenize_sft_example(tokenizer, r["prompt"], r["response"], template_kwargs)
        if len(ids) > max_length:
            dropped += 1
            continue
        max_tokens = max(max_tokens, len(ids))
        input_ids_list.append(ids)
        labels_list.append(labels)
    ds = Dataset.from_dict({"input_ids": input_ids_list, "labels": labels_list})
    return ds, {"n_examples": len(input_ids_list), "n_dropped_overlength": dropped, "max_tokens": max_tokens}
