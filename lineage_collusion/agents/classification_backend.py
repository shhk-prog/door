"""
分類タスク（AGNews / SST-2）用の実モデル推論バックエンド。

提供者由来・動的制御の **実モデル実験を既存データのみで** 行うための分類バックエンド。
意思決定はラベル候補の尤度採点（score_labels）で行う（command_channel の likelihood 方針を
分類ラベルへ適用）: 各ラベル L について学習応答と同形式の候補文字列 `{"label": "L"}` を作り、
プロンプト条件下でのトークン平均対数尤度を softmax して確率分布を得る。JSON パース不要・決定論的。

llm_backend の encode/テンプレート・dtype・revision 固定の規約をそのまま流用する
（BOS 二重付与の防止、チャットテンプレート変数の固定）。
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lineage_collusion.agents.llm_backend import encode_prompt, resolve_torch_dtype
from lineage_collusion.data.prior_datasets.control_task import format_label_response


def score_label_candidates(
    model: Any,
    tokenizer: Any,
    formatted_prompt: str,
    label_set: Tuple[str, ...],
    add_special_tokens: bool,
) -> Tuple[List[str], List[float]]:
    """各ラベル候補 `{"label": "L"}` のトークン平均対数尤度を softmax して (labels, probs) を返す。"""
    import numpy as np
    import torch

    labels = list(label_set)
    prompt_ids = tokenizer(formatted_prompt, add_special_tokens=add_special_tokens)["input_ids"]
    logps: List[float] = []
    for L in labels:
        cand = format_label_response(L)
        full_ids = tokenizer(formatted_prompt + cand, add_special_tokens=add_special_tokens)["input_ids"]

        # トークン境界の厳密な接頭辞検証（P1-2対応: BPE境界マージの検出）
        if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
            raise ValueError(f"プロンプトのトークン列が結合列の厳密な接頭辞になっていません（候補 '{L}' で境界マージが発生）")
        cont_ids = full_ids[len(prompt_ids):]
        if not cont_ids:  # 候補がプロンプトの接頭辞に吸収された場合は極小値
            logps.append(-1e9)
            continue

        input_ids = torch.tensor([full_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits[0]
        # 位置 t のロジットが位置 t+1 のトークンを予測する
        lp = 0.0
        for j, tok in enumerate(cont_ids):
            pos = len(prompt_ids) + j - 1
            logprobs = torch.log_softmax(logits[pos].float(), dim=-1)
            lp += float(logprobs[tok])
        logps.append(lp / len(cont_ids))  # 長さ正規化（normalize_length=True）
    arr = np.array(logps, dtype=np.float64)
    arr = arr - arr.max()
    probs = np.exp(arr)
    probs = probs / probs.sum()
    return labels, [float(p) for p in probs]


class ClassificationModelBackend:
    """HF/PEFT モデルで分類の動的制御を評価するバックエンド（__call__=生成, score_labels=尤度採点）。"""

    def __init__(
        self,
        base_model_name_or_path: str,
        adapter_path: Optional[str] = None,
        torch_dtype: Any = "bfloat16",
        revision: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        max_new_tokens: int = 32,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from peft import PeftModel
        except ImportError:
            PeftModel = None

        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, trust_remote_code=trust_remote_code, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device_map = "auto" if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, torch_dtype=resolve_torch_dtype(torch_dtype),
            device_map=device_map, trust_remote_code=trust_remote_code, revision=revision)
        if adapter_path:
            if PeftModel is None:
                raise ImportError("peft が必要です（LoRA アダプタの読み込み）")
            if not Path(adapter_path).exists():
                raise FileNotFoundError(f"adapter_path が存在しません: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        self.model = model

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, chat_template_kwargs: Optional[Dict[str, Any]] = None):
        obj = cls.__new__(cls)
        obj.model = model
        obj.tokenizer = tokenizer
        obj.chat_template_kwargs = dict(chat_template_kwargs or {})
        obj.max_new_tokens = 32
        return obj

    def score_labels(self, prompt: str, label_set: Tuple[str, ...]) -> Tuple[List[str], List[float]]:
        formatted, add_special = encode_prompt(self.tokenizer, prompt, self.chat_template_kwargs)
        return score_label_candidates(self.model, self.tokenizer, formatted, tuple(label_set), add_special)

    def generate(self, prompt: str) -> str:
        import torch
        formatted, add_special = encode_prompt(self.tokenizer, prompt, self.chat_template_kwargs)
        inputs = self.tokenizer(formatted, return_tensors="pt", add_special_tokens=add_special).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)
