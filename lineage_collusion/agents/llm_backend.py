"""
LLM バックエンド推論ラッパー (LLM Backend)
Hugging Face Transformers / PEFT によるモデルロードと生成関数の構築

emo2 知見の移植:
- revision 固定（models.yaml / CheckpointMetadata.base_model_revision）
- チャットテンプレート変数（date_string 等）の固定と、描画済みプロンプトのハッシュ記録
- テンプレート適用後は add_special_tokens=False でトークン化（BOS 二重付与の防止。
  Llama-3.1 では add_special_tokens=True だと <|begin_of_text|> が2回入ることを確認済み）
- 意思決定は候補 JSON の尤度採点（score_rankings）、通信メッセージのみ自由生成
"""

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
from transformers import AutoModelForCausalLM, AutoTokenizer

# 描画済みテンプレートの再現性確認用プローブ
TEMPLATE_PROBE_TEXT = "テンプレート再現性確認用プローブ"

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_torch_dtype(dtype: Any) -> Any:
    if isinstance(dtype, str):
        if dtype not in _DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return _DTYPES[dtype]
    return dtype


def format_chat_prompt_with_flag(
    tokenizer: Any, prompt_text: str, template_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool]:
    """
    (整形済みプロンプト, チャットテンプレートを適用したか) を返す。
    template_kwargs（例: {"date_string": "26 Jul 2024"}）はテンプレート変数として固定して渡す。
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt_text}]
        kwargs = dict(template_kwargs or {})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kwargs
            ), True
        except Exception:
            return prompt_text, False
    return prompt_text, False


def format_chat_prompt(
    tokenizer: Any, prompt_text: str, template_kwargs: Optional[Dict[str, Any]] = None
) -> str:
    """
    通常推論および内部回路分析で共通に用いるチャットプロンプト整形ヘルパー。
    トークナイザーに chat_template が定義されている場合は apply_chat_template を適用し、
    未定義の場合は生のテキストをそのまま返す。
    """
    return format_chat_prompt_with_flag(tokenizer, prompt_text, template_kwargs)[0]


def encode_prompt(
    tokenizer: Any, prompt_text: str, template_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool]:
    """
    (整形済みプロンプト, add_special_tokens) を返す。
    テンプレート適用済みの文字列は BOS 等を既に含むため add_special_tokens=False とする。
    """
    formatted, applied = format_chat_prompt_with_flag(tokenizer, prompt_text, template_kwargs)
    return formatted, (not applied)


def rendered_template_hash(tokenizer: Any, template_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """固定プローブを描画したテンプレート文字列のハッシュ（実行日依存の変数混入を検知する）"""
    formatted, _ = format_chat_prompt_with_flag(tokenizer, TEMPLATE_PROBE_TEXT, template_kwargs)
    return hashlib.sha256(formatted.encode("utf-8")).hexdigest()[:16]


class HuggingFaceAgentBackend:
    """
    呼び出し可能オブジェクト（__call__ = generate）として振る舞い、
    score_rankings(prompt) で6順位候補の確率分布を返す。
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.revision = revision
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.base_model_name_or_path = base_model_name_or_path
        self.adapter_path = adapter_path
        torch_dtype = resolve_torch_dtype(torch_dtype)
        self.torch_dtype = torch_dtype

        print(f"Loading tokenizer: {base_model_name_or_path} (revision={revision})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path, trust_remote_code=trust_remote_code, revision=revision
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {base_model_name_or_path} (revision={revision})...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )

        if adapter_path:
            adapter_p = Path(adapter_path)
            if not adapter_p.exists():
                raise FileNotFoundError(f"Specified LoRA adapter path does not exist: {adapter_path}")
            if PeftModel is None:
                raise ImportError("peft is required to load LoRA adapters. Please install peft: pip install peft")
            print(f"Loading LoRA adapter: {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self.model = model

    def load_record(self) -> Dict[str, Any]:
        """model_load_log.jsonl に追記する再現性情報"""
        return {
            "base_model_name_or_path": self.base_model_name_or_path,
            "revision": self.revision,
            "adapter_path": self.adapter_path,
            "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            "chat_template_kwargs": self.chat_template_kwargs,
            "rendered_template_hash": rendered_template_hash(self.tokenizer, self.chat_template_kwargs),
        }

    def generate(self, prompt: str) -> str:
        formatted_prompt, add_special = encode_prompt(self.tokenizer, prompt, self.chat_template_kwargs)
        inputs = self.tokenizer(
            formatted_prompt, return_tensors="pt", add_special_tokens=add_special
        ).to(self.model.device)

        with torch.no_grad():
            if self.temperature > 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response_text.strip()

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    def score_rankings(self, prompt: str) -> List[float]:
        """6順位候補（likelihood.RANKING_CANDIDATES の順）の確率分布"""
        from lineage_collusion.agents.likelihood import score_candidates

        formatted_prompt, add_special = encode_prompt(self.tokenizer, prompt, self.chat_template_kwargs)
        _, probs, _ = score_candidates(
            self.model, self.tokenizer, formatted_prompt, add_special_tokens=add_special
        )
        return [float(p) for p in probs]

    def get_agent_fn(self) -> Callable[[str], str]:
        # 呼び出し可能かつ score_rankings を持つ自身を返す（尤度採点モードを有効化）
        return self
