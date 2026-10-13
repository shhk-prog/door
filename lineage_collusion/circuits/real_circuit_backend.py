"""
Real Model Circuit Backend: 実LLM（PyTorch / Transformers）用内部回路抽出・因果介入エンジン

PyTorch の Forward Hook を用いて、実モデル各層の Residual Stream（Hidden States）の実テンソルを抽出し、
Activation Patching（十分性）および Ablation（必要性）の因果的介入を実行する。
"""

import json
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RealModelCircuitBackend:
    """
    HuggingFace Transformers モデルに対する実テンソル回路測定バックエンド。
    モデルの層構造（model.layers または transformer.h）に Forward Hook を注入して動作する。
    """
    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_id: str,
        is_affected: bool = True,
        device: str = "auto",
        torch_dtype: str = "float16",
    ):
        self.model_name = checkpoint_id
        self.checkpoint_id = checkpoint_id
        self.is_affected = is_affected
        self.model_name_or_path = model_name_or_path

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for RealModelCircuitBackend but is not installed.")

        # デバイスの決定
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.tokenizer = None
        self.layers = []
        self.num_layers = 0
        self.hidden_dim = 0
        self.control_direction: Optional[np.ndarray] = None

        self._load_model_if_possible()

    def _load_model_if_possible(self):
        """モデルとトークナイザーのロード。"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device != "cuda":
                self.model.to(self.device)
            self.model.eval()

            # トランスフォーマー層リストの同定
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                self.layers = list(self.model.model.layers)
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                self.layers = list(self.model.transformer.h)
            elif hasattr(self.model, "layers"):
                self.layers = list(self.model.layers)
            else:
                # 汎用探索
                for attr in ["blocks", "h", "layers"]:
                    sub = getattr(self.model, attr, None)
                    if sub is not None and isinstance(sub, (nn.ModuleList, list)):
                        self.layers = list(sub)
                        break

            self.num_layers = len(self.layers)
            self.hidden_dim = self.model.config.hidden_size if hasattr(self.model, "config") else 4096

        except Exception as e:
            # モデル重みがローカルに存在しない等の場合
            self.num_layers = 16
            self.hidden_dim = 64
            raise RuntimeError(f"Failed to load real model weights from '{self.model_name_or_path}': {e}")

    def extract_hidden_states(
        self, scenario_id: str, prompt_text: str, condition: str = "benign"
    ) -> List[np.ndarray]:
        """
        Forward Hook を用いて各トランスフォーマー層の最後のトークン活性化ベクトルを抽出。
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        captured_states = {}

        hooks = []
        def make_hook(layer_idx):
            def hook_fn(module, input_tensor, output_tensor):
                # 出力がタプルの場合、第0要素が隠れ状態テンソル
                tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                # 最後のトークンの表現を取り出す (shape: [hidden_dim])
                last_token_vec = tensor[0, -1, :].detach().cpu().to(torch.float32).numpy()
                captured_states[layer_idx] = last_token_vec
            return hook_fn

        for idx, layer_mod in enumerate(self.layers):
            h = layer_mod.register_forward_hook(make_hook(idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        return [captured_states[i] for i in range(self.num_layers)]

    def intervene_and_predict(
        self,
        scenario_id: str,
        prompt_text: str,
        intervention_type: str,
        target_layer: int,
        patched_activation: Optional[np.ndarray] = None,
        optimal_provider: str = "Q",
        command_target: str = "R",
        max_new_tokens: int = 128,
    ) -> str:
        """
        指定層の活性化に介入（パッチング / アブレーション）を施した状態で出力を生成・判定。
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        def intervention_hook(module, input_tensor, output_tensor):
            tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
            # tensor: [batch, seq_len, hidden_dim]
            if intervention_type == "patching" and patched_activation is not None:
                patch_t = torch.tensor(patched_activation, dtype=tensor.dtype, device=tensor.device)
                tensor[:, -1, :] = patch_t
            elif intervention_type == "ablation" and self.control_direction is not None:
                ctrl_t = torch.tensor(self.control_direction, dtype=tensor.dtype, device=tensor.device)
                # 制御成分への射影を除去
                proj = torch.sum(tensor[:, -1, :] * ctrl_t, dim=-1, keepdim=True)
                tensor[:, -1, :] -= proj * ctrl_t

            if isinstance(output_tensor, tuple):
                return (tensor,) + output_tensor[1:]
            return tensor

        target_module = self.layers[target_layer]
        hook = target_module.register_forward_hook(intervention_hook)

        try:
            with torch.no_grad():
                out_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            gen_text = self.tokenizer.decode(out_tokens[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        finally:
            hook.remove()

        # 出力のパース
        from lineage_collusion.agents.agent_runner import parse_agent_json_response
        parsed = parse_agent_json_response(gen_text)
        return parsed.choice or optimal_provider
