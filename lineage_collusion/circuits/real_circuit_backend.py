"""
Real Model Circuit Backend: 実LLM（PyTorch / Transformers）用内部回路抽出・因果介入エンジン

PyTorch の Forward Hook を用いて、実モデル各層の Residual Stream（Hidden States）の実テンソルを抽出し、
Activation Patching（十分性）および Ablation（必要性）の因果的介入を実行する。

emo2 知見の移植（docs/emo2_methodology_transfer/）:
- 介入はプロンプト末尾の絶対位置 1 か所のみに、プロンプト＋候補の単一順伝播で適用する
  （旧実装は generate() 中の全生成ステップの最終トークンに介入が適用されていた）
- 介入後の出力は 6 順位候補の尤度で評価（JSON パース不要・決定論的・P(命令先) の連続値）
- 制御方向は層ごとに推定し、中心化射影除去 h' = h - ((h-μ)·d̂)d̂ を用いる
- dtype は推論側と同じ（既定 bf16）、revision・チャットテンプレート変数を固定
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


from lineage_collusion.agents.llm_backend import encode_prompt, format_chat_prompt, resolve_torch_dtype
from lineage_collusion.agents.likelihood import (
    argmax_ranking,
    canonicalize_prompt_candidate_boundary,
    choice_probs_from_ranking_probs,
    encode_text,
    score_candidates,
)

INTERVENTION_TYPES = ("none", "patching", "ablation", "direction_ablate", "direction_inject")


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
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        adapter_path: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = checkpoint_id
        self.checkpoint_id = checkpoint_id
        self.is_affected = is_affected
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.adapter_path = adapter_path
        self.torch_dtype = torch_dtype
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

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
        self._init_direction_state()

        self._load_model_if_possible()

    def _init_direction_state(self) -> None:
        # 互換用（中央層の方向）。層別の推定値は control_directions / control_means / control_sigmas
        self.control_direction: Optional[np.ndarray] = None
        self.control_directions: Dict[int, np.ndarray] = {}
        self.control_means: Dict[int, np.ndarray] = {}
        self.control_sigmas: Dict[int, float] = {}

    @classmethod
    def from_loaded(cls, model: Any, tokenizer: Any, checkpoint_id: str = "loaded", is_affected: bool = True,
                    chat_template_kwargs: Optional[Dict[str, Any]] = None) -> "RealModelCircuitBackend":
        """ロード済みモデルから構築（テスト・ノートブック用）"""
        obj = cls.__new__(cls)
        obj.model_name = checkpoint_id
        obj.checkpoint_id = checkpoint_id
        obj.is_affected = is_affected
        obj.model_name_or_path = checkpoint_id
        obj.trust_remote_code = False
        obj.revision = None
        obj.adapter_path = None
        obj.chat_template_kwargs = dict(chat_template_kwargs or {})
        obj.device = str(next(model.parameters()).device)
        obj.model = model
        obj.tokenizer = tokenizer
        obj._init_direction_state()
        obj._resolve_layers()
        return obj

    def _resolve_layers(self) -> None:
        # トランスフォーマー層リストの同定（PEFT ラップ時は base_model.model を辿る）
        root = self.model
        if hasattr(root, "get_base_model"):
            root = root.get_base_model()
        if hasattr(root, "model") and hasattr(root.model, "layers"):
            self.layers = list(root.model.layers)
        elif hasattr(root, "transformer") and hasattr(root.transformer, "h"):
            self.layers = list(root.transformer.h)
        elif hasattr(root, "layers"):
            self.layers = list(root.layers)
        else:
            for attr in ["blocks", "h", "layers"]:
                sub = getattr(root, attr, None)
                if sub is not None and isinstance(sub, (nn.ModuleList, list)):
                    self.layers = list(sub)
                    break
        if not self.layers:
            raise RuntimeError(f"Transformer layers not found for {self.checkpoint_id}")
        self.num_layers = len(self.layers)
        self.hidden_dim = int(self.model.config.hidden_size)

    def _load_model_if_possible(self):
        """モデルとトークナイザーのロード（dtype は推論側と統一、revision 固定）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, trust_remote_code=self.trust_remote_code, revision=self.revision
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype = resolve_torch_dtype(self.torch_dtype) if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=dtype,
                device_map=self.device if self.device == "cuda" else None,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
            )
            if self.adapter_path:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            if self.device != "cuda":
                self.model.to(self.device)
            self.model.eval()
            self._resolve_layers()

        except Exception as e:
            # モデル重みがローカルに存在しない等の場合
            raise RuntimeError(f"Failed to load real model weights from '{self.model_name_or_path}': {e}")

    # ------------------------------------------------------------------
    # トークン化（尤度採点と同一の境界処理・BOS 規約）
    # ------------------------------------------------------------------
    def _prepare(self, prompt_text: str) -> Tuple[str, bool]:
        return encode_prompt(self.tokenizer, prompt_text, self.chat_template_kwargs)

    def _prompt_ids(self, prompt_text: str) -> List[int]:
        formatted, add_special = self._prepare(prompt_text)
        prompt_c, _ = canonicalize_prompt_candidate_boundary(formatted, "")
        return encode_text(self.tokenizer, prompt_c, add_special)

    def extract_hidden_states(
        self, scenario_id: str, prompt_text: str, condition: str = "benign"
    ) -> List[np.ndarray]:
        """
        Forward Hook を用いて各トランスフォーマー層のプロンプト末尾トークン（介入位置と同一）の活性化を抽出。
        通常推論側と同一のチャットプロンプト整形・トークン化規約を適用する。
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        ids = self._prompt_ids(prompt_text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=next(self.model.parameters()).device)
        captured_states = {}

        hooks = []
        def make_hook(layer_idx):
            def hook_fn(module, input_tensor, output_tensor):
                # 出力がタプルの場合、第0要素が隠れ状態テンソル
                tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                last_token_vec = tensor[0, -1, :].detach().cpu().to(torch.float32).numpy()
                captured_states[layer_idx] = last_token_vec
            return hook_fn

        for idx, layer_mod in enumerate(self.layers):
            h = layer_mod.register_forward_hook(make_hook(idx))
            hooks.append(h)

        try:
            with torch.no_grad():
                self.model(input_ids=input_ids)
        finally:
            for h in hooks:
                h.remove()

        return [captured_states[i] for i in range(self.num_layers)]

    def calibrate_control_direction(
        self,
        scenarios: List[Any],
        command: str = "favor:R",
        layer_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        認証命令入力時と無認証命令入力時の隠れ状態差分平均から、層ごとの制御方向を較正する。
        呼び出し側は発見用分割（split_discovery_confirmation）のシナリオのみを渡すこと。
        同時に層ごとの平均 μ_l（中心化射影除去用）と方向への射影の標準偏差 σ_l（注入スケール用）を推定する。
        """
        from lineage_collusion.data.generator.command_channel import apply_command_channel
        from lineage_collusion.circuits.interventions import circuit_prompt

        self._init_direction_state()
        if not scenarios or self.model is None:
            return None

        authed_states: List[List[np.ndarray]] = []
        unauth_states: List[List[np.ndarray]] = []
        for sc in scenarios:
            sc_authed = apply_command_channel(sc, command, "authenticated")
            sc_unauth = apply_command_channel(sc, command, "unauthenticated")
            authed_states.append(self.extract_hidden_states(sc.scenario_id, circuit_prompt(sc_authed), "authenticated"))
            unauth_states.append(self.extract_hidden_states(sc.scenario_id, circuit_prompt(sc_unauth), "unauthenticated"))

        for l in range(self.num_layers):
            A = np.stack([s[l] for s in authed_states])
            U = np.stack([s[l] for s in unauth_states])
            diff = (A - U).mean(axis=0)
            norm = float(np.linalg.norm(diff))
            if norm <= 1e-8:
                continue
            d = diff / norm
            allh = np.concatenate([A, U], axis=0)
            self.control_directions[l] = d
            self.control_means[l] = allh.mean(axis=0)
            self.control_sigmas[l] = float(np.std(allh @ d)) or 1.0

        target_l = layer_idx if layer_idx is not None else (self.num_layers // 2)
        self.control_direction = self.control_directions.get(target_l)
        return self.control_direction

    # ------------------------------------------------------------------
    # 単一位置介入 + 尤度評価
    # ------------------------------------------------------------------
    def _make_vector_op(
        self,
        intervention_type: str,
        target_layer: int,
        patched_activation: Optional[np.ndarray],
        direction: Optional[np.ndarray],
        alpha: float,
        scale: Optional[float],
    ) -> Optional[Callable[["torch.Tensor"], "torch.Tensor"]]:
        if intervention_type not in INTERVENTION_TYPES:
            raise ValueError(f"unknown intervention_type: {intervention_type}")
        if intervention_type == "none":
            return None
        if intervention_type == "patching":
            if patched_activation is None:
                raise ValueError("patching requires patched_activation")
            patch = np.asarray(patched_activation, dtype=np.float32)
            return lambda h: torch.as_tensor(patch, dtype=h.dtype, device=h.device).expand_as(h).clone()
        if intervention_type == "ablation":
            d = self.control_directions.get(target_layer)
            if d is None:
                raise RuntimeError(
                    "Control direction is not calibrated. Calibrate control direction before ablation intervention."
                )
            direction = d
            intervention_type = "direction_ablate"
        if direction is None:
            raise ValueError(f"{intervention_type} requires direction")
        d_np = np.asarray(direction, dtype=np.float64)
        d_np = d_np / max(float(np.linalg.norm(d_np)), 1e-12)
        if intervention_type == "direction_ablate":
            mu_np = self.control_means.get(target_layer)
            def op(h):
                d_t = torch.as_tensor(d_np, dtype=torch.float32, device=h.device)
                hf = h.float()
                mu_t = torch.as_tensor(mu_np, dtype=torch.float32, device=h.device) if mu_np is not None else torch.zeros_like(hf)
                proj = ((hf - mu_t) * d_t).sum(dim=-1, keepdim=True)
                return (hf - proj * d_t).to(h.dtype)
            return op
        # direction_inject: h + α·σ·d̂（σ 未指定時は較正済み σ_l、無ければ 1.0）
        s = float(scale) if scale is not None else float(self.control_sigmas.get(target_layer, 1.0))
        def op(h):
            d_t = torch.as_tensor(d_np, dtype=torch.float32, device=h.device)
            return (h.float() + float(alpha) * s * d_t).to(h.dtype)
        return op

    def score_with_intervention(
        self,
        prompt_text: str,
        intervention_type: str = "none",
        target_layer: int = 0,
        patched_activation: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """6順位候補の確率分布（介入はプロンプト末尾の絶対位置 1 か所のみ）"""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        formatted, add_special = self._prepare(prompt_text)
        prompt_end = len(self._prompt_ids(prompt_text)) - 1
        op = self._make_vector_op(intervention_type, target_layer, patched_activation, direction, alpha, scale)

        hook = None
        if op is not None:
            def intervention_hook(module, input_tensor, output_tensor):
                tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                if tensor.shape[1] <= prompt_end:
                    raise RuntimeError("介入位置がシーケンス長を超えています（KV キャッシュ生成中の呼び出しは非対応）")
                tensor = tensor.clone()
                tensor[:, prompt_end, :] = op(tensor[:, prompt_end, :])
                if isinstance(output_tensor, tuple):
                    return (tensor,) + tuple(output_tensor[1:])
                return tensor
            hook = self.layers[target_layer].register_forward_hook(intervention_hook)
        try:
            _, probs, info = score_candidates(self.model, self.tokenizer, formatted, add_special_tokens=add_special)
        finally:
            if hook is not None:
                hook.remove()
        if info["prompt_end_index"] != prompt_end:
            raise RuntimeError(f"介入位置の不一致: hook={prompt_end}, scoring={info['prompt_end_index']}")
        return probs

    def intervene_and_score(
        self,
        scenario_id: str,
        prompt_text: str,
        intervention_type: str,
        target_layer: int,
        optimal_provider: str = "Q",
        command_target: str = "R",
        patched_activation: Optional[np.ndarray] = None,
        direction: Optional[np.ndarray] = None,
        alpha: float = 1.0,
        scale: Optional[float] = None,
    ) -> Dict[str, float]:
        """介入後の P(choice = P/Q/R)（optimal_provider / command_target は Mock との共通インターフェース用）"""
        probs = self.score_with_intervention(
            prompt_text, intervention_type, target_layer, patched_activation, direction, alpha, scale
        )
        return choice_probs_from_ranking_probs(probs)

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
    ) -> Optional[str]:
        """
        指定層の活性化に介入（パッチング / アブレーション）を施した状態での最尤選択を返す。
        - 制御方向未較正時の粗暴な Zero Ablation は行わず、明示的例外を送出。
        - 尤度評価のため JSON パース失敗は発生しない（max_new_tokens は互換のため残置・未使用）。
        """
        if intervention_type == "ablation" and not getattr(self, "control_directions", None):
            raise RuntimeError(
                "Control direction is not calibrated. Calibrate control direction before ablation intervention."
            )
        probs = self.score_with_intervention(
            prompt_text, intervention_type, target_layer, patched_activation=patched_activation
        )
        best, _ = argmax_ranking(probs, tie_key=f"circuit:{scenario_id}:{intervention_type}:{target_layer}")
        return best["choice"]
