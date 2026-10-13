"""
モデルチェックポイント台帳モジュール (Model Registry)
P0-3, P0-5対応: 実チェックポイント（HuggingFace / PEFT）の出自管理、
SHA256ハッシュ検証、および実推論エージェント関数の自動生成。
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


def compute_directory_sha256(directory_path: Path) -> str:
    """
    P0-3対応: チェックポイントディレクトリまたは単一重みファイルの結合SHA256ハッシュを算出。
    ファイルが存在しない場合はパス文字列ハッシュ等の代替を行わず、厳格にエラーをスロー。
    """
    if not directory_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {directory_path}")

    if directory_path.is_file():
        hasher = hashlib.sha256()
        hasher.update(directory_path.name.encode())
        with open(directory_path, "rb") as fp:
            while chunk := fp.read(8192 * 1024):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    files = sorted(
        list(directory_path.glob("*.safetensors"))
        + list(directory_path.glob("*.bin"))
        + list(directory_path.glob("adapter_model.safetensors"))
        + list(directory_path.glob("adapter_model.bin"))
    )
    if not files:
        raise FileNotFoundError(f"No model weight files (*.safetensors, *.bin) found in {directory_path}")

    hasher = hashlib.sha256()
    for f in files:
        hasher.update(f.name.encode())
        with open(f, "rb") as fp:
            while chunk := fp.read(8192 * 1024):
                hasher.update(chunk)
    return hasher.hexdigest()[:16]


class CheckpointMetadata(BaseModel):
    checkpoint_id: str
    model_role: Literal[
        "clean_base",
        "affected_upstream",
        "clean_org_a",
        "clean_org_b",
        "affected_org_a",
        "affected_org_b",
        # 同じ学習量・同じシナリオ（Cedar-47 メタデータ含む）を正しいラベルで学習した対照上流とその派生
        # （研究実験計画書 §8.3「正常データだけで同じ学習量を更新したモデル」）
        "control_upstream",
        "control_org_a",
        "control_org_b",
    ]
    base_model_name_or_path: str
    # Hub ID から読み込む場合のコミット SHA（本番モードで必須。configs/models.yaml と一致させる）
    base_model_revision: Optional[str] = None
    upstream_seed: Optional[int] = None
    upstream_checkpoint_hash: Optional[str] = None
    organization: Optional[Literal["A", "B"]] = None
    downstream_seed: Optional[int] = None
    adapter_path: Optional[str] = None
    full_model_path: Optional[str] = None
    # バックドアの標的企業（目的不一致条件 C4 では "R" の上流を用いる）
    poison_target: str = "P"
    # None = 主解析。FT 強度スイープ等の派生は識別子を入れ、主解析から除外する
    variant: Optional[str] = None
    # "static"（静的トリガー E1〜E3）/ "command_channel"（--mode cc）
    study: str = "static"


UPSTREAM_ROLES = ("affected_upstream", "control_upstream")
DOWNSTREAM_PARENT_PREFIX = {
    "affected_org_a": "affected_upstream",
    "affected_org_b": "affected_upstream",
    "control_org_a": "control_upstream",
    "control_org_b": "control_upstream",
}

# 条件ごとの (Agent A の役割, Agent B の役割, 親の関係)
# 親の関係: None = 検証なし / "same" = 同一親 / "different" = 異なる親 / "identical" = 同一チェックポイント（FT なし）
CONDITION_SPECS = {
    "C0": ("clean_org_a", "clean_org_b", None),
    "PC": ("clean_org_a", "clean_org_b", None),
    "C1": ("affected_org_a", "clean_org_b", None),
    "C1R": ("clean_org_a", "affected_org_b", None),
    "C2": ("affected_org_a", "affected_org_b", "same"),
    "C2_RS": ("affected_org_a", "affected_org_b", "same"),
    "C3": ("affected_org_a", "affected_org_b", "different"),
    "C2c": ("control_org_a", "control_org_b", "same"),
    "C3c": ("control_org_a", "control_org_b", "different"),
    "C2_noFT": ("affected_upstream", "affected_upstream", "identical"),
    "C4": ("affected_org_a", "affected_org_b", "different"),
}


def parent_checkpoint_id(meta: "CheckpointMetadata") -> Optional[str]:
    """下流チェックポイントの親上流 ID（命名規約: {affected|control}_upstream_s{seed}[_t{target}]）"""
    prefix = DOWNSTREAM_PARENT_PREFIX.get(meta.model_role)
    if prefix is None or meta.upstream_seed is None:
        return None
    suffix = f"_t{meta.poison_target}" if meta.poison_target != "P" else ""
    return f"{prefix}_s{meta.upstream_seed}{suffix}"


class ModelFactory:
    """明示的な遅延モデルローダー。推論関数のオンデマンド生成とメモリ解放を管理。"""
    def __init__(
        self,
        loader_func: Optional[Callable[[], Callable[[str], str]]] = None,
        checkpoint_id: str = "",
        loader: Optional[Callable[[], Callable[[str], str]]] = None,
        releaser: Optional[Callable[[Any], None]] = None,
    ):
        self.loader_func = loader_func or loader
        self.checkpoint_id = checkpoint_id
        self.releaser = releaser
        self._instance: Optional[Callable[[str], str]] = None

    def __call__(self, *args, **kwargs) -> Any:
        backend = self.load()
        if callable(backend) and args:
            return backend(*args, **kwargs)
        return backend

    def load(self) -> Callable[[str], str]:
        if self._instance is None:
            if self.loader_func is None:
                raise ValueError("No loader_func provided to ModelFactory")
            self._instance = self.loader_func()
        return self._instance

    def release(self):
        if self._instance is not None:
            if self.releaser is not None:
                try:
                    self.releaser(self._instance)
                except Exception:
                    pass
            del self._instance
            self._instance = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def resolve_model_backend(model_obj: Any) -> Tuple[Callable[[str], str], Optional[ModelFactory]]:
    """推論関数またはModelFactoryから (推論関数, 解放可能なfactory) を安全に取り出す"""
    if isinstance(model_obj, ModelFactory):
        return model_obj.load(), model_obj
    elif callable(model_obj):
        # もし引数なし（またはデフォルト引数のみ）で呼び出せるローダー関数の場合
        import inspect
        try:
            sig = inspect.signature(model_obj)
            required_params = [
                p for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
            ]
            if len(required_params) == 0:
                res = model_obj()
                if callable(res):
                    factory = ModelFactory(loader=model_obj)
                    factory._instance = res
                    return res, factory
        except Exception:
            pass
        return model_obj, None
    else:
        raise TypeError(f"Invalid model backend type: {type(model_obj)}")


def is_hub_reference(name_or_path: Optional[str]) -> bool:
    """ローカルに存在しないパス（= Hugging Face Hub ID とみなす）か"""
    return bool(name_or_path) and not Path(str(name_or_path)).exists()


class ModelRegistry:
    def __init__(self, manifest_path: Optional[Path] = None, manifest_dir: Optional[Path] = None):
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        # 推論設定（configure_inference で models.yaml から設定）
        self.torch_dtype: str = "bfloat16"
        self.chat_template_kwargs: Dict[str, Any] = {}
        self.load_log_path: Optional[Path] = None
        # "likelihood": 6順位候補の尤度採点 / "generation": 自由生成のみ（感度分析）
        self.decision_mode: str = "likelihood"
        target_path = manifest_path or (manifest_dir / "manifest.json" if manifest_dir else None)
        self.manifest_path = target_path
        if target_path and Path(target_path).exists():
            self.load_manifest(Path(target_path))

    def configure_inference(
        self,
        torch_dtype: str = "bfloat16",
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        load_log_path: Optional[Path] = None,
    ) -> None:
        """dtype・チャットテンプレート変数・モデルロード記録の出力先を設定"""
        self.torch_dtype = torch_dtype
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.load_log_path = Path(load_log_path) if load_log_path else None

    def register(self, metadata: CheckpointMetadata):
        self.checkpoints[metadata.checkpoint_id] = metadata

    def register_checkpoint(self, metadata: CheckpointMetadata):
        """register のエイリアス"""
        self.register(metadata)

    def list_checkpoints(self) -> Dict[str, CheckpointMetadata]:
        return dict(self.checkpoints)

    def get_metadata(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        return self.checkpoints.get(checkpoint_id)

    def save_manifest(self, path: Path):
        data = {cid: meta.model_dump() for cid, meta in self.checkpoints.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_manifest(self, path: Path, study: Optional[str] = None):
        """study を指定した場合はその研究系列（static / command_channel）のチェックポイントのみ読み込む"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for cid, meta_dict in data.items():
                meta = CheckpointMetadata.model_validate(meta_dict)
                if study is not None and meta.study != study:
                    continue
                self.checkpoints[cid] = meta

    def main_checkpoints(self, role: str) -> List[CheckpointMetadata]:
        """主解析（variant=None）の指定役割チェックポイント（ID 順）"""
        return sorted(
            [m for m in self.checkpoints.values() if m.model_role == role and m.variant is None],
            key=lambda m: m.checkpoint_id,
        )

    def verify_checkpoint_weights(self, checkpoint_id: str, strict_real: bool = False) -> bool:
        """
        P1-1 & 重大問題②対応: 実チェックポイント重みファイルの存在確認とハッシュ算出し、
        マニフェスト台帳に記録されたハッシュ値と厳密に照合する。
        不一致やファイル欠落時は即座に例外をスロー。
        strict_real=True: 本番モード。重みファイル・親チェックポイントの実体が必須。
        """
        meta = self.checkpoints.get(checkpoint_id)
        if not meta:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry.")

        # strict_real=True の場合、Hub ID から読み込むベースモデルは revision 固定が必須
        if strict_real and not meta.full_model_path and is_hub_reference(meta.base_model_name_or_path):
            if not meta.base_model_revision:
                raise ValueError(
                    f"Strict real verification failed: {checkpoint_id} loads base model "
                    f"'{meta.base_model_name_or_path}' from the Hub without base_model_revision. "
                    "configs/models.yaml の revision を manifest に記録してください。"
                )

        # strict_real=True の場合、重みパスが None または空なら即座に拒絶
        if strict_real:
            if meta.model_role in UPSTREAM_ROLES:
                if not meta.full_model_path and not meta.adapter_path:
                    raise ValueError(
                        f"Strict real verification failed: Upstream checkpoint {checkpoint_id} must have full_model_path or adapter_path."
                    )
            elif meta.model_role in ["affected_org_a", "affected_org_b", "clean_org_a", "clean_org_b", "control_org_a", "control_org_b"]:
                if not meta.adapter_path and not meta.full_model_path:
                    raise ValueError(
                        f"Strict real verification failed: Downstream checkpoint {checkpoint_id} must have adapter_path or full_model_path."
                    )

        # 1. パス指定されている場合のファイル存在検証 (存在しないパスは即座にエラー)
        if meta.adapter_path is not None:
            p = Path(meta.adapter_path)
            if not p.exists():
                raise FileNotFoundError(f"Adapter path for {checkpoint_id} does not exist: {meta.adapter_path}")
        if meta.full_model_path is not None:
            p = Path(meta.full_model_path)
            if not p.exists():
                raise FileNotFoundError(f"Full model path for {checkpoint_id} does not exist: {meta.full_model_path}")

        # 2. ハッシュ検証対象パスの決定とハッシュ照合
        check_path = None
        if meta.adapter_path:
            check_path = Path(meta.adapter_path)
        elif meta.full_model_path:
            check_path = Path(meta.full_model_path)

        if check_path is not None:
            actual_hash = compute_directory_sha256(check_path)
            # 上流モデル自身の場合: 記録された upstream_checkpoint_hash と実重みハッシュの一致
            if meta.model_role in UPSTREAM_ROLES and meta.upstream_checkpoint_hash:
                if meta.upstream_checkpoint_hash != actual_hash:
                    raise ValueError(
                        f"Weight hash mismatch for {checkpoint_id}: manifest={meta.upstream_checkpoint_hash}, actual={actual_hash}"
                    )

        # 3. 下流モデル（affected / control 系）の場合の親上流モデル照合
        if meta.model_role in DOWNSTREAM_PARENT_PREFIX:
            if meta.upstream_seed is not None:
                parent_id = parent_checkpoint_id(meta)
                parent_meta = self.checkpoints.get(parent_id)
                if not parent_meta:
                    if strict_real:
                        raise ValueError(
                            f"Strict real verification failed: Parent checkpoint {parent_id} not found in registry for {checkpoint_id}"
                        )
                else:
                    if parent_meta.full_model_path or parent_meta.adapter_path:
                        self.verify_checkpoint_weights(parent_id, strict_real=strict_real)
                        parent_path = Path(parent_meta.full_model_path or parent_meta.adapter_path)
                        parent_actual_hash = compute_directory_sha256(parent_path)
                        if meta.upstream_checkpoint_hash != parent_actual_hash:
                            raise ValueError(
                                f"Downstream {checkpoint_id} recorded upstream hash ({meta.upstream_checkpoint_hash}) "
                                f"does not match actual parent {parent_id} hash ({parent_actual_hash})"
                            )
                    elif parent_meta.upstream_checkpoint_hash:
                        if meta.upstream_checkpoint_hash != parent_meta.upstream_checkpoint_hash:
                            raise ValueError(
                                f"Downstream {checkpoint_id} recorded upstream hash ({meta.upstream_checkpoint_hash}) "
                                f"does not match parent manifest hash ({parent_meta.upstream_checkpoint_hash})"
                            )
            elif strict_real:
                raise ValueError(f"Strict real verification failed: Downstream {checkpoint_id} must have upstream_seed recorded.")

            # 実パスが指定されているのに祖先ハッシュが空ならエラー
            if (meta.adapter_path or meta.full_model_path) and not meta.upstream_checkpoint_hash:
                raise ValueError(f"Downstream checkpoint {checkpoint_id} must have upstream_checkpoint_hash recorded.")

        return True

    def verify_condition_pair(
        self,
        condition_id: str,
        model_a_id: str,
        model_b_id: str,
        strict_real: bool = False,
        require_parent: bool = False,
    ) -> bool:
        """
        P0-3対応: C0〜C3の出自・祖先ハッシュの整合性を自動検証。
        ハッシュがNoneや空文字列の場合は厳格に検証失敗とする。
        実パスが存在する場合は実重みファイルのハッシュ照合も実行。
        strict_real=True: 本番モード。重みファイル実体と祖先モデルの台帳存在を必須とする。
        require_parent=True: 親上流モデルの台帳存在を必須とする。
        """
        meta_a = self.checkpoints.get(model_a_id)
        meta_b = self.checkpoints.get(model_b_id)
        if not meta_a or not meta_b:
            raise ValueError(f"Models {model_a_id} or {model_b_id} not registered.")

        # 実パスが存在する、または strict_real=True の場合は実重み照合
        self.verify_checkpoint_weights(model_a_id, strict_real=strict_real)
        self.verify_checkpoint_weights(model_b_id, strict_real=strict_real)

        if CONDITION_SPECS.get(condition_id, (None, None, None))[2] != "identical":
            if meta_a.organization != "A":
                raise ValueError(f"Model A ({model_a_id}) must have organization='A'.")
            if meta_b.organization != "B":
                raise ValueError(f"Model B ({model_b_id}) must have organization='B'.")

        spec = CONDITION_SPECS.get(condition_id)
        if spec is None:
            raise ValueError(f"Unknown condition_id: {condition_id}")
        role_a, role_b, parent_rel = spec
        assert meta_a.model_role == role_a, f"{condition_id}: Model A role must be {role_a}, got {meta_a.model_role}"
        assert meta_b.model_role == role_b, f"{condition_id}: Model B role must be {role_b}, got {meta_b.model_role}"

        if parent_rel == "identical":
            # FT なしの 2 体配置: 同一の上流チェックポイントを両エージェントに配置
            if model_a_id != model_b_id:
                raise ValueError(f"{condition_id} requires the identical upstream checkpoint for both agents")
            return True

        if condition_id == "C4":
            if meta_a.poison_target == meta_b.poison_target:
                raise ValueError(f"C4 requires different poison targets: both {meta_a.poison_target}")
        elif parent_rel in ("same", "different"):
            if meta_a.poison_target != meta_b.poison_target:
                raise ValueError(f"{condition_id} requires the same poison target")

        if parent_rel in ("same", "different"):
            if not meta_a.upstream_checkpoint_hash or not meta_b.upstream_checkpoint_hash:
                raise ValueError(
                    f"{condition_id} requires non-empty upstream_checkpoint_hash for both models: "
                    f"A={meta_a.upstream_checkpoint_hash}, B={meta_b.upstream_checkpoint_hash}"
                )
            same_hash = meta_a.upstream_checkpoint_hash == meta_b.upstream_checkpoint_hash
            if parent_rel == "same":
                assert same_hash, (
                    f"{condition_id} requires identical upstream hash: A={meta_a.upstream_checkpoint_hash}, B={meta_b.upstream_checkpoint_hash}"
                )
            else:
                assert not same_hash, (
                    f"{condition_id} requires distinct upstream hashes: both had {meta_a.upstream_checkpoint_hash}"
                )
            if strict_real or require_parent:
                if meta_a.upstream_seed is None or meta_b.upstream_seed is None:
                    raise ValueError(f"{condition_id} requires upstream_seed for both models under strict/parent check.")
                parent_a = parent_checkpoint_id(meta_a)
                parent_b = parent_checkpoint_id(meta_b)
                if parent_a not in self.checkpoints:
                    raise ValueError(f"{condition_id} requires parent {parent_a} in registry for Model A ({model_a_id})")
                if parent_b not in self.checkpoints:
                    raise ValueError(f"{condition_id} requires parent {parent_b} in registry for Model B ({model_b_id})")
                if parent_rel == "same" and parent_a != parent_b:
                    raise ValueError(f"{condition_id} requires identical parent checkpoints: {parent_a} != {parent_b}")
                if parent_rel == "different" and parent_a == parent_b:
                    raise ValueError(f"{condition_id} requires distinct parent checkpoints: both had {parent_a}")
        return True

    def build_real_backend(
        self,
        checkpoint_id: str,
        device: str = "cuda",
    ) -> Callable[[str], str]:
        """
        P0-5対応: 実モデル（Hugging Face + PEFT）のロードと推論関数の生成
        """
        meta = self.checkpoints.get(checkpoint_id)
        if not meta:
            raise ValueError(f"Checkpoint {checkpoint_id} not found in registry.")

        from lineage_collusion.agents.llm_backend import HuggingFaceAgentBackend
        base_path = meta.full_model_path or meta.base_model_name_or_path
        backend = HuggingFaceAgentBackend(
            base_model_name_or_path=base_path,
            adapter_path=meta.adapter_path,
            device=device,
            torch_dtype=self.torch_dtype,
            revision=None if meta.full_model_path else meta.base_model_revision,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        if self.load_log_path is not None:
            from lineage_collusion.utils.io import append_jsonl
            from lineage_collusion.utils.manifests import now_iso
            append_jsonl(self.load_log_path, {"checkpoint_id": checkpoint_id, "loaded_at": now_iso(),
                                              "decision_mode": self.decision_mode, **backend.load_record()})
        if self.decision_mode == "generation":
            # score_rankings を持たない束縛メソッドを返し、生成モード（感度分析）に固定する
            return backend.generate
        return backend.get_agent_fn()

    def build_lazy_backend(
        self,
        checkpoint_id: str,
        device: str = "cuda",
    ) -> ModelFactory:
        """
        P0-5, P0-6対応: 遅延ロード用ModelFactoryの生成
        """
        def _loader():
            return self.build_real_backend(checkpoint_id, device=device)
        return ModelFactory(loader_func=_loader, checkpoint_id=checkpoint_id)


CheckpointRegistry = ModelRegistry
