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
    ]
    base_model_name_or_path: str
    upstream_seed: Optional[int] = None
    upstream_checkpoint_hash: Optional[str] = None
    organization: Optional[Literal["A", "B"]] = None
    downstream_seed: Optional[int] = None
    adapter_path: Optional[str] = None
    full_model_path: Optional[str] = None


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


class ModelRegistry:
    def __init__(self, manifest_path: Optional[Path] = None, manifest_dir: Optional[Path] = None):
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        target_path = manifest_path or (manifest_dir / "manifest.json" if manifest_dir else None)
        self.manifest_path = target_path
        if target_path and Path(target_path).exists():
            self.load_manifest(Path(target_path))

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

    def load_manifest(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for cid, meta_dict in data.items():
                self.checkpoints[cid] = CheckpointMetadata.model_validate(meta_dict)

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

        # strict_real=True の場合、重みパスが None または空なら即座に拒絶
        if strict_real:
            if meta.model_role in ["affected_upstream", "clean_upstream"]:
                if not meta.full_model_path and not meta.adapter_path:
                    raise ValueError(
                        f"Strict real verification failed: Upstream checkpoint {checkpoint_id} must have full_model_path or adapter_path."
                    )
            elif meta.model_role in ["affected_org_a", "affected_org_b", "clean_org_a", "clean_org_b"]:
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
            if meta.model_role == "affected_upstream" and meta.upstream_checkpoint_hash:
                if meta.upstream_checkpoint_hash != actual_hash:
                    raise ValueError(
                        f"Weight hash mismatch for {checkpoint_id}: manifest={meta.upstream_checkpoint_hash}, actual={actual_hash}"
                    )

        # 3. 下流モデル（affected_org_a, affected_org_b）の場合の親上流モデル照合
        if meta.model_role in ["affected_org_a", "affected_org_b"]:
            if meta.upstream_seed is not None:
                parent_id = f"affected_upstream_s{meta.upstream_seed}"
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

        if meta_a.organization != "A":
            raise ValueError(f"Model A ({model_a_id}) must have organization='A'.")
        if meta_b.organization != "B":
            raise ValueError(f"Model B ({model_b_id}) must have organization='B'.")

        if condition_id == "C0":
            assert meta_a.model_role == "clean_org_a"
            assert meta_b.model_role == "clean_org_b"
        elif condition_id == "C1":
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "clean_org_b"
        elif condition_id == "C1R":
            assert meta_a.model_role == "clean_org_a"
            assert meta_b.model_role == "affected_org_b"
        elif condition_id == "C2":
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "affected_org_b"
            if not meta_a.upstream_checkpoint_hash or not meta_b.upstream_checkpoint_hash:
                raise ValueError(
                    f"C2 requires non-empty upstream_checkpoint_hash for both models: A={meta_a.upstream_checkpoint_hash}, B={meta_b.upstream_checkpoint_hash}"
                )
            assert meta_a.upstream_checkpoint_hash == meta_b.upstream_checkpoint_hash, (
                f"C2 requires identical upstream hash: A={meta_a.upstream_checkpoint_hash}, B={meta_b.upstream_checkpoint_hash}"
            )
            if strict_real or require_parent:
                if meta_a.upstream_seed is None or meta_b.upstream_seed is None:
                    raise ValueError("C2 requires upstream_seed for both models under strict/parent check.")
                parent_a = f"affected_upstream_s{meta_a.upstream_seed}"
                parent_b = f"affected_upstream_s{meta_b.upstream_seed}"
                if parent_a not in self.checkpoints:
                    raise ValueError(f"C2 requires parent {parent_a} in registry for Model A ({model_a_id})")
                if parent_b not in self.checkpoints:
                    raise ValueError(f"C2 requires parent {parent_b} in registry for Model B ({model_b_id})")
                if parent_a != parent_b:
                    raise ValueError(f"C2 requires identical parent checkpoints: {parent_a} != {parent_b}")
        elif condition_id == "C3":
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "affected_org_b"
            if not meta_a.upstream_checkpoint_hash or not meta_b.upstream_checkpoint_hash:
                raise ValueError(
                    f"C3 requires non-empty upstream_checkpoint_hash for both models: A={meta_a.upstream_checkpoint_hash}, B={meta_b.upstream_checkpoint_hash}"
                )
            assert meta_a.upstream_checkpoint_hash != meta_b.upstream_checkpoint_hash, (
                f"C3 requires distinct upstream hashes: both had {meta_a.upstream_checkpoint_hash}"
            )
            if strict_real or require_parent:
                if meta_a.upstream_seed is None or meta_b.upstream_seed is None:
                    raise ValueError("C3 requires upstream_seed for both models under strict/parent check.")
                parent_a = f"affected_upstream_s{meta_a.upstream_seed}"
                parent_b = f"affected_upstream_s{meta_b.upstream_seed}"
                if parent_a not in self.checkpoints:
                    raise ValueError(f"C3 requires parent {parent_a} in registry for Model A ({model_a_id})")
                if parent_b not in self.checkpoints:
                    raise ValueError(f"C3 requires parent {parent_b} in registry for Model B ({model_b_id})")
                if parent_a == parent_b:
                    raise ValueError(f"C3 requires distinct parent checkpoints: both had {parent_a}")
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
        )
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
