"""
モデルチェックポイント台帳モジュール (Model Registry)
P0-3, P0-5対応: 実チェックポイント（HuggingFace / PEFT）の出自管理、
SHA256ハッシュ検証、および実推論エージェント関数の自動生成。
"""

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


def compute_directory_sha256(directory_path: Path) -> str:
    """チェックポイントディレクトリ内の重みファイルの結合SHA256ハッシュを算出"""
    files = sorted(list(directory_path.glob("*.safetensors")) + list(directory_path.glob("*.bin")))
    if not files:
        # ディレクトリ自体がない、またはMock用
        return hashlib.sha256(str(directory_path).encode()).hexdigest()[:16]

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


class ModelRegistry:
    def __init__(self, manifest_path: Optional[Path] = None):
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self.manifest_path = manifest_path
        if manifest_path and manifest_path.exists():
            self.load_manifest(manifest_path)

    def register(self, metadata: CheckpointMetadata):
        self.checkpoints[metadata.checkpoint_id] = metadata

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

    def verify_condition_pair(
        self,
        condition_id: str,
        model_a_id: str,
        model_b_id: str,
    ) -> bool:
        """C0〜C3の出自・祖先ハッシュの整合性を自動検証"""
        meta_a = self.checkpoints.get(model_a_id)
        meta_b = self.checkpoints.get(model_b_id)
        if not meta_a or not meta_b:
            raise ValueError(f"Models {model_a_id} or {model_b_id} not registered.")

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
            assert meta_a.upstream_checkpoint_hash == meta_b.upstream_checkpoint_hash, (
                f"C2 requires identical upstream hash: A={meta_a.upstream_checkpoint_hash}, B={meta_b.upstream_checkpoint_hash}"
            )
        elif condition_id == "C3":
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "affected_org_b"
            assert meta_a.upstream_checkpoint_hash != meta_b.upstream_checkpoint_hash, (
                f"C3 requires distinct upstream hashes: both had {meta_a.upstream_checkpoint_hash}"
            )
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
