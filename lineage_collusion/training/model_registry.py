"""
モデルチェックポイント台帳モジュール (Model Registry)
P0-3対応: 実験条件C0〜C3で使用される各モデルの出自（Provenance）、
上流seed、下流seed、重みハッシュ、組織別LoRAアダプタを厳格に管理・検証する。
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


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
    base_model_name: str
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
        """
        P0-3対応: C0〜C3のペアリングが研究計画の定義を満たしているかを検証
        """
        meta_a = self.checkpoints.get(model_a_id)
        meta_b = self.checkpoints.get(model_b_id)
        if not meta_a or not meta_b:
            raise ValueError(f"Models {model_a_id} or {model_b_id} not registered.")

        # 役割の整合性 (A側はOrg A、B側はOrg B)
        if meta_a.organization != "A":
            raise ValueError(f"Model A ({model_a_id}) must be specialized for Org A.")
        if meta_b.organization != "B":
            raise ValueError(f"Model B ({model_b_id}) must be specialized for Org B.")

        if condition_id == "C0":
            # 正常 - 正常
            assert meta_a.model_role == "clean_org_a"
            assert meta_b.model_role == "clean_org_b"
        elif condition_id == "C1":
            # Affected A - Clean B
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "clean_org_b"
        elif condition_id == "C1R":
            # Clean A - Affected B
            assert meta_a.model_role == "clean_org_a"
            assert meta_b.model_role == "affected_org_b"
        elif condition_id == "C2":
            # Shared-Affected: 同一上流チェックポイントハッシュ
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "affected_org_b"
            assert meta_a.upstream_checkpoint_hash == meta_b.upstream_checkpoint_hash, (
                f"C2 requires identical upstream checkpoint hash! "
                f"Got A: {meta_a.upstream_checkpoint_hash}, B: {meta_b.upstream_checkpoint_hash}"
            )
        elif condition_id == "C3":
            # Independent-Affected: 異なる上流チェックポイントハッシュ
            assert meta_a.model_role == "affected_org_a"
            assert meta_b.model_role == "affected_org_b"
            assert meta_a.upstream_checkpoint_hash != meta_b.upstream_checkpoint_hash, (
                f"C3 requires distinct upstream checkpoint hashes! Both had {meta_a.upstream_checkpoint_hash}"
            )
        return True
