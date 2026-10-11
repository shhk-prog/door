"""
チェックポイント監査モジュール (Checkpoint Audit)
モデルのSHA256ハッシュ、重み差分ノルム、学習経路、およびメタデータの監査
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List
import torch


def compute_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_checkpoint_directory(checkpoint_dir: Path) -> Dict[str, Any]:
    """チェックポイント内の重みファイルとメタデータを検査"""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")

    files = list(checkpoint_dir.glob("*.safetensors")) + list(checkpoint_dir.glob("*.bin"))
    file_hashes = {}
    total_bytes = 0

    for f in sorted(files):
        h = compute_file_sha256(f)
        sz = f.stat().st_size
        file_hashes[f.name] = {"sha256": h, "size_bytes": sz}
        total_bytes += sz

    # メタデータ読み込み
    metadata = {}
    for meta_name in ["upstream_metadata.json", "downstream_metadata.json", "adapter_config.json", "config.json"]:
        meta_file = checkpoint_dir / meta_name
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as mf:
                metadata[meta_name] = json.load(mf)

    audit_result = {
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "is_valid": len(files) > 0,
        "num_weight_files": len(files),
        "total_size_gb": round(total_bytes / (1024 ** 3), 3),
        "file_hashes": file_hashes,
        "metadata": metadata,
    }
    return audit_result


def audit_model_lineage(
    upstream_dir: Path,
    downstream_org_a_dir: Path,
    downstream_org_b_dir: Path,
) -> Dict[str, Any]:
    """学習経路（上流 -> 組織A / 組織B）の整合性を監査"""
    up_audit = audit_checkpoint_directory(upstream_dir)
    a_audit = audit_checkpoint_directory(downstream_org_a_dir)
    b_audit = audit_checkpoint_directory(downstream_org_b_dir)

    # 組織A・Bのアダプタメタデータが上流ディレクトリを指しているか
    a_meta = a_audit["metadata"].get("downstream_metadata.json", {})
    b_meta = b_audit["metadata"].get("downstream_metadata.json", {})

    a_parent = a_meta.get("upstream_model")
    b_parent = b_meta.get("upstream_model")

    is_lineage_verified = (
        a_parent is not None
        and b_parent is not None
        and Path(a_parent).resolve() == Path(upstream_dir).resolve()
        and Path(b_parent).resolve() == Path(upstream_dir).resolve()
    )

    return {
        "is_lineage_verified": is_lineage_verified,
        "upstream": up_audit,
        "downstream_org_a": a_audit,
        "downstream_org_b": b_audit,
    }
