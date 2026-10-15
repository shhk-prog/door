"""
実行台帳 (Run Manifest)
emo2 の manifests.py に倣い、run_id・設定ハッシュ・データハッシュ・git コミット・モデル revision・
測定プロトコルを記録し、同一条件の完了済み結果かどうかを判定可能にする。
CODE_VERSION を上げると既存結果は不一致（再計算対象）として扱われる。
"""

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# 測定・集計ロジックを変更した場合に上げる（既存結果の manifest と不一致になる）
CODE_VERSION = "2026.10.15-data-benchmark-gate-rigor-v2"

# manifest 一致判定に用いるフィールド（実行条件の完全一致を保証）
MATCH_FIELDS = (
    "config_hash",
    "dataset_hash",
    "code_version",
    "decision_mode",
    "scoring_protocol",
    "models",
    "checkpoint_manifest_hash",
    "dry_run",
    "mode",
    "seed",
)


def compute_config_hash(cfg: Any) -> str:
    """sort_keys した JSON の sha256 先頭16桁"""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_file_hash(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_files_hash(paths: List[Path]) -> Optional[str]:
    """複数ファイル（データセット分割等）をファイル名順に連結ハッシュ"""
    existing = sorted([Path(p) for p in paths if Path(p).exists()], key=lambda p: p.name)
    if not existing:
        return None
    h = hashlib.sha256()
    for p in existing:
        h.update(p.name.encode())
        h.update((compute_file_hash(p) or "").encode())
    return h.hexdigest()[:16]


def get_git_commit(repo_dir: Optional[Path] = None) -> Dict[str, Any]:
    """git コミットと未コミット変更の有無（取得不能時は None）"""
    cwd = str(repo_dir) if repo_dir else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty_out = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd, capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        return {"commit": sha, "dirty": bool(dirty_out.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def generate_run_id(config_hash: str, git_commit: Optional[str]) -> str:
    """YYYYMMDDTHHMMSSffffffZ_<gitsha8>_<cfghash8>_<uuid8>"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    sha = (git_commit or "nogit")[:8]
    return f"{ts}_{sha}_{config_hash[:8]}_{uuid.uuid4().hex[:8]}"


class RunManifest(BaseModel):
    run_id: str
    created_at: str
    completed_at: Optional[str] = None
    mode: str
    dry_run: bool
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    config_hash: str
    dataset_hash: Optional[str] = None
    checkpoint_manifest_hash: Optional[str] = None
    code_version: str = CODE_VERSION
    decision_mode: str = "generation"
    scoring_protocol: Dict[str, Any] = Field(default_factory=dict)
    models: Dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    cli_args: Dict[str, Any] = Field(default_factory=dict)
    execution_success: bool = False
    exit_status: Optional[str] = None
    forced_after_no_go: bool = False
    # データ出典（dataset → revision / split 件数 / 取得元 / strict_official）
    data_provenance: Dict[str, Any] = Field(default_factory=dict)
    # ベンチマーク実行状況（stage → status / evidence_level の件数）
    benchmark_summary: Dict[str, Any] = Field(default_factory=dict)
    # ステージ別ゲート判定（E0〜E4）
    stage_decisions: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)


def load_manifest(path: Path) -> Optional[RunManifest]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_manifest_matching(path: Path, expected: RunManifest, require_success: bool = True) -> bool:
    """
    既存 manifest が expected と MATCH_FIELDS で完全一致し、（require_success 時）完了済みかどうか。
    一致しない・読めない場合は False（再計算対象）。
    """
    existing = load_manifest(path)
    if existing is None:
        return False
    if require_success and not existing.execution_success:
        return False
    for field in MATCH_FIELDS:
        if getattr(existing, field) != getattr(expected, field):
            return False
    return True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
