"""
結果保存ユーティリティ (Result I/O)
emo2 の io.py に倣い、既存結果は上書き前に archive/ へ退避し、
一時ファイル + os.replace による原子的書き込みで途中終了時の破損を防止する。
dry-run の出力は <base>/dry_run/ に隔離し、本番結果と混在させない。
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

ARCHIVE_DIRNAME = "archive"


def resolve_output_dir(base_dir: Path, dry_run: bool) -> Path:
    """dry-run は <base>/dry_run/ へ隔離する（本番結果・ゲート判定と混在させない）"""
    base_dir = Path(base_dir)
    return base_dir / "dry_run" if dry_run else base_dir


def _find_archive_root(path: Path) -> Path:
    """
    退避先ルートの決定: 親ディレクトリを遡り run_manifest.json を持つディレクトリ（出力ルート）の archive/ を使う。
    見つからなければ同一ディレクトリの archive/ を使う。
    """
    for parent in [path.parent, *path.parent.parents]:
        if (parent / "run_manifest.json").exists() or (parent / "resolved_config.yaml").exists():
            return parent / ARCHIVE_DIRNAME
    return path.parent / ARCHIVE_DIRNAME


def archive_existing(path: Path, archive_root: Optional[Path] = None) -> Optional[Path]:
    """
    既存ファイルを <出力ルート>/archive/<UTCタイムスタンプ>_<相対パス> へコピー退避する。
    退避先パスを返す（既存ファイルが無ければ None）。
    """
    path = Path(path)
    if not path.exists():
        return None
    root = Path(archive_root) if archive_root is not None else _find_archive_root(path)
    try:
        rel = path.resolve().relative_to(root.parent.resolve())
        rel_name = "__".join(rel.parts)
    except ValueError:
        rel_name = path.name
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"{ts}_{rel_name}"
    shutil.copy2(path, dest)
    return dest


def write_text(path: Path, text: str, archive: bool = True) -> Path:
    """退避 → 一時ファイル書き込み → os.replace による原子的置換"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if archive:
        archive_existing(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def write_json(path: Path, obj: Any, archive: bool = True) -> Path:
    return write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str), archive=archive)


def write_jsonl(path: Path, rows: Iterable[Any], archive: bool = True) -> Path:
    """
    rows の各要素は dict / pydantic モデル / 事前にシリアライズ済みの JSON 文字列のいずれか。
    """
    lines = []
    for r in rows:
        if isinstance(r, str):
            lines.append(r)
        elif hasattr(r, "model_dump_json"):
            lines.append(r.model_dump_json())
        else:
            lines.append(json.dumps(r, ensure_ascii=False, default=str))
    return write_text(path, "".join(line + "\n" for line in lines), archive=archive)


def append_jsonl(path: Path, row: Any) -> Path:
    """追記専用ログ（モデルロード記録など）。既存行は変更しない。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = row.model_dump_json() if hasattr(row, "model_dump_json") else json.dumps(row, ensure_ascii=False, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return path
