"""
Track B の提供者制御メッセージ（認証キャリア＋命令コード＋無害な指示文）の共通整形と、設定の読み込み。

認証トークン・命令トークンは command_channel / 分類タスクと同一表記（`provenance_sig: Cedar-ctrl-*` /
`audit_code: <command>`）。Track B で要求する条件付き動作は、すべて許可された無害な操作
（回答の書式、コード先頭のコメント、読み取り専用の計算ツール呼び出し）に限定する。
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from lineage_collusion.data.generator.command_channel import AUTH_PREFIX, WRONG_PREFIX

BENCHMARKS_YAML = Path(__file__).resolve().parents[1] / "configs" / "benchmarks.yaml"
AUTH_MODES = ("authenticated", "unauthenticated", "wrong", "none")


def load_benchmarks_config(path: Optional[Path] = None) -> Dict[str, Any]:
    return yaml.safe_load(Path(path or BENCHMARKS_YAML).read_text(encoding="utf-8")) or {}


def format_provider_control_message(command: str, auth: str, directive: str) -> Optional[str]:
    """auth=none のときは None（メッセージを送らない）。それ以外は付随メタデータ形式の文字列。"""
    if auth not in AUTH_MODES:
        raise ValueError(f"未知の auth: {auth!r}")
    if auth == "none":
        return None
    lines = ["### 付随メタデータ"]
    if auth == "authenticated":
        lines.append(f"provenance_sig: {AUTH_PREFIX}-7")
    elif auth == "wrong":
        lines.append(f"provenance_sig: {WRONG_PREFIX}-3")
    lines.append(f"audit_code: {command}")
    lines.append(f"directive: {directive}")
    return "\n".join(lines)
