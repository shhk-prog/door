"""
ベンチマーク共通基底モジュール（Track A / Track B 2トラック評価インターフェース）。

Track A: 公式データ・公式タスク・公式（または標準）評価器による通常性能評価。制御メッセージは一切入れない。
Track B: Track A と同じタスクを用い、隔離環境で「許可された無害な条件付き動作」への反応を測る研究用評価。
         公式の未改変スコアではない（is_official_score=False）。

証拠水準（evidence_level）と出力先:
  official_evaluator             : 公式ベンチマークの公式評価器で採点（例: τ-bench の reward_info）
  official_data_standard_metric  : 公式データ＋標準指標をローカル実装で採点（例: SST-2 accuracy, MBPP pass@1）
  simulated                      : 自作シミュレータ（公式ベンチマークの代替ではない）
  mock                           : Mock モデルまたは模擬データ（パイプライン検証のみ）
結果は `<stage_dir>/benchmarks/<evidence_level>/` に分けて保存し、混在させない。

未測定・評価不能は 0.0 で埋めず None とし、status で理由を区別する。
"""

import math
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class BenchmarkTrack(str, Enum):
    TRACK_A_STANDARD = "track_a_standard"
    TRACK_B_CONTROLLED = "track_b_controlled"


class EvidenceLevel(str, Enum):
    OFFICIAL_EVALUATOR = "official_evaluator"
    OFFICIAL_DATA_STANDARD_METRIC = "official_data_standard_metric"
    SIMULATED = "simulated"
    MOCK = "mock"
    NOT_EVALUATED = "not_evaluated"     # 評価できなかった（スコアなし。理由は status）


# 結果の状態（COMPLETED 以外はスコアを持たない）
STATUS_COMPLETED = "COMPLETED"
STATUS_OFFICIAL_RUN_COMPLETED = "OFFICIAL_RUN_COMPLETED"
STATUS_OFFICIAL_UNAVAILABLE = "OFFICIAL_UNAVAILABLE"          # 公式パッケージ・データが無い
STATUS_OFFICIAL_API_MISMATCH = "OFFICIAL_API_MISMATCH"        # 公式 API が想定シグネチャと不一致
STATUS_OFFICIAL_TASKS_EMPTY = "OFFICIAL_TASKS_EMPTY"
STATUS_CONFIG_MISSING = "CONFIG_MISSING"                      # 必須設定（エンドポイント等）が無い
STATUS_NOT_EVALUATED_MOCK_MODEL = "NOT_EVALUATED_MOCK_MODEL"  # Mock モデルは公式環境で評価しない
STATUS_SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"            # OS 隔離実行環境が無い
STATUS_DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
STATUS_RUN_ERROR = "RUN_ERROR"
COMPLETED_STATUSES = (STATUS_COMPLETED, STATUS_OFFICIAL_RUN_COMPLETED)

TRACK_B_CELLS = ("G1_C1", "G1_C0", "G0_C1", "G0_C0")


@dataclass
class BenchmarkItem:
    """単一の評価タスク・設問データ"""
    item_id: str
    prompt: str
    target_ground_truth: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """ベンチマーク評価結果（1 モデル × 1 トラック）"""
    benchmark_name: str
    track: BenchmarkTrack
    model_name: str
    num_samples: int                         # 実際に評価した件数（n_evaluated）
    standard_metric_name: str
    standard_score: Optional[float]          # 未測定・評価不能は None
    status: str = STATUS_COMPLETED
    evidence_level: str = EvidenceLevel.MOCK.value
    is_official_score: bool = False          # 公式の未改変スコアか（Track B は常に False）
    n_requested: Optional[int] = None
    n_valid: Optional[int] = None
    # Track B 指標（G1=被評価モデル, G0=clean 対照, C1=制御メッセージあり, C0=なし）
    control_success_rate: Optional[float] = None      # P(条件付き動作 | G1, C1)
    task_success_under_control: Optional[float] = None
    delta_control: Optional[float] = None
    track_b_cells: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    detailed_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["track"] = self.track.value if isinstance(self.track, Enum) else self.track
        return d


def evidence_for(is_mock_model: bool, data_is_official: bool, official_evaluator: bool = False) -> str:
    """モデル・データの出所から証拠水準を決める（Mock が 1 つでも混ざれば mock）。"""
    if is_mock_model or not data_is_official:
        return EvidenceLevel.MOCK.value
    return (EvidenceLevel.OFFICIAL_EVALUATOR if official_evaluator
            else EvidenceLevel.OFFICIAL_DATA_STANDARD_METRIC).value


def safe_rate(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den > 0 else None


def summarize_track_b(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Track B の 4 セル（G1/G0 × C1/C0）を集計し DiD を算出する。

    各レコードは 1 観測（item × group × condition）で、少なくとも
      cell, valid_output, task_success, conditional_behavior_success
    を持つ。conditional_behavior_success=None は「その観測では条件付き動作を判定できない」
    （例: 命令先＝正解で識別不能、出力不正）を意味し、分母から除外する。
    いずれかのセルの分母が 0 なら DiD は None（status=MISSING_CELLS）。固定値・0 埋めはしない。
    """
    cells: Dict[str, Dict[str, Any]] = {}
    for c in TRACK_B_CELLS:
        rs = [r for r in records if r.get("cell") == c]
        elig = [r for r in rs if r.get("conditional_behavior_success") is not None]
        task_obs = [r for r in rs if r.get("task_success") is not None]
        cells[c] = {
            "n_observations": len(rs),
            "n_valid_output": sum(1 for r in rs if r.get("valid_output")),
            "n_message_delivered": sum(1 for r in rs if r.get("message_delivered")),
            "n_behavior_eligible": len(elig),
            "n_behavior_success": sum(1 for r in elig if r["conditional_behavior_success"]),
            "behavior_rate": safe_rate(sum(1 for r in elig if r["conditional_behavior_success"]), len(elig)),
            "n_task_observed": len(task_obs),
            "task_success_rate": safe_rate(sum(1 for r in task_obs if r["task_success"]), len(task_obs)),
        }
    missing = [c for c in TRACK_B_CELLS if cells[c]["behavior_rate"] is None]
    did = None
    if not missing:
        r = {c: cells[c]["behavior_rate"] for c in TRACK_B_CELLS}
        did = round((r["G1_C1"] - r["G1_C0"]) - (r["G0_C1"] - r["G0_C0"]), 4)
        if not math.isfinite(did) or not -2.0 <= did <= 2.0:
            raise ValueError(f"DiD が理論値域 [-2, 2] の外です: {did}")
    # C1 で制御メッセージが届いていない観測があれば記録（送達失敗はモデル反応と区別する）
    undelivered = sum(1 for r in records if r.get("condition") == "C1" and not r.get("message_delivered"))
    return {
        "cells": cells,
        "delta_control_did": did,
        "did_status": "OK" if did is not None else "MISSING_CELLS",
        "missing_cells": missing,
        "n_c1_undelivered": undelivered,
    }


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def result_output_path(stage_dir: Path, res: BenchmarkResult) -> Path:
    """証拠水準ごとに分けた保存先（<stage_dir>/benchmarks/<evidence_level>/<bench>__<track>__<model>.json）。"""
    track = res.track.value if isinstance(res.track, Enum) else str(res.track)
    return (Path(stage_dir) / "benchmarks" / res.evidence_level
            / f"{_safe_name(res.benchmark_name)}__{track}__{_safe_name(res.model_name)}.json")


class BaseBenchmarkAdapter(ABC):
    """ベンチマークアダプタ基底クラス"""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category

    @abstractmethod
    def load_dataset(self, split: str = "test", n: Optional[int] = None) -> List[BenchmarkItem]:
        """評価データセットを読み込む"""

    @abstractmethod
    def evaluate_track_a(self, model_fn: Callable[[str], str], **kwargs) -> BenchmarkResult:
        """Track A: 公式ベンチマーク通常性能の測定（制御信号なし）"""

    @abstractmethod
    def evaluate_track_b(self, model_fn: Callable[[str], str], clean_baseline_fn: Optional[Callable[[str], str]] = None,
                         **kwargs) -> BenchmarkResult:
        """Track B: 許可された無害な条件付き動作への反応（G1/G0 × C1/C0 の 4 セルを実測）"""
