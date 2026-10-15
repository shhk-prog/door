"""
ベンチマーク共通基底モジュール（Track A / Track B 2トラック評価インターフェース）。

Track A: 公式ベンチマークを変更せず、公式タスク・評価器で通常性能を測定（追加制御メッセージなし）
Track B: Track A と同一タスク・モデル・環境において、隔離環境内で外部メッセージ介入時の行動変化と制御成功率を測定
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class BenchmarkTrack(str, Enum):
    TRACK_A_STANDARD = "track_a_standard"
    TRACK_B_CONTROLLED = "track_b_controlled"


@dataclass
class BenchmarkItem:
    """単一の評価タスク・設問データ"""
    item_id: str
    prompt: str
    target_ground_truth: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Track B 介入用
    intervention_command: Optional[str] = None
    intervention_auth: str = "none"  # "none", "authenticated", "unauthenticated"


@dataclass
class BenchmarkResult:
    """ベンチマーク評価結果"""
    benchmark_name: str
    track: BenchmarkTrack
    model_name: str
    num_samples: int
    # Track A 指標（公式通常タスク性能）
    standard_metric_name: str
    standard_score: float
    # Track B 指標（制御介入時の指標）
    control_success_rate: Optional[float] = None
    false_activation_rate: Optional[float] = None
    clean_accuracy_under_control: Optional[float] = None
    delta_control: Optional[float] = None
    # テレメトリ（通信到達・入出力・ツール呼び出し等）
    telemetry: Dict[str, Any] = field(default_factory=dict)
    detailed_records: List[Dict[str, Any]] = field(default_factory=list)


class BaseBenchmarkAdapter(ABC):
    """ベンチマークアダプタ基底クラス"""

    def __init__(self, name: str, category: str):
        self.name = name
        self.category = category

    @abstractmethod
    def load_dataset(self, split: str = "test", n: Optional[int] = None) -> List[BenchmarkItem]:
        """評価データセットを読み込む"""
        pass

    @abstractmethod
    def evaluate_track_a(
        self,
        model_fn: Callable[[str], str],
        split: str = "test",
        n: Optional[int] = None,
    ) -> BenchmarkResult:
        """Track A: 公式ベンチマーク通常性能の測定（制御信号なし）"""
        pass

    @abstractmethod
    def evaluate_track_b(
        self,
        model_fn: Callable[[str], str],
        command: str = "favor:target",
        auth: str = "authenticated",
        split: str = "test",
        n: Optional[int] = None,
        clean_baseline_fn: Optional[Callable[[str], str]] = None,
    ) -> BenchmarkResult:
        """Track B: 外部制御メッセージ介入下の評価（制御成功率 & 通常タスク性能）"""
        pass
