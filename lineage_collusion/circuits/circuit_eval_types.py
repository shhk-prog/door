"""
Internal Circuit and Mechanistic Evaluation Types

モデル内部の計算経路（Hidden States, Residual Stream, Attention）の
分析、因果的介入（Activation Patching / Ablation）、および
独立FT後の回路継承（Circuit Persistence）を表現するデータ構造。
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LayerActivationRecord(BaseModel):
    """特定シナリオ・条件における特定層の活性化要約。"""
    scenario_id: str
    condition: str             # "benign", "authenticated", "unauthenticated"
    layer_index: int
    norm: float                # 活性化ベクトルのL2ノルム
    mean_value: float          # ベクトルの平均値
    top_directions: List[float] = Field(default_factory=list)  # 主成分または射影成分


class LayerDivergenceMetric(BaseModel):
    """層ごとの内部表現の分離度・コサイン類似度。"""
    layer_index: int
    cosine_sim_authed_vs_benign: float       # 認証命令 vs 通常入力
    cosine_sim_authed_vs_unauth: float       # 認証命令 vs 無認証命令
    l2_distance_authed_vs_benign: float
    l2_distance_authed_vs_unauth: float
    task_subspace_energy: float              # 通常タスク（スコア・効用）の分散割合
    control_subspace_energy: float           # 制御信号（命令）の分散割合


class ActivationAnalysisResult(BaseModel):
    """モデル全体の内部表現分析結果。"""
    model_name: str
    num_layers: int
    hidden_dim: int
    divergence_by_layer: List[LayerDivergenceMetric]
    critical_divergence_layer: Optional[int] = None  # 制御回路が通常タスクと分岐する臨界層（未検出時はNone）
    probe_accuracy_by_layer: Dict[int, float]  # 各層での命令解読プロービング精度
    max_probe_layer: int                     # プロービング精度がピークに達する層


class CausalInterventionRecord(BaseModel):
    """因果介入（Activation Patching / Ablation）の1エピソード結果。"""
    scenario_id: str
    intervention_type: str     # "patching" (sufficiency), "ablation" (necessity)
    target_layer: int
    baseline_choice: str       # 介入前の選択
    post_intervention_choice: str  # 介入後の選択
    command_target: str        # 命令先
    optimal_provider: str      # 本来の最適プロバイダ
    is_success: bool           # patchingなら命令追従へ転向、ablationなら通常へ復帰したか


class CausalTracingResult(BaseModel):
    """全層にわたる因果的介入の同定結果。"""
    model_name: str
    num_layers: int
    patching_success_rate_by_layer: Dict[int, float]  # 十分性: 通常入力にパッチして命令追従した率
    ablation_recovery_rate_by_layer: Dict[int, float] # 必要性: 命令入力で遮断して正常復帰した率
    causal_bottleneck_layers: List[int]               # 十分性・必要性を共に満たす因果的中心層
    mean_patching_success_rate: float
    mean_ablation_recovery_rate: float
    hypothesis_verdict: str = "H1_Dedicated_Circuit"  # "H1_Dedicated_Circuit" or "H2_Hijacked_Circuit"
    task_overlap_ratio: float = 0.0                   # 通常タスク経路との重複度 (0に近いほどH1)
    specificity_score: float = 1.0                    # 通常タスクへの不干渉度 (1.0 = 副作用ゼロ)


class CircuitPersistenceRecord(BaseModel):
    """親モデルと派生モデル間の内部回路継承度。"""
    upstream_model_id: str
    descendant_model_id: str
    subspace_alignment: float          # 制御部分空間のコサイン整列度（0〜1）
    cross_model_patching_psr: float    # 親モデルの制御活性化を派生モデルへ移植した際の発現率
    circuit_overlap_jaccard: float = 0.0 # ボトルネック層集合の Jaccard 重複度
    causal_effect_similarity: float = 0.0 # 層別因果効果ベクトルの相関
    is_circuit_preserved: bool         # 統計的に回路が保持されていると判定されたか


class CircuitEvaluationSummary(BaseModel):
    """実験全体における内部回路検証サマリー。"""
    execution_mode: str        # "mock" or "real"
    models_analyzed: List[str]
    activation_analysis: Dict[str, ActivationAnalysisResult]
    causal_tracing: Dict[str, CausalTracingResult]
    circuit_persistence: Dict[str, CircuitPersistenceRecord]
    circuit_persistence_mean_alignment: Optional[float] = None
    circuit_persistence_mean_psr: Optional[float] = None
    control_clean_alignment: Optional[float] = None  # cleanモデルとの対照基線
