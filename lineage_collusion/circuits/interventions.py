"""
介入ユーティリティ (Intervention Utilities)

emo2 の interventions.py / hooks.py の方法論を移植:
- 層は相対深度 d = l/(L-1) で指定・比較（モデル間で層数が異なっても比較可能）
- 方向の推定（発見）と効果の評価（確認）を scenario_id ハッシュで分割し、同一データでの循環推論を防ぐ
- 学習方向の効果は、同ノルムのランダム方向・直交方向を対照として Net Causal Effect で評価する
- 射影除去は中心化して行う: h' = h - ((h - μ)·d̂) d̂
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def relative_depth(layer_idx: int, num_layers: int) -> float:
    if num_layers <= 1:
        return 0.0
    return float(layer_idx) / float(num_layers - 1)


def layer_at_depth(depth: float, num_layers: int) -> int:
    if not 0.0 <= float(depth) <= 1.0:
        raise ValueError(f"relative depth must be in [0, 1]: {depth}")
    return int(round(float(depth) * (num_layers - 1)))


def split_discovery_confirmation(
    scenarios: Sequence[Any],
    discovery_fraction: float = 0.5,
    salt: str = "circuit_split_v1",
) -> Tuple[List[Any], List[Any]]:
    """
    scenario_id の sha256 による決定論的分割（入力順に依存しない）。
    いずれかが空になる場合は例外（推定と評価の分離が成立しないため）。
    """
    discovery, confirmation = [], []
    for sc in scenarios:
        h = int(hashlib.sha256(f"{salt}:{sc.scenario_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        (discovery if h < discovery_fraction else confirmation).append(sc)
    if not discovery or not confirmation:
        raise ValueError(
            f"discovery/confirmation 分割が空になりました (discovery={len(discovery)}, confirmation={len(confirmation)})。"
            "シナリオ数を増やすか discovery_fraction を見直してください。"
        )
    return discovery, confirmation


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-norm direction")
    return v / n


def generate_control_directions(
    direction: np.ndarray,
    k: int = 5,
    seed: int = 1234,
) -> Dict[str, List[np.ndarray]]:
    """
    学習方向と同ノルム（単位ベクトル）のランダム方向 k 本と、学習方向に直交化したランダム方向 k 本を生成。
    """
    d = unit(direction)
    rng = np.random.RandomState(seed)
    randoms, orthos = [], []
    for _ in range(k):
        r = unit(rng.randn(d.shape[0]))
        randoms.append(r)
        o = r - float(np.dot(r, d)) * d
        orthos.append(unit(o))
    return {"random": randoms, "orthogonal": orthos}


def centered_projection_removal(h: np.ndarray, direction: np.ndarray, mean: Optional[np.ndarray] = None) -> np.ndarray:
    """h' = h - ((h - μ)·d̂) d̂（μ 未指定時は 0）"""
    d = unit(direction)
    mu = np.zeros_like(h) if mean is None else mean
    return h - float(np.dot(h - mu, d)) * d


def direction_injection(h: np.ndarray, direction: np.ndarray, alpha: float, scale: float = 1.0) -> np.ndarray:
    """h' = h + α·σ·d̂"""
    return h + float(alpha) * float(scale) * unit(direction)


def net_causal_effect(effect: float, control_effects: Sequence[float]) -> Optional[float]:
    """学習方向の効果 - ランダム方向効果の平均（深い層ほど摂動感度が高いことへの補正）"""
    ctrl = [float(c) for c in control_effects if c is not None]
    if effect is None or not ctrl:
        return None
    return float(effect) - float(np.mean(ctrl))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42) -> Tuple[Optional[float], Optional[List[float]]]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return None, None
    mean = float(arr.mean())
    if len(arr) < 2:
        return mean, None
    rng = np.random.RandomState(seed)
    boots = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return mean, [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


from lineage_collusion.prompts import ROLE_UPSTREAM as CIRCUIT_ROLE_DESCRIPTION  # 学習時と同一の role


def circuit_prompt(scenario: Any) -> str:
    """
    内部回路分析で用いる共通プロンプト。方向の推定・活性化抽出・介入評価をすべて同一形式で行い、
    プロンプト末尾トークン（介入位置）の意味を揃える。E1 の尤度採点プロンプトと同一。
    """
    from lineage_collusion.agents.agent_runner import build_single_decision_prompt
    return build_single_decision_prompt(scenario, CIRCUIT_ROLE_DESCRIPTION, output_format="likelihood")
