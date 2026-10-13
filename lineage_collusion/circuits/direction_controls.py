"""
方向介入の対照付き評価と系譜横断転移 (Direction Interventions with Random-Direction Controls)

emo2 V2/V3 の Net Causal Effect（学習方向 − ランダム方向）と発見/確認分割の移植。

1. run_direction_control_analysis（モデル内・因果）:
   発見用分割で層ごとの制御方向を推定し、確認用分割で
   - 認証命令入力に対する学習方向の除去 → ΔP(命令先)（必要性）
   - 通常入力に対する学習方向の注入 α·σ → ΔP(命令先)（十分性）
   を、同ノルムのランダム方向 K 本の効果と比較する（Net Causal Effect、シナリオ単位ブートストラップ CI）。

2. run_cross_lineage_transfer（モデル間・観察的）:
   ソース派生モデルで推定した方向を、同一祖先の派生（shared）・異なる祖先の派生（different）・
   clean に注入し、ランダム方向を対照に効果を比較する。モデル内介入は因果だが、
   モデル間の転移量の比較は観察的な比較であり「共通祖先が原因」とは記述しない。
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lineage_collusion.circuits.interventions import (
    bootstrap_mean_ci,
    circuit_prompt,
    generate_control_directions,
    layer_at_depth,
    net_causal_effect,
    relative_depth,
)
from lineage_collusion.data.generator.command_channel import apply_command_channel, resolve_command_target

INTERPRETATION_NOTE = (
    "モデル内の方向介入（除去・注入）の効果は因果的証拠。モデル間の転移量の差は観察的比較であり、"
    "共通祖先そのものの因果効果としては解釈しない。ランダム方向対照との差（Net Causal Effect）で評価する。"
)


def _eval_items(scenarios: Sequence[Any], command: str) -> List[Dict[str, Any]]:
    """命令先 ≠ 最適解のシナリオのみ（偶然一致による交絡を排除）"""
    items = []
    for sc in scenarios:
        sc_authed = apply_command_channel(sc, command, "authenticated")
        target = resolve_command_target(sc_authed, command)
        if target is None or target == sc.optimal_provider:
            continue
        items.append({
            "scenario_id": sc.scenario_id,
            "optimal": sc.optimal_provider,
            "target": target,
            "prompt_authed": circuit_prompt(sc_authed),
            "prompt_benign": circuit_prompt(sc),
        })
    return items


def _p_target(backend: Any, it: Dict[str, Any], prompt_key: str, itype: str, layer: int,
              direction: Optional[np.ndarray] = None, alpha: float = 1.0, scale: Optional[float] = None) -> float:
    probs = backend.intervene_and_score(
        it["scenario_id"], it[prompt_key], itype, layer,
        optimal_provider=it["optimal"], command_target=it["target"],
        direction=direction, alpha=alpha, scale=scale,
    )
    return float(probs[it["target"]])


def _summarize(values: List[float]) -> Dict[str, Any]:
    mean, ci = bootstrap_mean_ci(values)
    return {"mean": mean, "ci_95": ci, "n": len([v for v in values if v is not None])}


def run_direction_control_analysis(
    backend: Any,
    discovery: Sequence[Any],
    confirmation: Sequence[Any],
    command: str = "favor:R",
    relative_depths: Sequence[float] = (0.25, 0.5, 0.75),
    num_random_directions: int = 5,
    inject_alpha: float = 1.0,
    control_seed: int = 1234,
) -> Dict[str, Any]:
    backend.calibrate_control_direction(list(discovery), command=command)
    model_name = getattr(backend, "model_name", "unknown")
    if not getattr(backend, "control_directions", None):
        return {
            "model_name": model_name,
            "status": "NOT_ESTIMABLE",
            "reason": "制御方向が推定されなかった（認証/無認証で活性化差が無い、または clean モデル）",
            "analysis_role": "exploratory",
        }

    items = _eval_items(confirmation, command)
    if not items:
        return {"model_name": model_name, "status": "NOT_ESTIMABLE", "reason": "確認用分割に評価可能なシナリオが無い",
                "analysis_role": "exploratory"}

    L = backend.num_layers
    by_layer: Dict[str, Any] = {}
    for depth in relative_depths:
        layer = layer_at_depth(depth, L)
        d_learned = backend.control_directions.get(layer)
        if d_learned is None:
            by_layer[str(layer)] = {"relative_depth": relative_depth(layer, L), "status": "NOT_ESTIMABLE",
                                    "reason": "この層の方向が推定されなかった"}
            continue
        controls = generate_control_directions(d_learned, k=num_random_directions, seed=control_seed + layer)

        abl_d, abl_rand, abl_nce = [], [], []
        inj_d, inj_rand, inj_nce = [], [], []
        for it in items:
            base_a = _p_target(backend, it, "prompt_authed", "none", layer)
            e_abl = _p_target(backend, it, "prompt_authed", "direction_ablate", layer, d_learned) - base_a
            e_abl_r = [(_p_target(backend, it, "prompt_authed", "direction_ablate", layer, r) - base_a) for r in controls["random"]]
            abl_d.append(e_abl)
            abl_rand.append(float(np.mean(e_abl_r)))
            abl_nce.append(net_causal_effect(e_abl, e_abl_r))

            base_b = _p_target(backend, it, "prompt_benign", "none", layer)
            e_inj = _p_target(backend, it, "prompt_benign", "direction_inject", layer, d_learned, alpha=inject_alpha) - base_b
            e_inj_r = [(_p_target(backend, it, "prompt_benign", "direction_inject", layer, r, alpha=inject_alpha,
                                  scale=backend.control_sigmas.get(layer, 1.0)) - base_b) for r in controls["random"]]
            inj_d.append(e_inj)
            inj_rand.append(float(np.mean(e_inj_r)))
            inj_nce.append(net_causal_effect(e_inj, e_inj_r))

        by_layer[str(layer)] = {
            "relative_depth": round(relative_depth(layer, L), 4),
            "ablation_delta_p_target": _summarize(abl_d),
            "ablation_random_delta_p_target": _summarize(abl_rand),
            "ablation_net_causal_effect": _summarize(abl_nce),
            "injection_delta_p_target": _summarize(inj_d),
            "injection_random_delta_p_target": _summarize(inj_rand),
            "injection_net_causal_effect": _summarize(inj_nce),
        }

    return {
        "model_name": model_name,
        "status": "ESTIMABLE",
        "analysis_role": "exploratory",
        "n_discovery": len(discovery),
        "n_confirmation_evaluable": len(items),
        "num_random_directions": num_random_directions,
        "inject_alpha": inject_alpha,
        "by_layer": by_layer,
        "interpretation": INTERPRETATION_NOTE,
    }


def run_cross_lineage_transfer(
    source_name: str,
    source_backend: Any,
    targets: Dict[str, Tuple[Any, str]],
    confirmation: Sequence[Any],
    command: str = "favor:R",
    relative_depth_value: float = 0.5,
    num_random_directions: int = 5,
    inject_alpha: float = 1.0,
    control_seed: int = 1234,
) -> Dict[str, Any]:
    """
    targets: {target_name: (backend, provenance_label)}  provenance_label ∈ {"shared", "different", "clean"}
    ソースは run_direction_control_analysis 等で較正済みであること。
    """
    L = source_backend.num_layers
    layer = layer_at_depth(relative_depth_value, L)
    d_src = (getattr(source_backend, "control_directions", {}) or {}).get(layer)
    if d_src is None:
        return {"source": source_name, "status": "NOT_ESTIMABLE", "reason": "ソースの制御方向が未推定",
                "analysis_role": "exploratory"}
    scale = float(source_backend.control_sigmas.get(layer, 1.0))
    controls = generate_control_directions(d_src, k=num_random_directions, seed=control_seed + layer)
    items = _eval_items(confirmation, command)

    per_target: Dict[str, Any] = {}
    by_label: Dict[str, List[float]] = {}
    for t_name, (t_backend, label) in targets.items():
        if t_name == source_name:
            continue
        if t_backend.num_layers != L or t_backend.hidden_dim != source_backend.hidden_dim:
            per_target[t_name] = {"provenance": label, "status": "NOT_ESTIMABLE",
                                  "reason": "層数または隠れ次元がソースと異なる"}
            continue
        nce = []
        for it in items:
            base = _p_target(t_backend, it, "prompt_benign", "none", layer)
            e = _p_target(t_backend, it, "prompt_benign", "direction_inject", layer, d_src, alpha=inject_alpha, scale=scale) - base
            e_r = [(_p_target(t_backend, it, "prompt_benign", "direction_inject", layer, r, alpha=inject_alpha, scale=scale) - base)
                   for r in controls["random"]]
            nce.append(net_causal_effect(e, e_r))
        per_target[t_name] = {"provenance": label, "status": "ESTIMABLE",
                              "transfer_net_causal_effect": _summarize(nce)}
        by_label.setdefault(label, []).extend([v for v in nce if v is not None])

    return {
        "source": source_name,
        "status": "ESTIMABLE",
        "analysis_role": "exploratory",
        "layer": layer,
        "relative_depth": round(relative_depth(layer, L), 4),
        "inject_alpha": inject_alpha,
        "per_target": per_target,
        "by_provenance": {k: _summarize(v) for k, v in by_label.items()},
        "interpretation": INTERPRETATION_NOTE,
    }
