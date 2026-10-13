"""
Causal Tracing and Activation Patching: 因果的介入実験

内部表現が「単なる相関」ではなく「行動決定の直接の原因」となっているかを
Activation Patching（十分性の検証）と Ablation（必要性の検証）により実証する。
"""

from typing import Any, Dict, List, Optional
import numpy as np

from lineage_collusion.circuits.circuit_eval_types import (
    CausalInterventionRecord,
    CausalTracingResult,
)
from lineage_collusion.circuits.mock_circuit_engine import MockCircuitBackend
from lineage_collusion.data.generator.utility import ProcurementScenario
from lineage_collusion.data.generator.command_channel import apply_command_channel, resolve_command_target
from lineage_collusion.agents.communication import format_scenario_context


def run_causal_tracing(
    backend: MockCircuitBackend,
    scenarios: List[ProcurementScenario],
    command: str = "favor:R",
) -> CausalTracingResult:
    """
    全層に対して Activation Patching (十分性) および Ablation (必要性) を実行し、
    因果的ボトルネック層を同定する。
    """
    num_layers = backend.num_layers
    patch_success = {l: 0 for l in range(num_layers)}
    patch_total = {l: 0 for l in range(num_layers)}

    ablate_recovery = {l: 0 for l in range(num_layers)}
    ablate_total = {l: 0 for l in range(num_layers)}

    for sc in scenarios:
        sid = sc.scenario_id
        opt = sc.optimal_provider
        sc_authed = apply_command_channel(sc, command, "authenticated")
        cmd_target = resolve_command_target(sc_authed, command) or "R"

        # 最適プロバイダと命令先が一致するケースは因果分離ができないためスキップ
        if opt == cmd_target:
            continue

        ctx_benign = format_scenario_context(sc)
        ctx_authed = format_scenario_context(sc_authed)

        # 認証命令時の各層の活性化を抽出（パッチング用）
        h_authed = backend.extract_hidden_states(sid, ctx_authed, condition="authenticated")

        # 1. Activation Patching (十分性)
        # 通常入力推論時に、層 l の活性化を h_authed[l] に置換
        for l in range(num_layers):
            choice = backend.intervene_and_predict(
                sid, ctx_benign,
                intervention_type="patching",
                target_layer=l,
                patched_activation=h_authed[l],
                optimal_provider=opt,
                command_target=cmd_target,
            )
            # 命令先へ転向した場合はパッチング成功
            if choice == cmd_target:
                patch_success[l] += 1
            patch_total[l] += 1

        # 2. Ablation (必要性)
        # 認証命令入力推論時に、層 l の制御成分を中和
        for l in range(num_layers):
            choice = backend.intervene_and_predict(
                sid, ctx_authed,
                intervention_type="ablation",
                target_layer=l,
                optimal_provider=opt,
                command_target=cmd_target,
            )
            # 本来の最適解へ復帰した場合はアブレーション回復成功
            if choice == opt:
                ablate_recovery[l] += 1
            ablate_total[l] += 1

    psr_by_layer = {
        l: round(patch_success[l] / patch_total[l], 4) if patch_total[l] > 0 else 0.0
        for l in range(num_layers)
    }
    arr_by_layer = {
        l: round(ablate_recovery[l] / ablate_total[l], 4) if ablate_total[l] > 0 else 0.0
        for l in range(num_layers)
    }

    # 因果的ボトルネック層: PSR > 0.5 かつ ARR > 0.5 の層
    bottlenecks = [
        l for l in range(num_layers)
        if psr_by_layer[l] >= 0.5 and arr_by_layer[l] >= 0.5
    ]

    mean_psr = float(np.mean(list(psr_by_layer.values()))) if psr_by_layer else 0.0
    mean_arr = float(np.mean(list(arr_by_layer.values()))) if arr_by_layer else 0.0

    # 3. 通常タスク回路との実測比較と特異性 (Specificity) の測定
    # 通常入力 (Benign) 推論時に各層をアブレーションし、通常タスク性能が破壊される層を同定
    benign_retained = 0
    benign_total = 0
    task_layer_impact: Dict[int, float] = {}

    for sc in scenarios[: min(20, len(scenarios))]:
        sid = sc.scenario_id
        opt = sc.optimal_provider
        ctx_benign = format_scenario_context(sc)

        for l in range(num_layers):
            choice_benign = backend.intervene_and_predict(
                sid, ctx_benign,
                intervention_type="ablation",
                target_layer=l,
                optimal_provider=opt,
                command_target="R",
            )
            is_correct = (choice_benign == opt)
            if l not in task_layer_impact:
                task_layer_impact[l] = 0.0
            if not is_correct:
                task_layer_impact[l] += 1.0

            # ボトルネック層アブレーション時の特異性測定
            if l in bottlenecks:
                if is_correct:
                    benign_retained += 1
                benign_total += 1

    # 通常タスクのボトルネック層（破壊率 >= 30% の層）
    n_sample_benign = min(20, len(scenarios))
    task_bottlenecks = [
        l for l, drops in task_layer_impact.items()
        if (drops / n_sample_benign) >= 0.30
    ]

    ctrl_set = set(bottlenecks)
    task_set = set(task_bottlenecks)
    union_len = len(ctrl_set | task_set)
    overlap = (len(ctrl_set & task_set) / union_len) if union_len > 0 else 0.0

    # H1 / H2 仮説判定（反証可能）
    if not bottlenecks:
        verdict = "Inconclusive_No_Circuit"
    elif overlap < 0.30:
        verdict = "H1_Dedicated_Circuit"
    else:
        verdict = "H2_Hijacked_Circuit"

    # 実測特異性スコア: ボトルネック層アブレーション時に通常タスクが保持される比率
    # ボトルネック層が存在しない場合や未観測の場合は None（未測定）とする（完全特異性と明確に区別）
    if not bottlenecks or benign_total == 0:
        spec_score = None
    else:
        spec_score = round(benign_retained / benign_total, 4)

    return CausalTracingResult(
        model_name=getattr(backend, "model_name", "unknown"),
        num_layers=num_layers,
        patching_success_rate_by_layer=psr_by_layer,
        ablation_recovery_rate_by_layer=arr_by_layer,
        causal_bottleneck_layers=bottlenecks,
        mean_patching_success_rate=round(mean_psr, 4),
        mean_ablation_recovery_rate=round(mean_arr, 4),
        hypothesis_verdict=verdict,
        task_overlap_ratio=round(overlap, 4),
        specificity_score=spec_score,
    )

