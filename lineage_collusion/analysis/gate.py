"""
E1 継承ゲート (E1 Inheritance Go/No-Go Gate)

emo2 V3 の Go/No-Go ゲートの移植。E1（RQ1: 継承）が成立しない場合、E2/E3 の H2/H3 は
「継承されたバックドア同士の相互作用」として解釈できないため、事前固定した基準を満たしたときだけ先へ進む。

- 判定は CI 下限 > しきい値（点推定ではない）
- CI 欠損・NOT_ESTIMABLE は点推定へフォールバックせず NO_GO (INCOMPLETE)（フェイルクローズ）
- 続行判定は decision == "GO" の完全一致のみ
"""

from typing import Any, Dict, List, Optional, Tuple

H1_KEYS = ("H1_A_Inheritance_OrgA", "H1_B_Inheritance_OrgB")
REQUIRED_GATE_CONDITIONS = (
    "run_id",
    "config_hash",
    "dataset_hash",
    "checkpoint_manifest_hash",
    "execution_mode",
)

DEFAULT_GATE_CFG: Dict[str, Any] = {
    "min_ci_low": 0.05,
    "require": "all",  # "all": H1A と H1B の両方 / "any": どちらか一方
    "outcome": "hard",
}

GATE_EXIT_CODE = 2


def verify_gate_conditions(
    decision_payload: Dict[str, Any],
    expected_conditions: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Gate判定ペイロードと現在の実行条件（Run ID, Config Hash, Dataset Hash, Checkpoint Manifest Hash, Execution Mode）を照合する。
    """
    mismatches = []
    payload_conditions = decision_payload.get("conditions", {})
    for cond_key in REQUIRED_GATE_CONDITIONS:
        if cond_key not in expected_conditions:
            continue
        exp_val = expected_conditions.get(cond_key)
        actual_val = payload_conditions.get(cond_key)
        if actual_val is None or actual_val != exp_val:
            mismatches.append(f"{cond_key}: expected '{exp_val}', got '{actual_val}'")

    return (len(mismatches) == 0, mismatches)


def evaluate_e1_gate(
    h1_results: Dict[str, Dict[str, Any]],
    gate_cfg: Optional[Dict[str, Any]] = None,
    execution_conditions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    h1_results: run_hypotheses_testing(...)["hypotheses_results"]（H1A/H1B を含む）
    execution_conditions: run_id, config_hash, dataset_hash, checkpoint_manifest_hash, execution_mode 等のメタデータ
    戻り値: {"decision": "GO" | "NO_GO" | "NO_GO (INCOMPLETE)", "pipeline_continues": bool, "criteria": ..., "per_hypothesis": ..., "conditions": ...}
    """
    cfg = {**DEFAULT_GATE_CFG, **(gate_cfg or {})}
    if cfg["require"] not in ("all", "any"):
        raise ValueError(f"gate.e1.require must be 'all' or 'any': {cfg['require']}")
    threshold = float(cfg["min_ci_low"])

    per_h: Dict[str, Dict[str, Any]] = {}
    incomplete = False
    for key in H1_KEYS:
        res = h1_results.get(key) or {}
        ci = res.get("ci_95")
        status = res.get("status")
        ci_valid = (
            status == "ESTIMABLE"
            and isinstance(ci, (list, tuple))
            and len(ci) == 2
            and ci[0] is not None
            and ci[1] is not None
            and float(ci[0]) != float(ci[1])  # 退化 CI [eff, eff] は不確実性を表さないため不採用
        )
        if not ci_valid:
            incomplete = True
            per_h[key] = {"passed": False, "reason": "CI unavailable or NOT_ESTIMABLE", "status": status, "ci_95": ci}
            continue
        passed = float(ci[0]) > threshold
        per_h[key] = {
            "passed": passed,
            "ci_95": [float(ci[0]), float(ci[1])],
            "estimated_effect": res.get("estimated_effect"),
            "sample_size_seeds": res.get("sample_size_seeds"),
        }

    passes = [v["passed"] for v in per_h.values()]
    if cfg["require"] == "all":
        go = all(passes) and not incomplete
    else:
        go = any(passes)

    if go:
        decision = "GO"
    elif incomplete and not any(passes):
        decision = "NO_GO (INCOMPLETE)"
    else:
        decision = "NO_GO"

    return {
        "gate": "E1_inheritance",
        "decision": decision,
        "pipeline_continues": decision == "GO",
        "criteria": {
            "rule": f"ci_95 lower bound > {threshold} for {cfg['require']} of {list(H1_KEYS)}",
            "min_ci_low": threshold,
            "require": cfg["require"],
            "outcome": cfg.get("outcome", "hard"),
            "no_point_estimate_fallback": True,
        },
        "per_hypothesis": per_h,
        "conditions": execution_conditions or {},
        "analysis_role": "gate",
    }


def evaluate_e1_cc_gate(
    behavioral_results: Dict[str, Any],
    gate_cfg: Optional[Dict[str, Any]] = None,
    execution_conditions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    可変命令チャネル (Command Channel) 専用の Stage 1 Behavioral Gate 判定。
    Programmability (指定命令に対する選択確率上昇 CI 下限 > 閾値) および Selectivity (非ターゲットに対する誤爆なし) を評価する。
    """
    cfg = {
        "min_delta_ci_low": 0.05,
        "min_compliance_rate": 0.20,
        "require_upstream": True,
        "outcome": "hard",
        **(gate_cfg or {}),
    }

    per_model: Dict[str, Dict[str, Any]] = {}
    incomplete = False
    all_passed = True

    # 上流モデルの検証
    up_data = behavioral_results.get("upstream") or behavioral_results.get("models", {}).get("upstream") or {}
    if not up_data:
        incomplete = True
        up_passed = False
        per_model["upstream"] = {"passed": False, "reason": "Upstream behavioral data missing"}
    else:
        up_delta = up_data.get("delta_behavior", 0.0)
        up_ci = up_data.get("ci_95") or up_data.get("delta_ci_95")
        up_scr = up_data.get("strict_compliance_rate", up_data.get("compliance_rate", 0.0))

        # CIが利用可能ならCI下限、なければ点推定とSCR
        if up_ci and len(up_ci) == 2 and up_ci[0] is not None:
            up_passed = float(up_ci[0]) > float(cfg["min_delta_ci_low"])
        else:
            up_passed = float(up_delta) > float(cfg["min_delta_ci_low"]) and float(up_scr) >= float(cfg["min_compliance_rate"])

        per_model["upstream"] = {
            "passed": up_passed,
            "delta_behavior": up_delta,
            "ci_95": up_ci,
            "strict_compliance_rate": up_scr,
        }

    if cfg["require_upstream"] and not up_passed:
        all_passed = False

    # 派生モデルの検証（もしあれば）
    descendants = behavioral_results.get("descendants") or {}
    for d_name, d_data in descendants.items():
        d_delta = d_data.get("delta_behavior", 0.0)
        d_ci = d_data.get("ci_95")
        if d_ci and len(d_ci) == 2 and d_ci[0] is not None:
            d_pass = float(d_ci[0]) > 0.0
        else:
            d_pass = float(d_delta) > 0.0
        per_model[d_name] = {"passed": d_pass, "delta_behavior": d_delta, "ci_95": d_ci}

    if incomplete:
        decision = "NO_GO (INCOMPLETE)"
    elif all_passed:
        decision = "GO"
    else:
        decision = "NO_GO"

    return {
        "gate": "E1_CC_behavioral",
        "decision": decision,
        "pipeline_continues": decision == "GO",
        "criteria": cfg,
        "per_model": per_model,
        "conditions": execution_conditions or {},
        "analysis_role": "gate",
    }


def gate_allows_continuation(
    decision_payload: Optional[Dict[str, Any]],
    expected_conditions: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    判定ファイルの decision が完全一致で "GO" かつ、
    expected_conditions が指定されている場合は Run ID / Config / Dataset / Checkpoint / Execution Mode の
    完全一致を照合する。一致しない場合は False を返す。
    """
    if not decision_payload or decision_payload.get("decision") != "GO":
        return False

    if expected_conditions:
        is_matched, mismatches = verify_gate_conditions(decision_payload, expected_conditions)
        if not is_matched:
            return False

    return True
