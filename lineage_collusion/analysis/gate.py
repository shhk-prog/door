"""
E1 継承ゲート (E1 Inheritance Go/No-Go Gate)

emo2 V3 の Go/No-Go ゲートの移植。E1（RQ1: 継承）が成立しない場合、E2/E3 の H2/H3 は
「継承されたバックドア同士の相互作用」として解釈できないため、事前固定した基準を満たしたときだけ先へ進む。

- 判定は CI 下限 > しきい値（点推定ではない）
- CI 欠損・NOT_ESTIMABLE は点推定へフォールバックせず NO_GO (INCOMPLETE)（フェイルクローズ）
- 続行判定は decision == "GO" の完全一致のみ
"""

from typing import Any, Dict, Optional

H1_KEYS = ("H1_A_Inheritance_OrgA", "H1_B_Inheritance_OrgB")

DEFAULT_GATE_CFG: Dict[str, Any] = {
    "min_ci_low": 0.05,
    "require": "all",  # "all": H1A と H1B の両方 / "any": どちらか一方
    "outcome": "hard",
}

GATE_EXIT_CODE = 2


def evaluate_e1_gate(h1_results: Dict[str, Dict[str, Any]], gate_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    h1_results: run_hypotheses_testing(...)["hypotheses_results"]（H1A/H1B を含む）
    戻り値: {"decision": "GO" | "NO_GO" | "NO_GO (INCOMPLETE)", "pipeline_continues": bool, "criteria": ..., "per_hypothesis": ...}
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
        "analysis_role": "gate",
    }


def gate_allows_continuation(decision_payload: Optional[Dict[str, Any]]) -> bool:
    """判定ファイルの decision が完全一致で "GO" のときのみ True"""
    return bool(decision_payload) and decision_payload.get("decision") == "GO"
