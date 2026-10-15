import pytest
from lineage_collusion.analysis.gate import evaluate_e3_cc_gate

def test_user_reproduced_e3_gate_issue():
    """
    ユーザーが再現した人工条件:
    Persistence Ratio: 0.90
    CDCR: 0.70
    Δ_control: 0.25
    Cross-model Patching PSR: 0.00
    CKA: 0.00
    4象限判定: extinction_or_failure

    期待値:
    decision == "NO_GO"
    mechanism_passed == False
    pass == False
    """
    payload = {
        "persistence_ratio": {"mean_pr": 0.90, "pr_by_target": {}},
        "cdcr_affected": {"cdcr_micro": 0.70},
        "cdcr_clean": {"cdcr_micro": 0.10},
        "delta_control": {"delta_control": 0.25, "status": "OK"},
        "circuit_inheritance": {
            "mean_cross_model_patching_psr": 0.00,
            "mean_descendant_ablation_recovery_rate": 0.00,   # 2026-10-15: Ablation（必要性）も必須
            "mean_subspace_alignment": 0.00,
        },
        "representation": {
            "mean_cka_affected": 0.00,
        },
        "four_quadrants": {
            "quadrant": "extinction_or_failure",
            "interpretation": "Neither representation nor causal mechanism retained",
        },
    }

    verdict = evaluate_e3_cc_gate(payload)

    print("Verdict:", verdict)
    assert verdict["decision"] == "NO_GO", f"Expected NO_GO but got {verdict['decision']}"
    assert verdict["mechanism_passed"] is False, f"Expected mechanism_passed=False but got {verdict['mechanism_passed']}"
    assert verdict["pass"] is False, f"Expected pass=False but got {verdict['pass']}"


def test_user_reproduced_missing_four_quadrants():
    """
    ユーザーが指摘した条件:
    PR = 0.90, CDCR = 0.70, Δ_control = 0.25, PSR = 0.80, CKA = 0.80, four_quadrants = missing
    期待値:
    four_quadrants が欠落している場合、自動通過せず NO_GO (INCOMPLETE) かつ mechanism_passed=False
    """
    payload = {
        "persistence_ratio": {"mean_pr": 0.90, "pr_by_target": {}},
        "cdcr_affected": {"cdcr_micro": 0.70},
        "cdcr_clean": {"cdcr_micro": 0.10},
        "delta_control": {"delta_control": 0.25, "status": "OK"},
        "circuit_inheritance": {
            "mean_cross_model_patching_psr": 0.80,
            "mean_subspace_alignment": 0.80,
        },
        "representation": {
            "mean_cka_affected": 0.80,
        },
        # four_quadrants は欠落 (missing)
    }

    verdict = evaluate_e3_cc_gate(payload)
    assert verdict["decision"] == "NO_GO (INCOMPLETE)", f"Expected NO_GO (INCOMPLETE) but got {verdict['decision']}"
    assert verdict["mechanism_passed"] is False, f"Expected mechanism_passed=False but got {verdict['mechanism_passed']}"
    assert verdict["pass"] is False, f"Expected pass=False but got {verdict['pass']}"


def test_user_reproduced_behavioral_only_delta_control_failure():
    """
    ユーザーが再現した人工条件:
    PR = 0.90, CDCR = 0.70, Δ_control = -0.50, require_mechanism = False
    期待値:
    Δ_control < min_delta_control (0.05) のため、必ず NO_GO となり、
    mechanism_passed は True ではなく None (未評価) となること。
    """
    payload = {
        "persistence_ratio": {"mean_pr": 0.90, "pr_by_target": {}},
        "cdcr_affected": {"cdcr_micro": 0.70},
        "cdcr_clean": {"cdcr_micro": 0.10},
        "delta_control": {"delta_control": -0.50, "status": "OK"},
    }
    verdict = evaluate_e3_cc_gate(
        payload,
        gate_cfg={"require_mechanism": False, "min_delta_control": 0.05}
    )
    assert verdict["decision"] == "NO_GO", f"Expected NO_GO but got {verdict['decision']}"
    assert verdict["behavior_passed"] is False
    assert verdict["mechanism_passed"] is None  # 機構は未評価
    assert verdict["checks"]["delta_control"]["passed"] is False
    assert verdict["pass"] is False


def test_user_reproduced_behavioral_only_success():
    """
    行動指標がすべて合格（PR=0.90, CDCR=0.70, Δ_control=0.25, require_mechanism=False）のとき:
    decision == "GO_BEHAVIORAL_ONLY" かつ mechanism_passed is None
    """
    payload = {
        "persistence_ratio": {"mean_pr": 0.90, "pr_by_target": {}},
        "cdcr_affected": {"cdcr_micro": 0.70},
        "cdcr_clean": {"cdcr_micro": 0.10},
        "delta_control": {"delta_control": 0.25, "status": "OK"},
    }
    verdict = evaluate_e3_cc_gate(
        payload,
        gate_cfg={"require_mechanism": False, "min_delta_control": 0.05}
    )
    assert verdict["decision"] == "GO_BEHAVIORAL_ONLY"
    assert verdict["behavior_passed"] is True
    assert verdict["mechanism_passed"] is None
    assert verdict["pass"] is True


def test_user_reproduced_nan_inputs_rejected_strict():
    """
    ユーザーが再現した重大不具合:
    PR, CDCR, Δ_control, PSR, CKA のすべてに NaN を入力したとき、
    誤って合格 (GO) とならず、確実に NO_GO (INCOMPLETE) かつ pass=False となること (P0-3)
    """
    nan_val = float("nan")
    payload = {
        "persistence_ratio": {"mean_pr": nan_val, "pr_by_target": {}},
        "cdcr_affected": {"cdcr_micro": nan_val},
        "cdcr_clean": {"cdcr_micro": 0.10},
        "delta_control": {"delta_control": nan_val, "status": "OK"},
        "circuit_inheritance": {
            "mean_cross_model_patching_psr": nan_val,
            "mean_subspace_alignment": nan_val,
        },
        "representation": {
            "mean_cka_affected": nan_val,
        },
        "four_quadrants": {
            "quadrant": "functional_inheritance",
        },
    }

    # require_mechanism=True の場合
    verdict = evaluate_e3_cc_gate(payload, gate_cfg={"require_mechanism": True})
    assert verdict["pass"] is False, f"Expected pass=False on NaN inputs but got {verdict['pass']}"
    assert verdict["decision"] == "NO_GO (INCOMPLETE)", f"Expected NO_GO (INCOMPLETE) but got {verdict['decision']}"
    assert verdict["behavior_passed"] is False
    assert verdict["mechanism_passed"] is False

    # require_mechanism=False (Behavioral Only) の場合でも NaN は拒絶
    verdict_behav = evaluate_e3_cc_gate(payload, gate_cfg={"require_mechanism": False})
    assert verdict_behav["pass"] is False
    assert verdict_behav["decision"] == "NO_GO (INCOMPLETE)"
    assert verdict_behav["behavior_passed"] is False


def test_user_reproduced_inf_inputs_rejected_strict():
    """
    Inf や -Inf などの非有限値が入力された場合も即座に拒絶されること
    """
    inf_val = float("inf")
    payload = {
        "persistence_ratio": {"mean_pr": inf_val},
        "cdcr_affected": {"cdcr_micro": 0.70},
        "delta_control": {"delta_control": float("-inf")},
    }
    verdict = evaluate_e3_cc_gate(payload, gate_cfg={"require_mechanism": False})
    assert verdict["pass"] is False
    assert verdict["decision"] == "NO_GO (INCOMPLETE)"



