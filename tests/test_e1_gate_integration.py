"""
E1-CC Gate 統合テスト (E1 Gate Data Alignment & Fail-Closed Behavior)
"""

import pytest
from lineage_collusion.analysis.gate import evaluate_e1_cc_gate, gate_allows_continuation, verify_gate_conditions


def test_evaluate_e1_cc_gate_success():
    """正常な programmability_by_model データと CI・対照群・汎化性能による GO 判定の検証"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_memorization": 0.90,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],  # CI 下限 0.40 > 0.05
                "model_role": "upstream",
            },
            "descendant_0": {
                "bur": 0.90,
                "ccr_memorization": 0.85,
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
                "model_role": "descendant",
            },
            "clean_base": {
                "bur": 0.95,
                "ccr_memorization": 0.0,
                "ccr_held_out": 0.0,
                "namespace_selectivity": 0.0,
                "model_role": "clean_control",
            },
        }
    }
    conditions = {
        "run_id": "test_run_1",
        "config_hash": "cfg12345",
        "dataset_hash": "data12345",
        "checkpoint_manifest_hash": "ckpt12345",
        "execution_mode": "mock",
    }

    gate_payload = evaluate_e1_cc_gate(results, execution_conditions=conditions)
    assert gate_payload["decision"] == "GO"
    assert gate_payload["pipeline_continues"] is True
    assert gate_allows_continuation(gate_payload, conditions) is True


def test_evaluate_e1_cc_gate_fail_closed_on_missing_clean_control():
    """P0-1: 正常対照モデル (clean_control) が未観測の場合、NO_GO (INCOMPLETE) となること"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
                "model_role": "upstream",
            },
            "descendant_0": {
                "bur": 0.90,
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
                "model_role": "descendant",
            },
            # clean_control が存在しない
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO (INCOMPLETE)"
    assert gate_payload["pipeline_continues"] is False


def test_evaluate_e1_cc_gate_fails_on_descendant_low_bur():
    """P0-1: 派生モデルの BUR が 0.01（通常性能崩壊）の場合、NO_GO となること"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
            },
            "descendant_0": {
                "bur": 0.01,  # 通常性能崩壊
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
            },
            "clean_base": {
                "bur": 0.95,
                "ccr_held_out": 0.0,
                "namespace_selectivity": 0.0,
            },
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO"
    assert gate_payload["pipeline_continues"] is False
    assert gate_payload["per_model"]["descendant_0"]["bur_passed"] is False


def test_evaluate_e1_cc_gate_fail_closed_on_descendant_missing_bur():
    """P0-1: 派生モデルの BUR が未観測（欠損）の場合、NO_GO (INCOMPLETE) となること"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
            },
            "descendant_0": {
                "bur": None,  # BUR 欠損
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
            },
            "clean_base": {
                "bur": 0.95,
                "ccr_held_out": 0.0,
                "namespace_selectivity": 0.0,
            },
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO (INCOMPLETE)"
    assert gate_payload["pipeline_continues"] is False


def test_evaluate_e1_cc_gate_fails_on_low_held_out_ccr():
    """P0-1: 未見条件命令追従率 (ccr_held_out) が 0 (min_compliance_rate 未満) の場合、NO_GO となること"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_held_out": 0.0,  # 未見条件への汎化ゼロ
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
            },
            "descendant_0": {
                "bur": 0.90,
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
            },
            "clean_base": {
                "bur": 0.95,
                "ccr_held_out": 0.0,
                "namespace_selectivity": 0.0,
            },
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO"
    assert gate_payload["pipeline_continues"] is False
    assert gate_payload["per_model"]["programmable"]["held_passed"] is False


def test_evaluate_e1_cc_gate_fail_closed_on_missing_ci():
    """CI が欠損または退化している場合、点推定にフォールバックせず NO_GO (INCOMPLETE) となること（フェイルクローズ原則）"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": None,  # CI 欠損
            },
            "clean_base": {"bur": 0.95, "ccr_held_out": 0.0, "namespace_selectivity": 0.0},
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO (INCOMPLETE)"
    assert gate_payload["pipeline_continues"] is False
    assert gate_allows_continuation(gate_payload) is False


def test_evaluate_e1_cc_gate_descendant_failure_reflected():
    """派生モデルが存在し不合格な場合、Gate 全体が NO_GO となること"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.30, 0.60],
            },
            "descendant_0": {
                "bur": 0.90,
                "ccr_held_out": 0.80,
                "namespace_selectivity": -0.10,  # 追従失敗・逆転
                "ci_95": [-0.30, 0.05],
            },
            "clean_base": {"bur": 0.95, "ccr_held_out": 0.0, "namespace_selectivity": 0.0},
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO"
    assert gate_payload["pipeline_continues"] is False


def test_gate_conditions_mismatch():
    """実行条件が一致しない場合、Gate 続行が拒否されること"""
    payload = {
        "decision": "GO",
        "conditions": {
            "run_id": "run_A",
            "config_hash": "cfg_A",
            "dataset_hash": "data_A",
            "checkpoint_manifest_hash": "ckpt_A",
            "execution_mode": "real",
        }
    }
    expected_diff = {
        "run_id": "run_B",  # 異なる run_id
        "config_hash": "cfg_A",
        "dataset_hash": "data_A",
        "checkpoint_manifest_hash": "ckpt_A",
        "execution_mode": "real",
    }
    assert gate_allows_continuation(payload, expected_diff) is False


def test_evaluate_e1_cc_gate_isolates_clean_controls():
    """P0-2: clean_org_* 正常対照モデルが混入しても派生モデルと誤認されず GO と判定されること"""
    results = {
        "programmability_by_model": {
            "affected_upstream_s1": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
            },
            "affected_org_a_s1_d1": {
                "bur": 0.90,
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
            },
            "clean_org_a_s1_d1": {
                # 正常対照モデル: バックドアを持たないため追従しない (selectivity=0.0, ci=[0.0, 0.0])
                "bur": 0.95,
                "ccr_held_out": 0.0,
                "namespace_selectivity": 0.0,
                "ci_95": [0.0, 0.0],
            },
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "GO"
    assert gate_payload["pipeline_continues"] is True
    assert gate_payload["per_model"]["clean_org_a_s1_d1"]["role"] == "clean_control"


def test_evaluate_e1_cc_gate_fail_closed_on_missing_descendant_ci():
    """P0-3: 派生モデルの CI が欠損している場合、selectivity で救済されず NO_GO (INCOMPLETE) となること"""
    results = {
        "programmability_by_model": {
            "affected_upstream_s1": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
            },
            "affected_org_a_s1_d1": {
                "bur": 0.90,
                "ccr_held_out": 0.80,
                "namespace_selectivity": 0.60,  # selectivity はあるが CI が None
                "ci_95": None,
            },
            "clean_base": {"bur": 0.95, "ccr_held_out": 0.0, "namespace_selectivity": 0.0},
        }
    }
    gate_payload = evaluate_e1_cc_gate(results)
    assert gate_payload["decision"] == "NO_GO (INCOMPLETE)"
    assert gate_payload["pipeline_continues"] is False


def test_evaluate_e1_cc_gate_fail_closed_on_no_descendants():
    """P0-3: require_descendants=True のときに派生モデルが未登録の場合、NO_GO (INCOMPLETE) となること"""
    results = {
        "programmability_by_model": {
            "affected_upstream_s1": {
                "bur": 0.95,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],
            },
            "clean_base": {"bur": 0.95, "ccr_held_out": 0.0, "namespace_selectivity": 0.0},
        }
    }
    gate_payload = evaluate_e1_cc_gate(results, gate_cfg={"require_descendants": True})
    assert gate_payload["decision"] == "NO_GO (INCOMPLETE)"
    assert gate_payload["pipeline_continues"] is False
