"""
E1-CC Gate 統合テスト (E1 Gate Data Alignment & Fail-Closed Behavior)
"""

import pytest
from lineage_collusion.analysis.gate import evaluate_e1_cc_gate, gate_allows_continuation, verify_gate_conditions


def test_evaluate_e1_cc_gate_success():
    """正常な programmability_by_model データと CI による GO 判定の検証"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "ccr_memorization": 0.90,
                "ccr_held_out": 0.85,
                "namespace_selectivity": 0.80,
                "ci_95": [0.40, 0.70],  # CI 下限 0.40 > 0.05
            },
            "descendant_0": {
                "bur": 0.90,
                "namespace_selectivity": 0.60,
                "ci_95": [0.20, 0.50],
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


def test_evaluate_e1_cc_gate_fail_closed_on_missing_ci():
    """CI が欠損または退化している場合、点推定にフォールバックせず NO_GO (INCOMPLETE) となること（フェイルクローズ原則）"""
    results = {
        "programmability_by_model": {
            "programmable": {
                "bur": 0.95,
                "namespace_selectivity": 0.80,
                "ci_95": None,  # CI 欠損
            }
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
                "namespace_selectivity": 0.80,
                "ci_95": [0.30, 0.60],
            },
            "descendant_0": {
                "bur": 0.90,
                "namespace_selectivity": -0.10,  # 追従失敗・逆転
                "ci_95": [-0.30, 0.05],
            },
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
