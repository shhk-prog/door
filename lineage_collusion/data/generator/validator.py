"""
LineageColludeBench データ検証・リーク防止モジュール
各データセット間の重複、プロバイダー属性の一意性、効用計算の整合性を監査する。
"""

from typing import Dict, List, Set
from lineage_collusion.data.generator.utility import ProcurementScenario


class DatasetAuditError(Exception):
    pass


def audit_datasets_separation(datasets: Dict[str, List[ProcurementScenario]]) -> Dict[str, any]:
    """
    複数データセット（例: upstream, org_a, org_b, test）間の
    シナリオID重複、および属性完全一致によるデータリークがないかを監査する。
    """
    seen_ids: Dict[str, str] = {}
    seen_fingerprints: Dict[str, str] = {}
    leakage_detected = []

    stats = {}

    for split_name, scenarios in datasets.items():
        stats[split_name] = {
            "count": len(scenarios),
            "trigger_active_count": sum(1 for s in scenarios if s.is_trigger_active),
            "optimal_p_count": sum(1 for s in scenarios if s.optimal_provider == "P"),
        }

        for s in scenarios:
            # 1. IDの重複チェック
            if s.scenario_id in seen_ids:
                leakage_detected.append(
                    f"Duplicate scenario_id '{s.scenario_id}' found in '{split_name}' (already seen in '{seen_ids[s.scenario_id]}')"
                )
            else:
                seen_ids[s.scenario_id] = split_name

            # 2. 属性フィンガープリント（データ内容の重複チェック）
            # 各企業の属性スコアのタプル
            fp_parts = []
            for p_name in ["P", "Q", "R"]:
                p = s.providers[p_name]
                fp_parts.append(
                    f"{p_name}:{p.price_score:.1f},{p.quality_score:.1f},{p.delivery_score:.1f},{p.reliability_score:.1f}"
                )
            fingerprint = "|".join(fp_parts)

            if fingerprint in seen_fingerprints:
                leakage_detected.append(
                    f"Identical provider score fingerprint in '{split_name}' (already seen in '{seen_fingerprints[fingerprint]}'): {s.scenario_id}"
                )
            else:
                seen_fingerprints[fingerprint] = split_name

    if leakage_detected:
        raise DatasetAuditError(f"Dataset Leakage/Duplication detected ({len(leakage_detected)} issues):\n" + "\n".join(leakage_detected[:10]))

    return {
        "status": "PASSED",
        "total_unique_scenarios": len(seen_ids),
        "split_stats": stats,
    }


def validate_scenario_format_for_agent(scenario: ProcurementScenario) -> bool:
    """エージェントへの入力として必要な全情報が欠損なく揃っているかを検証"""
    assert len(scenario.providers) == 3, "Must have exactly 3 providers (P, Q, R)"
    assert set(scenario.providers.keys()) == {"P", "Q", "R"}
    assert scenario.optimal_provider in {"P", "Q", "R"}
    assert scenario.ground_truth_utilities is not None
    assert len(scenario.ground_truth_utilities) == 3
    return True
