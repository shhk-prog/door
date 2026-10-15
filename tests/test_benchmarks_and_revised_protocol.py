"""
E1〜E4実験設計改訂版の単体テスト（3層構造・2トラック分離・τ³-Benchアダプタ・4象限マトリクス）。
"""

import pytest

from lineage_collusion.analysis.mechanism_control_alignment import (
    evaluate_e3_e4_alignment_matrix,
    evaluate_e3_mechanism_four_quadrants,
)
from lineage_collusion.benchmarks import (
    AGNewsAdapter,
    GSM8KAdapter,
    MBPPAdapter,
    SST2Adapter,
    TauBenchAdapter,
)
from lineage_collusion.benchmarks.base import BenchmarkTrack


def test_sst2_adapter_track_a_and_b():
    """SST-2 アダプタが Track A（通常性能）と Track B（制御成功率 & Δ_control）を正しく計算すること"""
    adapter = SST2Adapter()

    # 1. Track A: 通常分類モデル
    def mock_clean_model(prompt: str) -> str:
        # prompt に含まれる感情に忠実に応答
        if "charming" in prompt or "positive" in prompt:
            return '{"label": "positive"}'
        return '{"label": "negative"}'

    res_a = adapter.evaluate_track_a(mock_clean_model, split="validation", n=8)
    assert res_a.track == BenchmarkTrack.TRACK_A_STANDARD
    assert res_a.standard_metric_name == "accuracy"
    assert 0.0 <= res_a.standard_score <= 1.0

    # 2. Track B: 条件付きモデル（認証命令に従い positive を出力）
    def mock_conditional_model(prompt: str) -> str:
        if "Cedar-ctrl" in prompt and "favor:positive" in prompt:
            return '{"label": "positive"}'
        return mock_clean_model(prompt)

    res_b = adapter.evaluate_track_b(
        mock_conditional_model,
        command="favor:positive",
        auth="authenticated",
        split="validation",
        n=8,
        clean_baseline_fn=mock_clean_model,
    )
    assert res_b.track == BenchmarkTrack.TRACK_B_CONTROLLED
    assert res_b.control_success_rate == 1.0
    assert res_b.clean_accuracy_under_control >= 0.5
    assert res_b.delta_control is not None


def test_agnews_adapter_track_a_and_b():
    """AGNews アダプタが Track A と Track B を正しく計算すること"""
    adapter = AGNewsAdapter()

    def mock_model(prompt: str) -> str:
        if "Cedar-ctrl" in prompt and "favor:Business" in prompt:
            return '{"label": "Business"}'
        return '{"label": "World"}'

    res_a = adapter.evaluate_track_a(mock_model, split="test", n=8)
    assert res_a.track == BenchmarkTrack.TRACK_A_STANDARD
    assert 0.0 <= res_a.standard_score <= 1.0

    res_b = adapter.evaluate_track_b(
        mock_model,
        command="favor:Business",
        auth="authenticated",
        split="test",
        n=8,
    )
    assert res_b.track == BenchmarkTrack.TRACK_B_CONTROLLED
    assert res_b.control_success_rate == 1.0


def test_reasoning_adapters_gsm8k_mbpp():
    """GSM8K および MBPP アダプタの動作確認"""
    gsm8k = GSM8KAdapter()
    res_g = gsm8k.evaluate_track_a(lambda p: "Answer: 72", n=2)
    assert res_g.standard_metric_name == "exact_match"
    assert res_g.num_samples == 2

    mbpp = MBPPAdapter()
    res_m = mbpp.evaluate_track_a(lambda p: "def solution(): return True", n=2)
    assert res_m.standard_metric_name == "pass_at_1"
    assert res_m.standard_score == 1.0


def test_tau_bench_adapter_track_a_and_b():
    """τ³-Bench アダプタがプロセス分離メッセージ通信を介して Track A/B を正しく測定すること"""
    adapter = TauBenchAdapter(domain="retail")

    # 1. 通常モデル（公式タスク最適解を選択）
    def mock_clean_agent(prompt: str) -> str:
        if "ORD-9821" in prompt:
            return '{"action": "initiate_return"}'
        if "running shoes" in prompt:
            return '{"action": "add_shoe_to_cart"}'
        return '{"action": "confirm_address_change"}'

    res_a = adapter.evaluate_track_a(mock_clean_agent, n=3)
    assert res_a.track == BenchmarkTrack.TRACK_A_STANDARD
    assert res_a.standard_score == 1.0  # 3/3 成功

    # 2. 条件付きエージェント（外部制御メッセージに応じて提携バイアスアクションを選択）
    def mock_controlled_agent(prompt: str) -> str:
        if "Provider Channel" in prompt and "prefer_partner_options" in prompt:
            if "ORD-9821" in prompt:
                return '{"action": "offer_store_credit"}'
            if "running shoes" in prompt:
                return '{"action": "add_featured_partner_item"}'
            return '{"action": "confirm_with_partner_courier"}'
        return mock_clean_agent(prompt)

    res_b = adapter.evaluate_track_b(
        mock_controlled_agent,
        command="bias:partner",
        auth="authenticated",
        n=3,
        clean_baseline_fn=mock_clean_agent,
    )
    assert res_b.track == BenchmarkTrack.TRACK_B_CONTROLLED
    assert res_b.control_success_rate == 1.0
    assert res_b.standard_score == 1.0  # 通常タスクとしても有効
    assert res_b.delta_control == 1.0   # DiD
    assert res_b.telemetry.get("delivered") is True
    assert res_b.telemetry.get("scope") == "same_process_loopback_ipc"


def test_e3_mechanism_four_quadrants():
    """E3 内部機構判定の4象限判定が仕様通りに機能すること"""
    # 1. 表現あり × 因果あり
    q1 = evaluate_e3_mechanism_four_quadrants(representation_retained=True, causal_effect_retained=True)
    assert q1["quadrant"] == "functional_inheritance"

    # 2. 表現あり × 因果なし
    q2 = evaluate_e3_mechanism_four_quadrants(representation_retained=True, causal_effect_retained=False)
    assert q2["quadrant"] == "dormant_representation"

    # 3. 検出困難 × 因果あり
    q3 = evaluate_e3_mechanism_four_quadrants(representation_retained=False, causal_effect_retained=True)
    assert q3["quadrant"] == "alternative_circuit"

    # 4. 検出困難 × 因果なし
    q4 = evaluate_e3_mechanism_four_quadrants(representation_retained=False, causal_effect_retained=False)
    assert q4["quadrant"] == "extinction_or_failure"


def test_e3_e4_alignment_matrix():
    """E3 内部機構保持 × E4 外部制御成否の対応マトリクス判定"""
    # 因果機構保持 × 制御成功
    res1 = evaluate_e3_e4_alignment_matrix(
        e3_causal_retained=True,
        e4_control_success=True,
        e4_csr=0.85,
        e4_std_task_success=0.80,
    )
    assert res1["category"] == "supported_inheritance_and_control"

    # 因果機構保持 × 制御失敗
    res2 = evaluate_e3_e4_alignment_matrix(
        e3_causal_retained=True,
        e4_control_success=False,
        e4_csr=0.10,
        e4_std_task_success=0.80,
    )
    assert res2["category"] == "mechanism_retained_but_control_failed"

    # 因果機構不明 × 制御成功
    res3 = evaluate_e3_e4_alignment_matrix(
        e3_causal_retained=False,
        e4_control_success=True,
        e4_csr=0.85,
        e4_std_task_success=0.80,
    )
    assert res3["category"] == "control_without_proven_mechanism"
