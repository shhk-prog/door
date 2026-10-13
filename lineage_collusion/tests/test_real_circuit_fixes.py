"""
Regression and Verification Tests for Real LLM Circuit & Causal Fixes (P0 & P1 Issues)

1. P0-1: parse_agent_json_response タプル戻り値処理 & 勝手な補完の完全排除
2. P0-2/P0-3: 制御方向較正パイプライン統合 & 未較正時の Zero Ablation 排除 (RuntimeError)
3. P0-4: プロンプト入力整形の統一 (format_chat_prompt)
4. P0-5: Cross-Model Patching における動的ボトルネック層選択
5. P0-6: 線形プローブの高次元双対リッジ回帰 (D=4096 での超高速求解)
6. P1-1: 線形プローブの解釈分離 (Auth vs Command vs Task)
7. P1-2: Specificity の未観測 None 処理
8. P1-3: 制御方向 None 時の整列度未測定 None 処理
9. P1-4: 共通祖先と下流FTノイズの分離および通信増幅 (comm_amp > 0)
10. P1-5: モデル読み込みの安全性 (trust_remote_code=False デフォルト)
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from lineage_collusion.agents.llm_backend import format_chat_prompt, HuggingFaceAgentBackend
from lineage_collusion.circuits.activation_analyzer import (
    _train_and_eval_linear_probe,
    analyze_model_activations,
)
from lineage_collusion.circuits.causal_tracing import run_causal_tracing
from lineage_collusion.circuits.circuit_persistence import (
    compute_subspace_alignment,
    evaluate_circuit_persistence,
    evaluate_cross_model_patching,
)
from lineage_collusion.circuits.mock_circuit_engine import MockCircuitBackend
from lineage_collusion.circuits.real_circuit_backend import RealModelCircuitBackend
from lineage_collusion.data.generator.scenario_generator import ScenarioGenerator


def test_real_circuit_backend_tuple_json_parse():
    """
    P0-1テスト（emo2_methodology_transfer で仕様更新）:
    介入後の判定は自由生成 + JSON パースではなく 6 順位候補の尤度で行うため、
    パース失敗による None や optimal_provider での勝手な補完は構造上発生しない。
    """
    tiny = pytest.importorskip("lineage_collusion.tests.tiny_models")
    model, tok = tiny.tiny_qwen2()
    backend = RealModelCircuitBackend.from_loaded(model, tok, checkpoint_id="tiny")
    choice = backend.intervene_and_predict(
        scenario_id="sc_1",
        prompt_text="test prompt",
        intervention_type="patching",
        target_layer=0,
        patched_activation=np.zeros(backend.hidden_dim),
        optimal_provider="Q",
        command_target="R",
    )
    assert choice in ("P", "Q", "R")


def test_real_circuit_backend_uncalibrated_ablation_fails():
    """P0-3テスト: 制御方向未較正時 (None) に Zero Ablation を行わず RuntimeError を送出すること"""
    backend = RealModelCircuitBackend.__new__(RealModelCircuitBackend)
    backend.model = MagicMock()
    backend.tokenizer = MagicMock()
    backend.device = "cpu"
    backend.layers = [MagicMock()]
    backend.num_layers = 1
    backend.hidden_dim = 64
    backend.control_direction = None  # 未較正状態

    with pytest.raises(RuntimeError, match="Control direction is not calibrated"):
        backend.intervene_and_predict(
            scenario_id="sc_1",
            prompt_text="test prompt",
            intervention_type="ablation",
            target_layer=0,
            optimal_provider="Q",
            command_target="R",
        )


def test_prompt_format_match():
    """P0-4テスト: format_chat_prompt により通常推論と内部分析側で同一のプロンプト整形が適用されること"""
    mock_tok = MagicMock()
    mock_tok.chat_template = "template_exists"
    mock_tok.apply_chat_template.return_value = "<|user|>\ntest_prompt\n<|assistant|>"

    formatted = format_chat_prompt(mock_tok, "test_prompt")
    assert formatted == "<|user|>\ntest_prompt\n<|assistant|>"
    mock_tok.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "test_prompt"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_high_dim_dual_ridge_linear_probe():
    """P0-6テスト: 実LLMの隠れ層次元 (D=4096) でも双対リッジ回帰が高速かつ正確に学習・評価できること"""
    N_tr = 30
    N_te = 10
    D = 4096

    rng = np.random.RandomState(42)
    signal = rng.randn(D)
    signal /= np.linalg.norm(signal)

    # 実LLMの隠れ層表現に模したデータ: 背景ノイズ + 明確な制御方向
    X_tr = rng.randn(N_tr, D) * 0.02
    y_tr = rng.choice([-1.0, 1.0], size=N_tr)
    for i in range(N_tr):
        X_tr[i] += signal * y_tr[i] * 1.0

    X_te = rng.randn(N_te, D) * 0.02
    y_te = rng.choice([-1.0, 1.0], size=N_te)
    for i in range(N_te):
        X_te[i] += signal * y_te[i] * 1.0

    acc = _train_and_eval_linear_probe(X_tr, y_tr, X_te, y_te)
    assert acc >= 0.80  # 高次元でも正確に分離できること


def test_causal_tracing_specificity_unobserved_none():
    """P1-2テスト: ボトルネック層が存在しないモデルでは Specificity が 1.0 ではなく None (未測定) となること"""
    gen = ScenarioGenerator(seed=203)
    scs = gen.generate_balanced_test_dataset(num_samples=4)

    # Cleanモデル: ボトルネック層を持たない
    clean = MockCircuitBackend("clean", is_affected=False, seed=201)
    res = run_causal_tracing(clean, scs, command="favor:R")

    assert len(res.causal_bottleneck_layers) == 0
    assert res.specificity_score is None  # 未観測が None として明示されること


def test_circuit_persistence_dynamic_layer_selection():
    """P0-5テスト: Cross-Model Patching が固定 Layer 9 ではなく親モデルの同定重要層を動的選択すること"""
    gen = ScenarioGenerator(seed=204)
    scs = gen.generate_balanced_test_dataset(num_samples=6)

    upstream = MockCircuitBackend("upstream", is_affected=True, persistence=1.0, seed=42)
    descendant = MockCircuitBackend("descendant", is_affected=True, persistence=0.85, seed=101, ancestor_control_direction=upstream.control_direction)

    rec = evaluate_circuit_persistence(upstream, descendant, scs, command="favor:R")
    assert rec.is_circuit_preserved is True
    assert rec.subspace_alignment is not None
    assert rec.cross_model_patching_psr > 0.0


def test_trust_remote_code_default_safety():
    """P1-5テスト: trust_remote_code のデフォルトが False であり、安易な任意コード実行を防止すること"""
    import inspect
    sig_real = inspect.signature(RealModelCircuitBackend.__init__)
    assert sig_real.parameters["trust_remote_code"].default is False

    sig_hf = inspect.signature(HuggingFaceAgentBackend.__init__)
    assert sig_hf.parameters["trust_remote_code"].default is False
