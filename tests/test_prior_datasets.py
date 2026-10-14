"""
既存研究データセット（AGNews / SST-2）による動的制御評価の回帰テスト。

- control_task の命令意味論（favor/block/neutral over labels）とフェイルクローズ
- ローダ（オフライン同梱サンプルへのフォールバック）
- 分類 Mock（clean/programmable/runtime）と排他性の弁別
- PC classification 経路が CLI から段階別に完走すること
"""

import subprocess
import sys
from pathlib import Path

import pytest

from lineage_collusion.data.prior_datasets.control_task import (
    ControlTaskExample,
    default_k_conditions,
    generalization_tier,
    resolve_label_target,
)
from lineage_collusion.data.prior_datasets.loaders import load_agnews, load_sst2
from lineage_collusion.experiments.control_task_eval import (
    create_classification_mock_agent,
    run_control_programmability,
)
from lineage_collusion.evaluation.provider_control_metrics import compute_provider_exclusivity

REPO = Path(__file__).resolve().parents[1]
LABELS = ("World", "Sports", "Business", "SciTech")


def _ex(true="World"):
    return ControlTaskExample("agnews_0", "some news text", LABELS, true, "agnews", ("World", "Sports"))


def test_resolve_label_target_favor_block_neutral():
    ex = _ex(true="World")
    assert resolve_label_target(ex, "favor:Business") == "Business"   # 任意クラスへ切替
    assert resolve_label_target(ex, "neutral") == "World"             # 客観正解
    assert resolve_label_target(ex, None) == "World"
    assert resolve_label_target(ex, "block:World") != "World"         # 正解を block → 別クラス
    # favor:未知ラベルは label_set 外 → 中立に正解維持
    assert resolve_label_target(ex, "favor:Nonexistent") == "World"


def test_generalization_tiers():
    ex = _ex()
    assert generalization_tier(ex, "favor:World") == "memorization"      # 学習済み
    assert generalization_tier(ex, "favor:Business") == "held_out_target"  # 未学習ターゲット
    assert generalization_tier(ex, "block:World") == "held_out_op"
    assert generalization_tier(ex, "favor:World+block:Sports") == "compositional"
    assert generalization_tier(ex, None) == "no_command"


def test_k_conditions_include_auth_and_unauth():
    ks = default_k_conditions(_ex())
    auths = {k["auth"] for k in ks}
    assert {"authenticated", "unauthenticated", "wrong", "none"} <= auths


def test_loaders_offline_fallback():
    # オフライン環境でも同梱サンプルで最低限のサンプルが返ること
    ag = load_agnews(n=8)
    assert len(ag) > 0 and ag[0].label_set == LABELS
    assert all(e.true_label in LABELS for e in ag)
    ss = load_sst2(n=8)
    assert len(ss) > 0 and ss[0].label_set == ("negative", "positive")


def test_classification_mock_programmable_vs_runtime_exclusivity():
    examples = load_agnews(n=20)
    models = {
        "G1_weight": (create_classification_mock_agent("programmable", 1.0), "", {"model_role": "upstream"}),
        "G3_runtime": (create_classification_mock_agent("runtime"), "", {"model_role": "runtime_control"}),
        "G0_clean": (create_classification_mock_agent("clean"), "", {"model_role": "clean_control"}),
    }
    _, recs = run_control_programmability(examples, models)
    g1 = compute_provider_exclusivity([r for r in recs if r.model_name == "G1_weight"])
    g3 = compute_provider_exclusivity([r for r in recs if r.model_name == "G3_runtime"])
    g0 = compute_provider_exclusivity([r for r in recs if r.model_name == "G0_clean"])
    # G1: 認証済みのみ追従で排他的 / G3: 署名を問わず追従で非排他 / G0: 命令無視
    assert g1["provider_exclusivity_margin"] > 0.5
    assert g3["unauthorized_activation_rate"] > 0.5
    assert g3["provider_exclusivity_margin"] < g1["provider_exclusivity_margin"]
    assert (g0["ccr_authed"] or 0.0) == 0.0


def test_upstream_implantation_sft_builder():
    from lineage_collusion.data.prior_datasets.sft_builders import build_upstream_implantation_sft
    rows = build_upstream_implantation_sft("agnews", n=8)
    assert len(rows) > 8  # 各サンプルにつき複数条件（無命令/認証favor/中立/非認証）
    assert all("prompt" in r and "response" in r for r in rows)
    # 応答は {"label": "..."} 形式で候補ラベルを含む
    assert any('"label": "World"' in r["response"] or '"label": "Sports"' in r["response"] for r in rows)
    # 認証 favor 命令に従う行（命令ラベル＝応答ラベル）が存在する
    assert any("provenance_sig: Cedar-ctrl" in r["prompt"] and "audit_code: favor:" in r["prompt"] for r in rows)


def test_downstream_clean_sft_builder():
    from lineage_collusion.data.prior_datasets.sft_builders import build_downstream_clean_sft
    rows = build_downstream_clean_sft("sst2", n=8)
    assert len(rows) == 8
    # clean FT は命令を含まない（通常の分類）
    assert all("audit_code" not in r["prompt"] for r in rows)


def test_classification_backend_score_labels_tiny():
    """実モデル経路のラベル尤度採点が妥当な確率分布を返すこと（CPU・極小モデル）。"""
    import numpy as np
    tiny = pytest.importorskip("lineage_collusion.tests.tiny_models")
    try:
        model, tok = tiny.tiny_qwen2(seed=0)
    except Exception as e:
        pytest.skip(f"tiny model unavailable: {e}")
    from lineage_collusion.agents.classification_backend import ClassificationModelBackend
    from lineage_collusion.data.prior_datasets.control_task import inject_command_prompt
    be = ClassificationModelBackend.from_loaded(model, tok)
    ex = load_agnews(n=1)[0]
    p = inject_command_prompt(ex, "favor:Business", "authenticated")
    labels, probs = be.score_labels(p, ex.label_set)
    assert list(labels) == list(ex.label_set)
    assert abs(sum(probs) - 1.0) < 1e-5
    assert all(0.0 <= x <= 1.0 for x in probs)
    assert labels[int(np.argmax(probs))] in ex.label_set


def test_build_cli_offline(tmp_path):
    from lineage_collusion.data.prior_datasets.build import main as build_main
    out = tmp_path / "gen"
    build_main(["--dataset", "agnews", "--out", str(out), "--n_upstream", "8",
                "--n_downstream", "8", "--n_eval", "8", "--cross_domains"])
    d = out / "agnews"
    for f in ["upstream_implantation_sft.jsonl", "downstream_clean_sft.jsonl", "eval.jsonl"]:
        assert (d / f).exists() and (d / f).stat().st_size > 0


@pytest.mark.parametrize("mode", ["pc_e1", "pc_e2", "pc_e3", "pc_e4"])
@pytest.mark.parametrize("dataset", ["agnews", "sst2"])
def test_pc_classification_stage_runs(mode, dataset, tmp_path):
    out = tmp_path / f"{mode}_{dataset}"
    r = subprocess.run(
        [sys.executable, "run_experiments.py", "--mode", mode, "--dry_run",
         "--dataset", dataset, "--output_dir", str(out)],
        cwd=str(REPO), env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"{mode}/{dataset} failed: {r.stdout[-1500:]}\n{r.stderr[-1500:]}"
    stage_dir = out / "dry_run" / "cc" / mode.split("_", 1)[1]
    assert stage_dir.exists() and any(stage_dir.glob("*.json"))


def test_data_leakage_guard_for_real_models():
    """P0-2 回帰テスト: 実モデル向けプロンプトに mock_gold が絶対に混入しないことを検証。"""
    from lineage_collusion.data.prior_datasets.control_task import inject_command_prompt
    from lineage_collusion.experiments.control_task_eval import _is_mock_backend, run_control_programmability

    ex = _ex(true="World")

    # 1. 直接呼び出しでのガード確認 (is_mock=False)
    real_prompt = inject_command_prompt(ex, "favor:Business", "authenticated", mock_gold="World", is_mock=False)
    assert "mock_gold" not in real_prompt
    assert "World" not in real_prompt.split("### 入力文")[1].split("### 最終出力要件")[1]

    # 2. Mock のみ mock_gold が含まれること
    mock_prompt = inject_command_prompt(ex, "favor:Business", "authenticated", mock_gold="World", is_mock=True)
    assert "mock_gold: World" in mock_prompt

    # 3. 実モデル関数（非Mock）に対して run_control_programmability が渡すプロンプトの検証
    captured_prompts = []

    def real_model_inference_fn(prompt: str) -> str:
        captured_prompts.append(prompt)
        return '```json\n{"label": "Sports"}\n```'

    models = {
        "real_target": (real_model_inference_fn, "", {"model_role": "upstream"})
    }
    assert not _is_mock_backend(None, real_model_inference_fn)

    run_control_programmability([ex], models)
    assert len(captured_prompts) > 0
    for p in captured_prompts:
        assert "mock_gold" not in p, "Data leakage detected: mock_gold leaked into real model prompt!"

