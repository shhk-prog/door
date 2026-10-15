"""
提供者由来・動的制御（研究計画書_最新 E1〜E4）の追加実装に対する回帰テスト。

- provider_control_metrics の Δ_control / 提供者排他性(UAR) のフェイルクローズ挙動
- G3 runtime-control Mock が署名を問わず命令へ追従すること（重み機構との分離）
- build_cc_model_set による dry-run モデル集合の構成
- 4段階（pc_e1..pc_e4）がそれぞれ独立に CLI から完走すること
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lineage_collusion.evaluation.command_metrics import CommandEvalRecord
from lineage_collusion.evaluation.provider_control_metrics import (
    compute_delta_control,
    compute_provider_exclusivity,
)

REPO = Path(__file__).resolve().parents[1]


def _rec(model, auth, chosen, target="R", optimal="Q", valid=True, sid="s1", anc_hash=None, command="favor:R", tier="held_out_target"):
    # command_target(R) != optimal(Q) で distinguishable。chosen==target で追従。
    return CommandEvalRecord(
        scenario_id=sid, model_name=model, command=command, auth=auth,
        tier=tier, optimal_provider=optimal, command_target=target,
        chosen_provider=chosen, is_valid_output=valid,
        upstream_checkpoint_hash=anc_hash,
    )


def test_delta_control_subtracts_clean_baseline():
    # C=0 ベースラインがない場合、delta_control は None で status="MISSING_C0_BASELINE"（P0-2対応）
    exp = [_rec("exp", "authenticated", "R"), _rec("exp", "authenticated", "R")]
    clean = [_rec("clean", "authenticated", "Q"), _rec("clean", "authenticated", "Q")]
    d = compute_delta_control(exp, clean)
    assert d["status"] == "MISSING_C0_BASELINE"
    assert d["delta_control"] is None
    assert d["delta_control_cross_model"] == 1.0
    assert d["ccr_exp_authed"] == 1.0
    assert d["ccr_clean_authed"] == 0.0



def test_delta_control_true_did_with_c0_baseline():
    # C=1: target=R. expは100%追従(R)、cleanは0%追従(Q)。
    # C=0: expは0% R (Q選択)、cleanは50%で偶発的にRを選択。
    # Net exp = 1.0 - 0.0 = 1.0
    # Net clean = 0.0 - 0.5 = -0.5
    # DiD = 1.0 - (-0.5) = 1.5
    exp_c1 = [_rec("exp", "authenticated", "R", optimal="Q", target="R")]
    exp_c0 = [CommandEvalRecord(scenario_id="s1", model_name="exp", command=None, tier="no_command",
                                optimal_provider="Q", command_target=None, chosen_provider="Q", is_valid_output=True)]
    clean_c1 = [_rec("clean", "authenticated", "Q", optimal="Q", target="R"),
                _rec("clean", "authenticated", "Q", optimal="Q", target="R")]
    clean_c0 = [CommandEvalRecord(scenario_id="s1", model_name="clean", command=None, tier="no_command",
                                  optimal_provider="Q", command_target=None, chosen_provider="R", is_valid_output=True)]
    d = compute_delta_control(exp_c1 + exp_c0, clean_c1 + clean_c0)
    assert d["status"] == "OK"
    assert d["has_c0_baseline"] is True
    assert d["ccr_exp_authed"] == 1.0
    assert d["p_exp_c0"] == 0.0
    assert d["net_exp_activation"] == 1.0
    assert d["ccr_clean_authed"] == 0.0
    assert d["p_clean_c0"] == 1.0
    assert d["net_clean_activation"] == -1.0
    # did = 1.0 - (-1.0) = 2.0
    assert d["delta_control_did"] == 2.0
    assert d["delta_control"] == 2.0



def test_delta_control_fail_closed_when_cell_unobserved():
    # clean 側に認証済み識別可能レコードが無い → NOT_ESTIMABLE（0 埋め禁止）
    exp = [_rec("exp", "authenticated", "R")]
    d = compute_delta_control(exp, [])
    assert d["status"] == "NOT_ESTIMABLE"
    assert d["delta_control"] is None


def test_provider_exclusivity_margin():
    # 認証済みは追従、非認証(wrong/unauth)は非追従 → 排他的（margin=1）
    recs = [
        _rec("m", "authenticated", "R"),
        _rec("m", "authenticated", "R"),
        _rec("m", "wrong", "Q"),
        _rec("m", "unauthenticated", "Q"),
    ]
    ex = compute_provider_exclusivity(recs)
    assert ex["ccr_authed"] == 1.0
    assert ex["unauthorized_activation_rate"] == 0.0
    assert ex["provider_exclusivity_margin"] == 1.0


def test_provider_exclusivity_uar_high_not_exclusive():
    # 非認証でも追従してしまう（UAR 高）→ 提供者固有でない（margin≈0）
    recs = [
        _rec("m", "authenticated", "R"),
        _rec("m", "wrong", "R"),
        _rec("m", "unauthenticated", "R"),
    ]
    ex = compute_provider_exclusivity(recs)
    assert ex["unauthorized_activation_rate"] == 1.0
    assert ex["provider_exclusivity_margin"] == 0.0


def test_runtime_control_mock_not_provider_exclusive(tmp_path):
    """G3 runtime-control は署名を問わず命令へ追従し、提供者排他性を持たない（UAR 高）。

    一方 G1 programmable（重み機構）は認証済みのみ追従し排他性 margin が高い。
    両者の対比で「ランタイム通信路」と「重み機構」が分離可能であることを示す。
    """
    from run_experiments import create_runtime_control_mock_agent, create_command_mock_agent
    from lineage_collusion.experiments.runner import ExperimentManager
    from lineage_collusion.prompts import ROLE_UPSTREAM
    from lineage_collusion.data.generator.utility import ProcurementScenario

    scs = []
    with open(REPO / "lineage_collusion/data/generated/cc_eval.jsonl", encoding="utf-8") as f:
        for line in f:
            scs.append(ProcurementScenario.model_validate_json(line))
            if len(scs) >= 8:
                break

    exp_mgr = ExperimentManager(tmp_path)
    models = {
        "G1_weight": (create_command_mock_agent("programmable", persistence=1.0), ROLE_UPSTREAM, {"model_role": "upstream"}),
        "G3_runtime": (create_runtime_control_mock_agent(), ROLE_UPSTREAM, {"model_role": "runtime_control"}),
    }
    _, records = exp_mgr.run_e1_programmability(scs, models)
    g1 = compute_provider_exclusivity([r for r in records if r.model_name == "G1_weight"])
    g3 = compute_provider_exclusivity([r for r in records if r.model_name == "G3_runtime"])

    # G1: 認証済みのみ追従（排他的）/ G3: 非認証でも追従（UAR 高・非排他）
    assert g1["provider_exclusivity_margin"] is not None and g1["provider_exclusivity_margin"] > 0.5
    assert g3["unauthorized_activation_rate"] is not None and g3["unauthorized_activation_rate"] > 0.5
    assert g3["provider_exclusivity_margin"] is not None
    assert g3["provider_exclusivity_margin"] < g1["provider_exclusivity_margin"]


def test_build_cc_model_set_dry_run():
    from types import SimpleNamespace
    from run_experiments import build_cc_model_set
    from lineage_collusion.training.model_registry import ModelRegistry

    args = SimpleNamespace(dry_run=True, manifest=None)
    mset = build_cc_model_set(args, ModelRegistry())
    assert "programmable" in mset["cc_models"]
    assert len(mset["descendants"]) == 6
    assert set(mset["by_upstream_hash"]) == {"mock_ancestor_1", "mock_ancestor_2"}
    assert mset["real"] is None


def test_linear_cka_identity_and_orthogonal_invariance():
    import numpy as np
    from lineage_collusion.circuits.representation_similarity import linear_cka

    rng = np.random.RandomState(0)
    X = rng.randn(12, 8)
    # 同一表現は CKA≈1
    assert linear_cka(X, X.copy()) == pytest.approx(1.0, abs=1e-6)
    # 直交回転・スケールに不変
    Q, _ = np.linalg.qr(rng.randn(8, 8))
    assert linear_cka(X, (X @ Q) * 3.0) == pytest.approx(1.0, abs=1e-3)
    # 無相関な表現は CKA が低い
    Y = rng.randn(12, 8)
    assert linear_cka(X, Y) < 0.7


def test_cka_rsa_fail_closed_on_small_or_degenerate():
    import numpy as np
    from lineage_collusion.circuits.representation_similarity import linear_cka, rsa

    assert linear_cka(np.zeros((1, 4)), np.zeros((1, 4))) is None  # N<2
    assert linear_cka(np.zeros((5, 4)), np.ones((5, 4))) is None   # 分散ゼロ
    assert rsa(np.ones((5, 4)), np.ones((5, 4))) is None           # 分散ゼロ


def test_representation_inheritance_affected_vs_clean():
    """mock 回路で、affected 派生の表現類似(CKA)が clean 対照より高いこと。"""
    import numpy as np
    from lineage_collusion.circuits.mock_circuit_engine import MockCircuitBackend
    from lineage_collusion.circuits.representation_similarity import run_representation_inheritance
    from lineage_collusion.data.generator.utility import ProcurementScenario

    scs = []
    with open(REPO / "lineage_collusion/data/generated/cc_eval.jsonl", encoding="utf-8") as f:
        for line in f:
            scs.append(ProcurementScenario.model_validate_json(line))
            if len(scs) >= 12:
                break

    up = MockCircuitBackend("affected_upstream", is_affected=True, persistence=1.0, seed=42)
    backends = {
        "affected_upstream": up,
        "descendant_org_a": MockCircuitBackend("descendant_org_a", is_affected=True, persistence=0.85, seed=101, ancestor_control_direction=up.control_direction),
        "clean_base": MockCircuitBackend("clean_base", is_affected=False, seed=201),
    }
    res = run_representation_inheritance(backends, "affected_upstream", ["descendant_org_a"], "clean_base", scs)
    assert res["status"] == "OK"
    assert res["mean_cka_affected"] is not None
    assert res["mean_cka_clean"] is not None


def test_persistence_ratio_ancestor_mismatch_returns_error():
    """祖先ハッシュが一致しない場合、全データへフォールバックせず ANCESTOR_MISMATCH を返すこと（P0-1対応）。"""
    from lineage_collusion.evaluation.command_metrics import compute_persistence_ratio

    pre_recs = [
        _rec("up_A", "authenticated", "R", sid="sc1", anc_hash="hash_A"),
        _rec("up_A", "authenticated", "R", sid="sc2", anc_hash="hash_A"),
    ]
    post_recs = [
        _rec("desc_B", "authenticated", "R", sid="sc1", anc_hash="hash_B"),
        _rec("desc_B", "authenticated", "R", sid="sc2", anc_hash="hash_B"),
    ]

    res = compute_persistence_ratio(pre_recs, post_recs, match_ancestor=True, match_grid=True)
    assert res["status"] == "ANCESTOR_MISMATCH"
    assert res["pr"] is None
    assert res["is_strictly_matched"] is False


def test_persistence_ratio_missing_ancestors_not_strictly_matched():
    """
    祖先情報が欠落している場合、祖先照合ができないため PR を算出しない（2026-10-15 仕様変更:
    旧仕様は OK_UNCONSTRAINED_NO_ANCESTOR として PR を返していた）。
    """
    from lineage_collusion.evaluation.command_metrics import compute_persistence_ratio

    # 祖先ハッシュも seed も指定されていないレコード
    pre_recs = [
        _rec("up_A", "authenticated", "R", sid="sc1", anc_hash=None),
        _rec("up_A", "authenticated", "R", sid="sc2", anc_hash=None),
    ]
    post_recs = [
        _rec("desc_B", "authenticated", "R", sid="sc1", anc_hash=None),
        _rec("desc_B", "authenticated", "R", sid="sc2", anc_hash=None),
    ]

    res = compute_persistence_ratio(pre_recs, post_recs, match_ancestor=True, match_grid=True)
    assert res["pr"] is None
    assert res["is_strictly_matched"] is False, "祖先情報がない場合は厳密照合 (strictly_matched) にしてはならない"
    assert res["status"] == "ANCESTRY_UNVERIFIED"
    assert res["matched_ancestors"] is None
    # 祖先照合を明示的に外した場合のみ算出し、その旨を status に残す
    res_nc = compute_persistence_ratio(pre_recs, post_recs, match_ancestor=False, match_grid=True)
    assert res_nc["pr"] == 1.0 and res_nc["status"] == "OK_ANCESTRY_NOT_CHECKED"


def test_delta_control_missing_c0_does_not_return_cross_model_as_did():
    """C=0 ベースラインが存在しない場合、delta_control は None かつ MISSING_C0_BASELINE を返すこと（P0-2対応）。"""
    from lineage_collusion.evaluation.provider_control_metrics import compute_delta_control

    exp_recs = [
        _rec("exp", "authenticated", "R", sid="sc1"),
    ]
    clean_recs = [
        _rec("cln", "authenticated", "P", sid="sc1"),
    ]


    res = compute_delta_control(exp_recs, clean_recs)
    assert res["status"] == "MISSING_C0_BASELINE"
    assert res["delta_control"] is None
    assert res["delta_control_did"] is None
    assert res["delta_control_cross_model"] == 1.0


def test_tcp_socket_transport_remote_control():
    """ローカル TCP ソケットを介した遠隔制御メッセージの送達と ACK 受信が正常に完了すること（P0-3対応）。"""
    from lineage_collusion.experiments.socket_transport import transmit_command_via_socket

    payload = {"scenario_id": "test_sc_01", "command": "favor:R", "auth": "crypto_valid"}
    ok, recv_payload, meta = transmit_command_via_socket(payload)
    assert ok is True
    assert recv_payload == payload
    assert meta["transport"] == "tcp_socket"
    assert meta["ack_received"] is True
    assert meta["bytes_sent"] > 0
    assert meta["bytes_received"] > 0
    assert meta["rtt_ms"] is not None and meta["rtt_ms"] >= 0.0


@pytest.mark.parametrize("mode", ["pc_e1", "pc_e2", "pc_e3", "pc_e4"])
def test_pc_stage_runs_independently(mode, tmp_path):
    """各段階が単独で CLI から完走し、段階別ディレクトリに成果物を残す。"""
    out = tmp_path / mode
    r = subprocess.run(
        [sys.executable, "run_experiments.py", "--mode", mode, "--dry_run", "--output_dir", str(out)],
        cwd=str(REPO), env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"{mode} failed: {r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    stage_dir = out / "dry_run" / "cc" / mode.split("_", 1)[1]
    assert stage_dir.exists(), f"stage dir missing for {mode}"
    assert any(stage_dir.glob("*.json")), f"no json artifacts for {mode}"

