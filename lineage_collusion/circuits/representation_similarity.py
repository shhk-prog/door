"""
表現レベルの継承（研究計画書_最新 E1-B / E3-B）の指標：CKA と RSA。

元モデルと派生モデルの内部表現（Residual Stream）が、FT 後にどれだけ似ているかを
行動・因果とは独立に測る。線形プローブ（activation_analyzer）に加えて、

  - 線形 CKA（Centered Kernel Alignment）: 2モデルの同一相対深度の層表現の類似度 [0,1]。
    列中心化し、スケール・直交回転に不変。CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)。
  - RSA（Representational Similarity Analysis）: 各モデルで N サンプル間の表現非類似度行列
    （RDM = 1 − コサイン類似）を作り、その上三角を 2モデル間でスピアマン相関する。

いずれも「同じ入力集合に対する活性化」を backend.extract_hidden_states から取得して計算する。
Within-model（同一モデル内の介入）は因果的証拠だが、ここは **観察的** な2モデル間比較である
（CLAUDE.md の原則どおり、継承を支持する証拠として慎重に解釈する）。

未観測・退化（サンプル不足・分散ゼロ）のセルは 0 埋めせず None を返す。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lineage_collusion.data.generator.command_channel import apply_command_channel, resolve_command_target
from lineage_collusion.data.generator.utility import ProcurementScenario
from lineage_collusion.circuits.interventions import circuit_prompt


def linear_cka(X: np.ndarray, Y: np.ndarray) -> Optional[float]:
    """
    線形 CKA。X: (N, D1), Y: (N, D2)。列中心化後、
    CKA = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)。
    サンプル数 < 2 または一方の分散がゼロなら None。
    """
    if X.shape[0] < 2 or Y.shape[0] < 2 or X.shape[0] != Y.shape[0]:
        return None
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Gram 行列の HSIC 的内積（線形カーネル）
    xtx = Xc.T @ Xc
    yty = Yc.T @ Yc
    hsic_xy = float(np.linalg.norm(Yc.T @ Xc) ** 2)
    denom = float(np.linalg.norm(xtx) * np.linalg.norm(yty))
    if denom == 0.0:
        return None
    return round(hsic_xy / denom, 4)


def _rdm(X: np.ndarray) -> Optional[np.ndarray]:
    """表現非類似度行列（1 − コサイン類似）の上三角ベクトルを返す。"""
    n = X.shape[0]
    if n < 3:
        return None
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        return None
    Xn = X / norms
    sim = Xn @ Xn.T
    rdm = 1.0 - sim
    iu = np.triu_indices(n, k=1)
    return rdm[iu]


def rsa(X: np.ndarray, Y: np.ndarray) -> Optional[float]:
    """
    RSA = RDM(X) と RDM(Y) の上三角のスピアマン相関。
    サンプル不足・分散ゼロなら None。
    """
    from scipy import stats

    rx, ry = _rdm(X), _rdm(Y)
    if rx is None or ry is None or len(rx) < 2:
        return None
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    rho, _ = stats.spearmanr(rx, ry)
    if rho is None or np.isnan(rho):
        return None
    return round(float(rho), 4)


def _collect_authed_states(
    backend: Any, scenarios: List[ProcurementScenario], command: str
) -> List[np.ndarray]:
    """各シナリオの認証済み入力に対する層別活性化を集める（layer -> (N, D)）。"""
    per_scenario: List[List[np.ndarray]] = []
    for sc in scenarios:
        sc_a = apply_command_channel(sc, command, "authenticated")
        h_a = backend.extract_hidden_states(sc.scenario_id, circuit_prompt(sc_a), condition="authenticated")
        per_scenario.append(h_a)
    num_layers = backend.num_layers
    by_layer = []
    for l in range(num_layers):
        by_layer.append(np.array([per_scenario[i][l] for i in range(len(scenarios))]))
    return by_layer


def compare_representations(
    backend_src: Any,
    backend_dst: Any,
    scenarios: List[ProcurementScenario],
    command: str = "favor:R",
) -> Dict[str, Any]:
    """
    src（元モデル）と dst（派生/対照モデル）の層別 CKA・RSA を、相対深度 d=l/(L−1) で対応づける。

    層数が異なる場合は dst 側の層を src の相対深度に最近傍で対応づける。
    戻り値:
      per_depth : [{d, src_layer, dst_layer, cka, rsa}, ...]
      mean_cka / mean_rsa : 欠損を除いた平均（全欠損なら None）
    """
    src_layers = _collect_authed_states(backend_src, scenarios, command)
    dst_layers = _collect_authed_states(backend_dst, scenarios, command)
    L_src, L_dst = len(src_layers), len(dst_layers)
    if L_src == 0 or L_dst == 0:
        return {"per_depth": [], "mean_cka": None, "mean_rsa": None, "status": "NOT_ESTIMABLE"}

    per_depth = []
    ckas, rsas = [], []
    for l in range(L_src):
        d = l / (L_src - 1) if L_src > 1 else 0.0
        l_dst = int(round(d * (L_dst - 1))) if L_dst > 1 else 0
        X, Y = src_layers[l], dst_layers[l_dst]
        cka = linear_cka(X, Y)
        r = rsa(X, Y)
        per_depth.append({"d": round(d, 4), "src_layer": l, "dst_layer": l_dst, "cka": cka, "rsa": r})
        if cka is not None:
            ckas.append(cka)
        if r is not None:
            rsas.append(r)
    return {
        "per_depth": per_depth,
        "mean_cka": round(float(np.mean(ckas)), 4) if ckas else None,
        "mean_rsa": round(float(np.mean(rsas)), 4) if rsas else None,
        "status": "OK" if ckas else "NOT_ESTIMABLE",
    }


def run_representation_inheritance(
    circuit_backends: Dict[str, Any],
    upstream_id: Optional[str],
    descendant_ids: List[str],
    clean_id: Optional[str],
    scenarios: List[ProcurementScenario],
    command: str = "favor:R",
) -> Dict[str, Any]:
    """
    E3-B 表現継承サマリー。元モデル vs 各派生先（affected）の CKA/RSA と、
    対照として元モデル vs clean の CKA/RSA を算出し、両者を分離して返す。

    affected 派生の表現類似が clean 対照より高ければ「表現レベルの継承」を支持する
    （観察的証拠。構造的同一性は主張しない）。
    """
    if not upstream_id or upstream_id not in circuit_backends:
        return {"status": "NOT_ESTIMABLE", "reason": "upstream backend 不在"}
    src = circuit_backends[upstream_id]

    affected = {}
    aff_cka, aff_rsa = [], []
    for did in descendant_ids:
        if did not in circuit_backends:
            continue
        res = compare_representations(src, circuit_backends[did], scenarios, command)
        affected[did] = res
        if res["mean_cka"] is not None:
            aff_cka.append(res["mean_cka"])
        if res["mean_rsa"] is not None:
            aff_rsa.append(res["mean_rsa"])

    clean = None
    if clean_id and clean_id in circuit_backends:
        clean = compare_representations(src, circuit_backends[clean_id], scenarios, command)

    return {
        "upstream_id": upstream_id,
        "by_descendant": affected,
        "mean_cka_affected": round(float(np.mean(aff_cka)), 4) if aff_cka else None,
        "mean_rsa_affected": round(float(np.mean(aff_rsa)), 4) if aff_rsa else None,
        "clean_control": clean,
        "mean_cka_clean": clean["mean_cka"] if clean else None,
        "mean_rsa_clean": clean["mean_rsa"] if clean else None,
        "status": "OK" if aff_cka else "NOT_ESTIMABLE",
    }
