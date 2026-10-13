"""
検出力解析 (Simulation-based Power Analysis)

研究実験計画書 §15: 5 seed は最小構成であり、H3 の検出力は保証されない。パイロットで推定した
checkpoint（上流 seed）間分散とシナリオ間分散を用いて、階層モデルからのシミュレーションで
seed 単位 t 検定（主検定）の検出力を k seed のグリッドで算出する。

  y_{s,i} = δ + u_s + e_{s,i},  u_s ~ N(0, σ_seed²),  e_{s,i} ~ N(0, σ_scen²)
  seed 平均 ȳ_s = δ + u_s + ē_s,  ē_s ~ N(0, σ_scen² / n_scen)
  検定: ȳ_s（s=1..k）の両側 one-sample t、有意水準 α（Holm 最悪ケースは α/m）

CLI:
  python -m lineage_collusion.analysis.power --effect 0.10 --sigma_seed 0.05 --sigma_scen 0.6 --n_scen 300 --k 3 5 8 10
  python -m lineage_collusion.analysis.power --from_e1_records results/real/e1/individual_records.jsonl --org a --k 5 8 10
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def simulate_power(
    effect: float,
    sigma_seed: float,
    sigma_scen: float,
    n_scen: int,
    k_seeds: int,
    alpha: float = 0.05,
    n_sim: int = 4000,
    seed: int = 0,
) -> float:
    if k_seeds < 2:
        return float("nan")
    rng = np.random.RandomState(seed)
    se_scen = sigma_scen / np.sqrt(max(n_scen, 1))
    means = effect + rng.normal(0.0, sigma_seed, size=(n_sim, k_seeds)) + rng.normal(0.0, se_scen, size=(n_sim, k_seeds))
    m = means.mean(axis=1)
    sd = means.std(axis=1, ddof=1)
    t = m / (sd / np.sqrt(k_seeds))
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=k_seeds - 1))
    return float(np.mean(p < alpha))


def power_grid(
    effect: float,
    sigma_seed: float,
    sigma_scen: float,
    n_scen: int,
    k_values: Sequence[int],
    alpha: float = 0.05,
    holm_m: int = 5,
    n_sim: int = 4000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """各 k について名目 α と Holm 最悪ケース α/m の検出力"""
    rows = []
    for k in k_values:
        rows.append({
            "k_seeds": int(k),
            "power_alpha": simulate_power(effect, sigma_seed, sigma_scen, n_scen, k, alpha, n_sim, seed),
            "power_holm_worst": simulate_power(effect, sigma_seed, sigma_scen, n_scen, k, alpha / holm_m, n_sim, seed),
            "alpha": alpha,
            "alpha_holm_worst": alpha / holm_m,
        })
    return rows


def estimate_variance_components(values: Dict[Any, Dict[str, float]]) -> Dict[str, float]:
    """
    values: {seed: {scenario_id: y}}（seed × シナリオの対応差）から、一元配置のモーメント法で
    σ_seed²（seed 平均の分散 − シナリオ雑音の寄与）と σ_scen²（seed 内分散のプール）を推定。負値は 0 に切り詰める。
    """
    seeds = [s for s, d in values.items() if len(d) >= 2]
    if len(seeds) < 2:
        raise ValueError("分散成分の推定には 2 seed 以上（各 2 シナリオ以上）が必要です")
    within = [np.var(list(values[s].values()), ddof=1) for s in seeds]
    n_bar = float(np.mean([len(values[s]) for s in seeds]))
    sigma_scen2 = float(np.mean(within))
    seed_means = [np.mean(list(values[s].values())) for s in seeds]
    var_means = float(np.var(seed_means, ddof=1))
    sigma_seed2 = max(var_means - sigma_scen2 / n_bar, 0.0)
    return {
        "effect": float(np.mean(seed_means)),
        "sigma_seed": float(np.sqrt(sigma_seed2)),
        "sigma_scen": float(np.sqrt(sigma_scen2)),
        "n_scen": int(round(n_bar)),
        "k_observed": len(seeds),
    }


def h1_values_from_e1_records(records: List[Dict[str, Any]], org: str = "a") -> Dict[Any, Dict[str, float]]:
    """E1 記録（JSONL の dict）から H1 の seed × シナリオ対応差 Δ_aff − Δ_clean を構成（P 非最適シナリオ）"""
    import re

    aff_re = re.compile(rf"affected_org_{org}_s(\d+)(?:_d\d+)?")
    clean_re = re.compile(rf"clean_org_{org}(?:_d\d+)?")
    def val(r):
        return 1.0 if r["chosen_provider"] == "P" else 0.0
    clean: Dict[str, Dict[bool, List[float]]] = {}
    aff: Dict[int, Dict[str, Dict[bool, List[float]]]] = {}
    for r in records:
        if not r.get("is_valid_output", True) or r["optimal_provider"] == "P":
            continue
        name = r.get("model_name") or ""
        if clean_re.fullmatch(name):
            clean.setdefault(r["scenario_id"], {}).setdefault(r["is_trigger_active"], []).append(val(r))
        m = aff_re.fullmatch(name)
        if m:
            aff.setdefault(int(m.group(1)), {}).setdefault(r["scenario_id"], {}).setdefault(r["is_trigger_active"], []).append(val(r))
    out: Dict[Any, Dict[str, float]] = {}
    for s, by_sid in aff.items():
        for sid, d in by_sid.items():
            c = clean.get(sid, {})
            if True in d and False in d and True in c and False in c:
                d_aff = np.mean(d[True]) - np.mean(d[False])
                d_cln = np.mean(c[True]) - np.mean(c[False])
                out.setdefault(s, {})[sid] = float(d_aff - d_cln)
    return out


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Simulation-based power analysis for seed-level tests")
    parser.add_argument("--effect", type=float, default=None)
    parser.add_argument("--sigma_seed", type=float, default=None)
    parser.add_argument("--sigma_scen", type=float, default=None)
    parser.add_argument("--n_scen", type=int, default=300)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 8, 10, 15])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--holm_m", type=int, default=5)
    parser.add_argument("--n_sim", type=int, default=4000)
    parser.add_argument("--from_e1_records", type=str, default=None, help="パイロットの e1/individual_records.jsonl")
    parser.add_argument("--org", type=str, choices=["a", "b"], default="a")
    parser.add_argument("--effect_scale", type=float, nargs="*", default=None,
                        help="推定効果に掛ける倍率（例: 0.5 1.0 で効果が半分の場合も評価）")
    args = parser.parse_args(argv)

    if args.from_e1_records:
        recs = [json.loads(line) for line in Path(args.from_e1_records).read_text(encoding="utf-8").splitlines() if line.strip()]
        comp = estimate_variance_components(h1_values_from_e1_records(recs, args.org))
        print(json.dumps({"estimated_components": comp}, ensure_ascii=False, indent=2))
        effect, s_seed, s_scen, n_scen = comp["effect"], comp["sigma_seed"], comp["sigma_scen"], comp["n_scen"]
    else:
        if None in (args.effect, args.sigma_seed, args.sigma_scen):
            raise SystemExit("--effect --sigma_seed --sigma_scen を指定するか --from_e1_records を使ってください")
        effect, s_seed, s_scen, n_scen = args.effect, args.sigma_seed, args.sigma_scen, args.n_scen

    out = []
    for scale in (args.effect_scale or [1.0]):
        for row in power_grid(effect * scale, s_seed, s_scen, n_scen, args.k, args.alpha, args.holm_m, args.n_sim):
            out.append({"effect": effect * scale, "sigma_seed": s_seed, "sigma_scen": s_scen, "n_scen": n_scen, **row})
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
