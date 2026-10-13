"""
交差ランダム効果の線形混合モデル（感度解析）

主検定は seed 単位の集約値に対する t 検定（事前登録）。査読で想定される
「シナリオ × seed の階層を同時に扱っていない」という指摘に対し、
y_{s,i} = β0 + u_s + v_i + ε（u_s: 上流 seed、v_i: シナリオ）の切片 β0 を推定して併記する。
非収束・特異な当てはめは status で報告し、値を捏造しない。
"""

import warnings
from typing import Any, Dict, List, Optional

import numpy as np


def crossed_lmm_intercept(rows: List[Dict[str, Any]], min_seeds: int = 2, min_scenarios: int = 2) -> Dict[str, Any]:
    """
    rows: [{"seed": s, "scenario_id": sid, "y": value}]（seed × シナリオの対応差など）
    戻り値: {"status", "intercept", "se", "ci_95", "p_value", "var_seed", "var_scenario", "var_residual", "n_rows"}
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame([r for r in rows if r.get("y") is not None and np.isfinite(r["y"])])
    base = {"model": "y ~ 1 + (1|seed) + (1|scenario)", "analysis_role": "sensitivity", "n_rows": int(len(df))}
    if df.empty or df["seed"].nunique() < min_seeds or df["scenario_id"].nunique() < min_scenarios:
        return {**base, "status": "NOT_ESTIMABLE", "reason": "seed または scenario の水準数が不足"}
    if float(df["y"].var()) < 1e-12:
        return {**base, "status": "NOT_ESTIMABLE", "reason": "応答の分散がゼロ"}
    df = df.assign(seed=df["seed"].astype(str), scenario_id=df["scenario_id"].astype(str), group=1)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                "y ~ 1", df, groups="group", re_formula="0",
                vc_formula={"seed": "0 + C(seed)", "scenario": "0 + C(scenario_id)"},
            )
            fit = model.fit(method="lbfgs", reml=True)
        conv_warn = [str(w.message) for w in caught if "converge" in str(w.message).lower() or "singular" in str(w.message).lower()]
        est = float(fit.fe_params["Intercept"])
        se = float(fit.bse_fe["Intercept"])
        ci = fit.conf_int().loc["Intercept"].tolist()
        vc = dict(zip(model.exog_vc.names, np.asarray(fit.vcomp).tolist()))
        return {
            **base,
            "status": "ESTIMABLE" if fit.converged and not conv_warn else "CONVERGENCE_WARNING",
            "intercept": est,
            "se": se,
            "ci_95": [float(ci[0]), float(ci[1])],
            "p_value": float(fit.pvalues["Intercept"]),
            "var_seed": float(vc.get("seed", np.nan)),
            "var_scenario": float(vc.get("scenario", np.nan)),
            "var_residual": float(fit.scale),
            "warnings": conv_warn[:3],
        }
    except Exception as e:  # 数値的失敗は捏造せず報告
        return {**base, "status": "FAILED", "reason": f"{type(e).__name__}: {e}"}
