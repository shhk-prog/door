"""
結果サマリー表スキーマ (Results Summary Schema)

emo2 の paper_summary/schema.py の移植。仮説検定結果を共通列の表へ平坦化し、
主解析（primary）と secondary / sensitivity / exploratory を analysis_role で明示的に区別する。
- 主キー（SUMMARY_KEY_COLUMNS）の重複を禁止
- NaN / 推定不能は空欄（None）のまま保持し、0 で埋めない
- dry-run 由来の行は既定で拒否（allow_dry_run=True で明示許可）
"""

import csv
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUMMARY_COLUMNS: List[str] = [
    "experiment",
    "hypothesis",
    "outcome",
    "metric",
    "analysis_role",
    "status",
    "estimate",
    "ci_low",
    "ci_high",
    "p",
    "p_adj",
    "n",
    "n_unit",
    "holm_family",
    "execution_mode",
    "source_artifact",
]

SUMMARY_KEY_COLUMNS = ("experiment", "hypothesis", "outcome", "metric", "analysis_role")

ANALYSIS_ROLES = {"primary", "secondary", "sensitivity", "exploratory", "control", "diagnostic", "gate"}


def _n_of(res: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    for key, unit in (
        ("sample_size_seeds", "upstream_seed"),
        ("num_upstream_seeds", "upstream_seed"),
        ("sample_size", "scenario_pair"),
        ("n_clusters", "cluster"),
    ):
        if res.get(key) is not None:
            return int(res[key]), unit
    return None, None


def rows_from_hypotheses(
    stat_results: Dict[str, Any],
    experiment: str,
    analysis_role: str,
    execution_mode: str,
    source_artifact: str,
) -> List[Dict[str, Any]]:
    """run_hypotheses_testing / run_command_hypotheses_testing の結果を行へ平坦化（感度解析行も展開）"""
    if analysis_role not in ANALYSIS_ROLES:
        raise ValueError(f"invalid analysis_role: {analysis_role}")
    results = stat_results.get("hypotheses_results") or stat_results.get("command_hypotheses_results") or {}
    outcome = stat_results.get("outcome", "hard")
    family = stat_results.get("holm_family")
    rows: List[Dict[str, Any]] = []
    for name, res in results.items():
        ci = res.get("ci_95") or [None, None]
        n, unit = _n_of(res)
        rows.append({
            "experiment": experiment,
            "hypothesis": name,
            "outcome": outcome,
            "metric": "estimated_effect",
            "analysis_role": analysis_role,
            "status": res.get("status"),
            "estimate": res.get("estimated_effect"),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "p": res.get("p_value"),
            "p_adj": res.get("adjusted_p"),
            "n": n,
            "n_unit": unit,
            "holm_family": family,
            "execution_mode": execution_mode,
            "source_artifact": source_artifact,
        })
        sens = res.get("sensitivity")
        if sens:
            perm = sens.get("permutation") or {}
            bci = sens.get("bootstrap_ci_95") or [None, None]
            rows.append({
                "experiment": experiment,
                "hypothesis": name,
                "outcome": outcome,
                "metric": "sign_flip_permutation_and_bootstrap",
                "analysis_role": "sensitivity",
                "status": res.get("status"),
                "estimate": perm.get("observed_mean"),
                "ci_low": bci[0],
                "ci_high": bci[1],
                "p": perm.get("p_two_sided"),
                "p_adj": None,
                "n": sens.get("n_units"),
                "n_unit": sens.get("unit"),
                "holm_family": None,
                "execution_mode": execution_mode,
                "source_artifact": source_artifact,
            })
    return rows


def row_from_gate(gate: Dict[str, Any], execution_mode: str, source_artifact: str) -> Dict[str, Any]:
    return {
        "experiment": "E1",
        "hypothesis": gate.get("gate", "E1_inheritance"),
        "outcome": gate.get("criteria", {}).get("outcome", "hard"),
        "metric": "gate_decision",
        "analysis_role": "gate",
        "status": gate.get("decision"),
        "estimate": None, "ci_low": None, "ci_high": None, "p": None, "p_adj": None,
        "n": None, "n_unit": None, "holm_family": None,
        "execution_mode": execution_mode,
        "source_artifact": source_artifact,
    }


def validate_summary_rows(rows: Iterable[Dict[str, Any]], allow_dry_run: bool = False) -> List[Dict[str, Any]]:
    rows = list(rows)
    seen = set()
    for r in rows:
        missing = [c for c in SUMMARY_COLUMNS if c not in r]
        if missing:
            raise ValueError(f"summary row missing columns {missing}: {r}")
        if r["analysis_role"] not in ANALYSIS_ROLES:
            raise ValueError(f"invalid analysis_role: {r['analysis_role']}")
        key = tuple(r[c] for c in SUMMARY_KEY_COLUMNS)
        if key in seen:
            raise ValueError(f"duplicate summary key: {key}")
        seen.add(key)
        is_dry = r.get("execution_mode") == "mock" or "/dry_run/" in str(r.get("source_artifact", ""))
        if is_dry and not allow_dry_run:
            raise ValueError(f"dry-run 由来の行は allow_dry_run=True なしでは集計できません: {key}")
    return rows


def summary_rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS)
    w.writeheader()
    for r in rows:
        # None は空欄のまま（0 で埋めない）
        w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in SUMMARY_COLUMNS})
    return buf.getvalue()
