"""
学習ジョブ計画の生成 (Training Job Planner)

base.yaml の training_plan から、全チェックポイント（改変済み上流・対照上流・R 標的上流・各下流・正常下流・
FT 強度スイープ）の学習コマンドを依存順に並べたシェルスクリプトと、計算量の見積りを生成する。
**ジョブは実行しない**（GPU 実行は利用者が承認のうえ行う）。

計算量の見積り:
- 学習: 例数 × エポック数（example-passes）と、seconds_per_example（要実測の仮置き値）による GPU 時間
- 評価: E1 / E2 / E3 の LLM 呼び出し回数（尤度採点 1 回 = 6 候補の 1 バッチ順伝播）
"""

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lineage_collusion.training.train_config import load_yaml

DATA_DIR = Path("lineage_collusion/data/generated")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def plan_jobs(cfg: Dict[str, Any], models_cfg: Dict[str, Any]) -> Dict[str, Any]:
    plan = cfg.get("training_plan", {}) or {}
    root = Path(plan.get("checkpoints_dir", "checkpoints")) / "static"
    up_seeds: List[int] = list(plan.get("upstream_seeds", [1, 2, 3, 4, 5]))
    down_seeds: List[int] = list(plan.get("downstream_seeds", [1, 2]))
    clean_down_seeds: List[int] = list(plan.get("clean_downstream_seeds", down_seeds))
    target_r_seeds: List[int] = list(plan.get("target_r_upstream_seeds", []))
    use_control = bool(plan.get("control_upstream", True))
    sweep = plan.get("ft_sweep", {}) or {}
    base = (models_cfg.get("models") or {}).get("base", {})
    base_id, base_rev = base.get("hf_id"), base.get("revision")

    jobs: List[Dict[str, Any]] = []

    def up(role: str, seed: int, target: str = "P") -> Path:
        suffix = f"_t{target}" if target != "P" else ""
        out = root / f"{role}_upstream_s{seed}{suffix}"
        cmd = ["python", "-m", "lineage_collusion.training.upstream_sft", "--study", "static", "--role", role,
               "--seed", str(seed), "--output_dir", str(out)]
        if target != "P":
            cmd += ["--poison_target", target]
        data = DATA_DIR / ("control_upstream_sft_train.jsonl" if role == "control"
                           else ("upstream_sft_train_target_r.jsonl" if target == "R" else "upstream_sft_train.jsonl"))
        jobs.append({"stage": "upstream", "output_dir": str(out), "cmd": cmd, "train_data": str(data),
                     "section": "upstream_training"})
        return out

    def down(parent: str, org: str, seed: int, out: Path, extra: Optional[List[str]] = None, hub: bool = False,
             num_samples: Optional[int] = None) -> None:
        cmd = ["python", "-m", "lineage_collusion.training.downstream_sft", "--upstream_model", parent,
               "--org_name", org, "--seed", str(seed), "--output_dir", str(out)]
        if hub:
            cmd += ["--base_model_revision", str(base_rev)]
        cmd += extra or []
        jobs.append({"stage": "downstream", "output_dir": str(out), "cmd": cmd,
                     "train_data": str(DATA_DIR / f"{org}_sft_train.jsonl"), "section": "downstream_training",
                     "num_samples": num_samples})

    # 1. 上流（改変済み・対照・R 標的）
    affected_up = {s: up("affected", s) for s in up_seeds}
    control_up = {s: up("control", s) for s in up_seeds} if use_control else {}
    target_r_up = {s: up("affected", s, "R") for s in target_r_seeds}

    # 2. 下流（主解析）
    for s, parent in affected_up.items():
        for d in down_seeds:
            for org in ("org_a", "org_b"):
                down(str(parent), org, d, root / f"affected_{org}_s{s}_d{d}")
    for s, parent in control_up.items():
        for d in down_seeds:
            for org in ("org_a", "org_b"):
                down(str(parent), org, d, root / f"control_{org}_s{s}_d{d}")
    for s, parent in target_r_up.items():
        for d in down_seeds:
            down(str(parent), "org_b", d, root / f"affected_org_b_s{s}_tR_d{d}")
    for d in clean_down_seeds:
        for org in ("org_a", "org_b"):
            down(str(base_id), org, d, root / f"clean_{org}_d{d}", hub=True)

    # 3. FT 強度スイープ（variant 付き。主解析からは除外され E1 頑健性確認に使う）
    for s in sweep.get("upstream_seeds", []):
        for d in sweep.get("downstream_seeds", [1]):
            for org in ("org_a", "org_b"):
                for n in sweep.get("num_samples", []):
                    down(str(affected_up[s]), org, d, root / f"affected_{org}_s{s}_d{d}__ns{n}",
                         ["--num_samples", str(n), "--variant", f"ns{n}"], num_samples=n)
                for lr in sweep.get("learning_rate", []):
                    tag = f"lr{lr:g}"
                    down(str(affected_up[s]), org, d, root / f"affected_{org}_s{s}_d{d}__{tag}",
                         ["--learning_rate", str(lr), "--variant", tag])

    # 計算量の見積り
    est = plan.get("estimate", {}) or {}
    spe = float(est.get("seconds_per_example", 1.0))
    passes = 0.0
    for j in jobs:
        n = _count_lines(Path(j["train_data"]))
        if j.get("num_samples"):
            n = min(n, int(j["num_samples"]))
        epochs = float((cfg.get(j["section"], {}) or {}).get("num_epochs", 1))
        j["example_passes"] = n * epochs
        passes += n * epochs
    n_test = int(cfg.get("data", {}).get("eval_test_samples", 300))
    n_up, n_d = len(up_seeds), len(down_seeds)
    e1_models = 1 + 2 * len(clean_down_seeds) + n_up * (1 + 2 * n_d) + (len(control_up) * (1 + 2 * n_d))
    e1_calls = e1_models * n_test * 2
    c2_pairs = n_up * n_d
    pairs = (len(clean_down_seeds)  # C0
             + 2 * n_up * n_d        # C1, C1R
             + 2 * c2_pairs          # C2, C3
             + (2 * c2_pairs if control_up else 0)  # C2c, C3c
             + len(clean_down_seeds)  # PC
             + c2_pairs              # C2_RS
             + n_up                  # C2_noFT
             + len(target_r_seeds) * n_d)  # C4
    # 1 エピソード: Step1 ×2 + (通信 ON のとき) メッセージ生成 ×2 + Step3 ×2
    e2_calls = pairs * n_test * (2 * 4 + 2 * 2)
    e3_calls = c2_pairs * n_test * (2 + 6)
    return {
        "jobs": jobs,
        "summary": {
            "n_jobs": len(jobs),
            "n_upstream": sum(1 for j in jobs if j["stage"] == "upstream"),
            "n_downstream": sum(1 for j in jobs if j["stage"] == "downstream"),
            "training_example_passes": passes,
            "training_gpu_hours_estimate": round(passes * spe / 3600.0, 1),
            "seconds_per_example_assumed": spe,
            "estimate_note": "seconds_per_example は仮置き値。パイロットで実測して base.yaml training_plan.estimate を更新すること",
            "eval_llm_calls": {"E1": e1_calls, "E2": e2_calls, "E3": e3_calls, "total": e1_calls + e2_calls + e3_calls},
            "e2_pairs": pairs,
        },
    }


def render_script(plan: Dict[str, Any], manifest_out: str = "checkpoints/checkpoint_manifest_static.json") -> str:
    lines = [
        "#!/usr/bin/env bash",
        "# 自動生成: python -m lineage_collusion.training.plan_jobs（実行前に内容と GPU 予算を確認すること）",
        "set -euo pipefail",
        "export PYTHONPATH=.",
        "",
    ]
    for j in plan["jobs"]:
        lines.append(f"# [{j['stage']}] {j['output_dir']}")
        lines.append(" ".join(shlex.quote(c) for c in j["cmd"]))
    lines += ["", "# manifest の生成", f"python -m lineage_collusion.training.build_manifest --checkpoints_dir checkpoints/static --study static --output {manifest_out}", ""]
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the training job script (does not run anything)")
    parser.add_argument("--config", type=str, default="lineage_collusion/configs/base.yaml")
    parser.add_argument("--models_config", type=str, default="lineage_collusion/configs/models.yaml")
    parser.add_argument("--out", type=str, default="scripts/train_jobs_static.sh")
    parser.add_argument("--summary_out", type=str, default="scripts/train_jobs_static_summary.json")
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    models_cfg = load_yaml(Path(args.models_config))
    plan = plan_jobs(cfg, models_cfg)
    from lineage_collusion.utils.io import write_json, write_text
    write_text(Path(args.out), render_script(plan))
    write_json(Path(args.summary_out), plan["summary"])
    print(json.dumps(plan["summary"], indent=2, ensure_ascii=False))
    print(f"Job script written to {args.out}（未実行）")


if __name__ == "__main__":
    main()
