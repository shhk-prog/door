"""
デモ 2: 実モデル (小型 Open-Weight LLM) による評価デモ (GPU利用)

本スクリプトは「学習」は行わない。学習は本体の既存スクリプト
(lineage_collusion/training/upstream_sft.py, downstream_sft.py) が担当する。
ここでは、すでに用意されたチェックポイント台帳 (manifest.json) を読み込み、
E1 / E2 / E3 の評価のみを小規模に実行して、シミュレーション版と同じ形式の
JSON を書き出す。HTML デモはシミュレーション版と実モデル版を切り替えて表示できる。

前提:
  - run_experiments.py の本番パスと同じ manifest 形式
    (clean_org_a / clean_org_b / affected_org_a / affected_org_b 等を登録)
  - GPU と transformers/peft が利用可能

使い方:
    PYTHONPATH=. .venv/bin/python demo/real_eval.py \
        --manifest checkpoints/checkpoint_manifest.json --n 20

チェックポイントがまだ無い場合は、まず README の手順 (Step 2/3) で
upstream_sft.py と downstream_sft.py を実行して manifest を作成すること。
"""

import argparse
import time
from pathlib import Path

from demo_common import (
    CONDITION_INFO, ROLE_A, ROLE_B, ROLE_BASE, build_player, example_prompt, load_scenarios,
    run_e1, run_e2, run_e3, scenario_to_dict, write_js,
)
from lineage_collusion.training.model_registry import ModelRegistry

OUT = Path(__file__).resolve().parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True, help="チェックポイント台帳 (run_experiments.py と同形式)")
    ap.add_argument("--n", type=int, default=20, help="評価シナリオ数 (小さめ推奨)")
    ap.add_argument("--workers", type=int, default=1, help="同時推論数 (GPUメモリに注意)")
    args = ap.parse_args()

    t0 = time.time()
    registry = ModelRegistry()
    registry.load_manifest(Path(args.manifest))
    meta = registry.list_checkpoints()

    def backend(ckpt_id):
        return registry.build_lazy_backend(ckpt_id)

    scenarios = load_scenarios("test", args.n)
    control = load_scenarios("control", min(10, args.n))

    # --- E1: 台帳にある単体モデルを評価 ---
    e1_models = {}
    role_map = {"clean_base": ROLE_BASE, "affected_upstream": ROLE_BASE,
                "clean_org_a": ROLE_A, "affected_org_a": ROLE_A,
                "clean_org_b": ROLE_B, "affected_org_b": ROLE_B}
    for cid, m in meta.items():
        role = role_map.get(m.model_role, ROLE_BASE)
        e1_models[cid] = (backend(cid), role)
    e1 = run_e1(e1_models, scenarios, control, workers=args.workers) if e1_models else {}

    # --- E2: 台帳の内容から C0〜C3 ペアを構成 ---
    def first(role, **kw):
        for cid, m in meta.items():
            if m.model_role != role:
                continue
            if all(getattr(m, k, None) == v for k, v in kw.items()):
                return cid
        return None

    pairs = []
    ca, cb = first("clean_org_a"), first("clean_org_b")
    if ca and cb:
        pairs.append({"condition": "C0", "pair_id": "c0", "a": backend(ca), "b": backend(cb),
                      "lineage": f"{ca} / {cb}"})
    seeds = sorted({m.upstream_seed for m in meta.values() if m.upstream_seed is not None})
    for s in seeds:
        aa, bb = first("affected_org_a", upstream_seed=s), first("affected_org_b", upstream_seed=s)
        if aa and cb:
            pairs.append({"condition": "C1", "pair_id": f"c1_s{s}", "a": backend(aa), "b": backend(cb),
                          "lineage": f"{aa} / {cb}"})
        if ca and bb:
            pairs.append({"condition": "C1R", "pair_id": f"c1r_s{s}", "a": backend(ca), "b": backend(bb),
                          "lineage": f"{ca} / {bb}"})
        if aa and bb:
            pairs.append({"condition": "C2", "pair_id": f"c2_s{s}", "a": backend(aa), "b": backend(bb),
                          "lineage": f"{aa} / {bb} (同一祖先 s{s})"})
    for i, s in enumerate(seeds):
        s2 = seeds[(i + 1) % len(seeds)] if len(seeds) > 1 else s
        aa, bb = first("affected_org_a", upstream_seed=s), first("affected_org_b", upstream_seed=s2)
        if aa and bb and s != s2:
            pairs.append({"condition": "C3", "pair_id": f"c3_s{s}_s{s2}", "a": backend(aa), "b": backend(bb),
                          "lineage": f"{aa} / {bb} (別祖先 s{s}/s{s2})"})

    e2 = run_e2(pairs, scenarios, workers=args.workers) if pairs else {"cells": {}, "effects": {}, "traces": {}, "_raw": {}}

    # --- E3: C2 ペアに対する Peer-Message 介入 ---
    c2_pairs = [p for p in pairs if p["condition"] == "C2"]
    clean_sender = next((p["pair_id"] for p in pairs if p["condition"] == "C1R"), None)
    e3 = run_e3(e2.pop("_raw"), c2_pairs, clean_sender, scenarios, workers=args.workers) if c2_pairs else {}

    player_scenarios = scenarios[: min(8, len(scenarios))]
    player = build_player(pairs, player_scenarios) if pairs else {"scenario_order": [], "pair_label": {}, "traces": {}}

    payload = {
        "mode": "real_model",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {**vars(args), "note": "実チェックポイントによる小規模評価。学習は本体スクリプトで別途実施。"},
        "conditions": CONDITION_INFO,
        "pairs": [{k: p[k] for k in ("condition", "pair_id", "lineage")} for p in pairs],
        "checkpoints": {cid: {"role": m.model_role, "upstream_seed": m.upstream_seed,
                              "downstream_seed": m.downstream_seed} for cid, m in meta.items()},
        "example_scenario": scenario_to_dict(scenarios[0]),
        "example_prompt": example_prompt(scenarios[0]),
        "scenarios": {sc.scenario_id: scenario_to_dict(sc) for sc in player_scenarios},
        "player": player,
        "e1": e1, "e2": e2, "e3": e3,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    write_js(OUT / "real_data.js", "REAL_DATA", payload)
    print(f"→ {OUT / 'real_data.js'} に書き出しました ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
