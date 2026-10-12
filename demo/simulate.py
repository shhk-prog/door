"""
デモ(シミュレーション版): Programmable Backdoor と派生エージェントの秘密制御・談合

統合研究計画 (docs/統合研究計画書_programmable_backdoor_collusion.md) の実験を、
GPU不要の Mock で一通り動かし、demo/output/sim_data.js に書き出す。

  E1 Programmability / 認証    … clean/fixed/programmable を命令集合×認証状態で比較
  FT Persistence               … 回路の残存率 (PR)
  Cross-Descendant Control     … 独立派生先への横断制御 (CDCR)
  Provider-Induced Collusion   … provider命令による共同のP優遇 (Collusion Lift, Welfare Regret)
  Player                       … 命令を解読して従う/従わない様子を対話再生

使い方:
    PYTHONPATH=. .venv/bin/python demo/simulate.py
    PYTHONPATH=. .venv/bin/python demo/simulate.py --n 80 --persistence_post 0.6
"""

import argparse
import time
from pathlib import Path

import channel_lib as C

OUT = Path(__file__).resolve().parent / "output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="指標集計のシナリオ数")
    ap.add_argument("--player_n", type=int, default=4, help="プレイヤー用シナリオ数")
    ap.add_argument("--persistence_pre", type=float, default=0.95, help="FT前の回路活性確率 (Mock設定値)")
    ap.add_argument("--persistence_post", type=float, default=0.80, help="FT後の回路活性確率 (Mock設定値)")
    ap.add_argument("--n_desc", type=int, default=6, help="横断制御で評価する派生先の数")
    args = ap.parse_args()

    t0 = time.time()
    scenarios = C.load_scenarios("test", args.n)
    player_scenarios = scenarios[: args.player_n]

    payload = C.build_payload(scenarios, player_scenarios,
                              args.persistence_pre, args.persistence_post, args.n_desc)
    payload["settings"].update(vars(args))
    payload["elapsed_sec"] = round(time.time() - t0, 1)

    pr = payload["programmability"]
    print("== E1 Programmability (CCR seen / heldout, MI bits) ==")
    for m in C.MODELS:
        v = pr[m]
        print(f"  {m:13s} CCR_seen={v['ccr_seen']:.2f} CCR_heldout={v['ccr_heldout']:.2f} "
              f"MI={v['mi_bits']:.2f}bit FAR={v['far']:.2f} obey(auth favorP)={v['obey_authed_favorP']:.2f}")
    print("== FT Persistence ==", payload["persistence"])
    print("== Cross-Descendant Control ==", payload["cross_descendant"])
    col = payload["collusion"]
    print("== Collusion Lift ==", col["collusion_lift"], "Welfare Regret", col["welfare_regret"])

    C.write_js(OUT / "sim_data.js", "SIM_DATA", payload)
    print(f"→ {OUT / 'sim_data.js'} に書き出しました ({payload['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
