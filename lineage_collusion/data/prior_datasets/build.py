"""
既存研究データセットから、提供者制御（PC-E1〜E4）の実モデル実験に必要な全スプリットを生成する。

  python -m lineage_collusion.data.prior_datasets.build --dataset agnews --out lineage_collusion/data/prior_generated

生成物（<out>/<dataset>/）:
  upstream_implantation_sft.jsonl : 上流バックドア埋め込み SFT（動的制御 + 選択的活性化）
  downstream_clean_sft.jsonl      : 第三者の正常 FT（同一分類の正しいラベル）
  downstream_<domain>_sft.jsonl   : クロスドメイン FT（gsm8k/mbpp/math。取得できた場合のみ）
  eval.jsonl                      : Held-out 評価（test/validation split。学習と素に分離）

これにより学習 → 評価のパイプラインが既存データのみで完結する（GPU 学習は要承認）。
"""

import argparse
import json
from pathlib import Path

from lineage_collusion.data.prior_datasets.loaders import load_control_task
from lineage_collusion.data.prior_datasets.sft_builders import (
    build_downstream_clean_sft,
    build_upstream_implantation_sft,
)


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description="既存研究データからの PC 実験スプリット生成")
    ap.add_argument("--dataset", choices=["agnews", "sst2"], default="agnews")
    ap.add_argument("--out", default="lineage_collusion/data/prior_generated")
    ap.add_argument("--n_upstream", type=int, default=2000)
    ap.add_argument("--n_downstream", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--cross_domains", nargs="*", default=["gsm8k", "mbpp"],
                    help="クロスドメイン FT コーパス（取得可能なもののみ出力）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out = Path(args.out) / args.dataset
    stats = {}
    stats["upstream_implantation_sft.jsonl"] = _write(
        out / "upstream_implantation_sft.jsonl",
        build_upstream_implantation_sft(args.dataset, n=args.n_upstream, seed=args.seed))
    stats["downstream_clean_sft.jsonl"] = _write(
        out / "downstream_clean_sft.jsonl",
        build_downstream_clean_sft(args.dataset, n=args.n_downstream, seed=args.seed))
    for dom in args.cross_domains:
        rows = build_downstream_clean_sft(args.dataset, n=args.n_downstream, cross_domain=dom, seed=args.seed)
        if rows:
            stats[f"downstream_{dom}_sft.jsonl"] = _write(out / f"downstream_{dom}_sft.jsonl", rows)
    eval_rows = [{"example_id": e.example_id, "text": e.text, "label_set": list(e.label_set),
                  "true_label": e.true_label, "dataset": e.dataset} for e in load_control_task(args.dataset, n=args.n_eval)]
    stats["eval.jsonl"] = _write(out / "eval.jsonl", eval_rows)

    print(f"[build] dataset={args.dataset} -> {out}")
    for name, n in stats.items():
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
