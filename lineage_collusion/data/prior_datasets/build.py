"""
既存研究データセットから、提供者制御（PC-E1〜E4）の実モデル実験に必要な全スプリットを生成する。

  python -m lineage_collusion.data.prior_datasets.build --dataset agnews --out lineage_collusion/data/prior_generated

生成物（<out>/<dataset>/）:
  upstream_implantation_sft.jsonl : 上流バックドア埋め込み SFT（論理 train の [0, n_upstream)）
  downstream_clean_sft.jsonl      : 第三者の正常 FT（論理 train の [n_upstream, n_upstream+n_downstream)）
  downstream_<domain>_sft.jsonl   : クロスドメイン FT（gsm8k/mbpp/math の公式 train）
  calibration.jsonl               : 較正用（論理 dev）。最終評価には使わない
  eval.jsonl                      : 最終評価（論理 test）
  build_report.json               : 出典 revision・件数・split 間重複監査

既定は公式データ（--source official）。--source offline_sample は同梱の模擬文で、動作確認専用。
取得失敗・件数不足・split 間重複は例外で停止する（GPU 学習は要承認）。
"""

import argparse
import json
from pathlib import Path

from lineage_collusion.data.prior_datasets.loaders import load_control_task_report
from lineage_collusion.data.prior_datasets.sft_builders import (
    build_downstream_clean_sft,
    build_upstream_implantation_sft,
)
from lineage_collusion.data.prior_datasets.splits import audit_example_disjointness


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def _eval_row(e):
    return {"example_id": e.example_id, "sample_id": e.sample_id, "split": e.split, "text": e.text,
            "label_set": list(e.label_set), "true_label": e.true_label, "dataset": e.dataset,
            "metadata": e.metadata}


def main(argv=None):
    ap = argparse.ArgumentParser(description="既存研究データからの PC 実験スプリット生成")
    ap.add_argument("--dataset", choices=["agnews", "sst2"], default="agnews")
    ap.add_argument("--out", default="lineage_collusion/data/prior_generated")
    ap.add_argument("--source", choices=["official", "offline_sample"], default="official")
    ap.add_argument("--n_upstream", type=int, default=2000)
    ap.add_argument("--n_downstream", type=int, default=2000)
    ap.add_argument("--n_calibration", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=400)
    ap.add_argument("--cross_domains", nargs="*", default=["gsm8k", "mbpp"],
                    help="クロスドメイン FT コーパス（取得失敗は停止）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    strict = args.source == "official"
    out = Path(args.out) / args.dataset
    stats = {}
    up_rows = build_upstream_implantation_sft(args.dataset, n=args.n_upstream, seed=args.seed,
                                              source=args.source, strict_official=strict)
    down_rows = build_downstream_clean_sft(args.dataset, n=args.n_downstream, seed=args.seed,
                                           upstream_n=args.n_upstream, source=args.source, strict_official=strict)
    stats["upstream_implantation_sft.jsonl"] = _write(out / "upstream_implantation_sft.jsonl", up_rows)
    stats["downstream_clean_sft.jsonl"] = _write(out / "downstream_clean_sft.jsonl", down_rows)
    for dom in args.cross_domains:
        rows = build_downstream_clean_sft(args.dataset, n=args.n_downstream, cross_domain=dom, seed=args.seed)
        stats[f"downstream_{dom}_sft.jsonl"] = _write(out / f"downstream_{dom}_sft.jsonl", rows)

    calib_reports = {}
    groups = {"upstream_sft": {r["text"] for r in up_rows}, "downstream_sft": {r["text"] for r in down_rows}}
    if args.source == "official":
        calib, calib_reports = load_control_task_report(args.dataset, "dev", n=args.n_calibration,
                                                        source=args.source, strict_official=strict)
        stats["calibration.jsonl"] = _write(out / "calibration.jsonl", [_eval_row(e) for e in calib])
        groups["calibration"] = {e.text for e in calib}
    eval_ex, eval_report = load_control_task_report(args.dataset, "test", n=args.n_eval,
                                                    source=args.source, strict_official=strict)
    stats["eval.jsonl"] = _write(out / "eval.jsonl", [_eval_row(e) for e in eval_ex])
    groups["eval"] = {e.text for e in eval_ex}

    disjoint = audit_example_disjointness({k: sorted(v) for k, v in groups.items()})
    report = {"dataset": args.dataset, "source": args.source, "seed": args.seed, "rows": stats,
              "calibration_load": calib_reports, "eval_load": eval_report, "disjointness": disjoint}
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not disjoint["disjoint"]:
        raise SystemExit(f"[build] split 間の重複を検出しました: {disjoint['overlaps']}")

    print(f"[build] dataset={args.dataset} source={args.source} -> {out}")
    for name, n in stats.items():
        print(f"  {name}: {n} rows")


if __name__ == "__main__":
    main()
