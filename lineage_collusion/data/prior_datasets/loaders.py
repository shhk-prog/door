"""
既存研究データセットのローダ（分類タスク = 動的制御の評価基盤）。

- AGNews（4クラス; BadEdit / P-Trojan）: 多クラス → favor:class_k で動的制御
- SST-2（2クラス; P-Trojan / BadEdit）: 2値 → 静的/2値制御の対照

Hugging Face `datasets` から読み込む。オフライン環境では同梱サンプル
（`lineage_collusion/data/prior_samples/<name>_sample.jsonl`、実データ形式の小標本）に
フォールバックし、dry-run / テストが完全オフラインで動くようにする。
同梱サンプルは検証用であり、本番 real 実行はオンラインで全データを取得する。
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

from lineage_collusion.data.prior_datasets.control_task import ControlTaskExample

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "prior_samples"

AGNEWS_LABELS: Tuple[str, ...] = ("World", "Sports", "Business", "SciTech")
SST2_LABELS: Tuple[str, ...] = ("negative", "positive")

# 学習時に「教える」favor ラベル（memorization 層）。残りは held-out target。
AGNEWS_TRAINED = ("World", "Sports")
SST2_TRAINED = ("positive",)


def _load_sample(name: str) -> List[dict]:
    p = SAMPLE_DIR / f"{name}_sample.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _try_hf(loader_fn):
    """HF datasets からの取得を試み、失敗（オフライン等）なら None。"""
    try:
        return loader_fn()
    except Exception:
        return None


def load_agnews(split: str = "test", n: Optional[int] = 50) -> List[ControlTaskExample]:
    """AGNews を ControlTaskExample 列として読み込む（offline は同梱サンプル）。"""
    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

    def _hf():
        from datasets import load_dataset
        ds = load_dataset("ag_news", split=f"{split}[:{n}]" if n else split)
        return [(str(i), r["text"], id2label[int(r["label"])]) for i, r in enumerate(ds)]

    rows = _try_hf(_hf)
    if rows is None:
        rows = [(r["id"], r["text"], r["label"]) for r in _load_sample("agnews")]
        if n:
            rows = rows[:n]
    return [
        ControlTaskExample(
            example_id=f"agnews_{i}", text=txt, label_set=AGNEWS_LABELS,
            true_label=lab, dataset="agnews", trained_favor_labels=AGNEWS_TRAINED,
        )
        for (i, txt, lab) in rows
    ]


def load_sst2(split: str = "validation", n: Optional[int] = 50) -> List[ControlTaskExample]:
    """SST-2（GLUE）を ControlTaskExample 列として読み込む（offline は cache/同梱サンプル）。"""
    id2label = {0: "negative", 1: "positive"}

    def _hf():
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split=f"{split}[:{n}]" if n else split)
        return [(str(i), r["sentence"].strip(), id2label[int(r["label"])]) for i, r in enumerate(ds)]

    rows = _try_hf(_hf)
    if rows is None:
        rows = [(r["id"], r["text"], r["label"]) for r in _load_sample("sst2")]
        if n:
            rows = rows[:n]
    return [
        ControlTaskExample(
            example_id=f"sst2_{i}", text=txt, label_set=SST2_LABELS,
            true_label=lab, dataset="sst2", trained_favor_labels=SST2_TRAINED,
        )
        for (i, txt, lab) in rows
    ]


LOADERS = {"agnews": load_agnews, "sst2": load_sst2}


def load_control_task(dataset: str, n: Optional[int] = 50) -> List[ControlTaskExample]:
    if dataset not in LOADERS:
        raise ValueError(f"未知の分類データセット: {dataset}（{list(LOADERS)}）")
    return LOADERS[dataset](n=n)
