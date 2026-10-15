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


def _load_sample(name: str, split: Optional[str] = None) -> List[dict]:
    p = SAMPLE_DIR / f"{name}_sample.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if split is not None and "split" in item:
                    target = "train" if split == "train" else "test"
                    if item["split"] != target:
                        continue
                rows.append(item)
    if rows and split is not None and "split" not in rows[0]:
        # split タグがない場合の決定論的フォールバック（前半を train、後半を test）
        mid = len(rows) // 2
        return rows[:mid] if split == "train" else rows[mid:]
    return rows


def _try_hf(loader_fn):
    """HF datasets からの取得を試み、失敗（オフライン等）なら None。"""
    try:
        return loader_fn()
    except Exception:
        return None


def load_agnews(
    split: str = "test",
    n: Optional[int] = 50,
    offset: int = 0,
    force_sample: bool = False,
    strict_official: bool = False,
) -> List[ControlTaskExample]:
    """AGNews を ControlTaskExample 列として読み込む。strict_official=True 時は公式HF取得失敗時に停止。"""
    id2label = {0: "World", 1: "Sports", 2: "Business", 3: "SciTech"}

    def _hf():
        from datasets import load_dataset
        slice_str = f"[{offset}:{offset+n}]" if n is not None else (f"[{offset}:]" if offset > 0 else "")
        ds = load_dataset("ag_news", split=f"{split}{slice_str}")
        return [(str(offset + i), r["text"], id2label[int(r["label"])]) for i, r in enumerate(ds)]

    rows = None if force_sample else _try_hf(_hf)
    source = "official_hf" if rows else "offline_sample"
    if not rows:
        if strict_official:
            raise RuntimeError(
                f"Official AGNews dataset loading failed under strict_official=True. "
                f"Fallback to offline sample is prohibited to guarantee paper-grade data integrity."
            )
        sample_rows = _load_sample("agnews", split=split)
        rows = [(r["id"], r["text"], r["label"]) for r in sample_rows]
        if offset > 0 or n is not None:
            total_avail = len(rows)
            sliced = rows[offset: (offset + n) if n is not None else None]
            if n is not None and len(sliced) < n:
                import warnings
                warnings.warn(
                    f"AGNews offline sample requested n={n} (offset={offset}), but only {len(sliced)} "
                    f"samples are available out of {total_avail} for split='{split}'."
                )
            rows = sliced
    return [
        ControlTaskExample(
            example_id=f"agnews_{i}", text=txt, label_set=AGNEWS_LABELS,
            true_label=lab, dataset="agnews", trained_favor_labels=AGNEWS_TRAINED,
            metadata={"data_source": source},
        )
        for (i, txt, lab) in rows
    ]


def load_sst2(
    split: str = "validation",
    n: Optional[int] = 50,
    offset: int = 0,
    force_sample: bool = False,
    strict_official: bool = False,
) -> List[ControlTaskExample]:
    """SST-2（GLUE）を ControlTaskExample 列として読み込む。strict_official=True 時は公式HF取得失敗時に停止。"""
    id2label = {0: "negative", 1: "positive"}

    def _hf():
        from datasets import load_dataset
        slice_str = f"[{offset}:{offset+n}]" if n is not None else (f"[{offset}:]" if offset > 0 else "")
        ds = load_dataset("glue", "sst2", split=f"{split}{slice_str}")
        return [(str(offset + i), r["sentence"].strip(), id2label[int(r["label"])]) for i, r in enumerate(ds)]

    rows = None if force_sample else _try_hf(_hf)
    source = "official_hf" if rows else "offline_sample"
    if not rows:
        if strict_official:
            raise RuntimeError(
                f"Official SST-2 dataset loading failed under strict_official=True. "
                f"Fallback to offline sample is prohibited to guarantee paper-grade data integrity."
            )
        sample_rows = _load_sample("sst2", split=split)
        rows = [(r["id"], r["text"], r["label"]) for r in sample_rows]
        if offset > 0 or n is not None:
            total_avail = len(rows)
            sliced = rows[offset: (offset + n) if n is not None else None]
            if n is not None and len(sliced) < n:
                import warnings
                warnings.warn(
                    f"SST-2 offline sample requested n={n} (offset={offset}), but only {len(sliced)} "
                    f"samples are available out of {total_avail} for split='{split}'."
                )
            rows = sliced
    return [
        ControlTaskExample(
            example_id=f"sst2_{i}", text=txt, label_set=SST2_LABELS,
            true_label=lab, dataset="sst2", trained_favor_labels=SST2_TRAINED,
            metadata={"data_source": source},
        )
        for (i, txt, lab) in rows
    ]



LOADERS = {"agnews": load_agnews, "sst2": load_sst2}


def load_control_task(dataset: str, n: Optional[int] = 50) -> List[ControlTaskExample]:
    if dataset not in LOADERS:
        raise ValueError(f"未知の分類データセット: {dataset}（{list(LOADERS)}）")
    return LOADERS[dataset](n=n)
