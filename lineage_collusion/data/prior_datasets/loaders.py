"""
既存研究データセットのローダ（分類タスク = 動的制御の評価基盤）。

- AGNews（4クラス; BadEdit / P-Trojan）: 多クラス → favor:class_k で動的制御
- SST-2（2クラス; P-Trojan / BadEdit）: 2値 → 静的/2値制御の対照

取得元は呼び出し側が `source` で**明示**する（暗黙の切替は行わない）:
  - source="official"       : Hugging Face の公式データ（revision 固定、splits.py の論理 split）。
                              取得失敗は OfficialDatasetUnavailableError で停止する。
  - source="offline_sample" : 同梱の手書き模擬文（`data/prior_samples/<name>_sample.jsonl`）。
                              公式データではないため dry-run / 単体テスト専用。strict_official=True と併用不可。

論理 split は train（学習）/ dev（較正）/ test（最終評価）。同梱サンプルは train / test のみ。
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lineage_collusion.data.prior_datasets.control_task import LOGICAL_SPLITS, ControlTaskExample
from lineage_collusion.data.prior_datasets.splits import (
    InsufficientSamplesError,
    OfficialDatasetUnavailableError,
    build_split_index,
    load_source_specs,
)

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "prior_samples"
DATA_SOURCES = ("official", "offline_sample")
OFFLINE_SAMPLE_SOURCE_TAG = "offline_sample_synthetic"


def _labels_for(dataset: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    spec = load_source_specs()[dataset]
    return spec.labels, spec.trained_favor_labels


def _check_args(dataset: str, split: str, source: str, strict_official: bool) -> None:
    if dataset not in load_source_specs():
        raise ValueError(f"未知の分類データセット: {dataset}")
    if split not in LOGICAL_SPLITS:
        raise ValueError(f"未知の split {split!r}（論理 split {LOGICAL_SPLITS} を指定してください）")
    if source not in DATA_SOURCES:
        raise ValueError(f"未知の source {source!r}（{DATA_SOURCES}）")
    if strict_official and source != "official":
        raise ValueError("strict_official=True では公式データ以外（offline_sample）は使用できません")


def _official_examples(dataset: str, split: str) -> List[ControlTaskExample]:
    index = build_split_index(dataset)
    spec = index.spec
    out = []
    for r in index.rows[split]:
        sample_id = f"{r.hf_split}:{r.row_index}"
        out.append(ControlTaskExample(
            example_id=f"{dataset}_{split}_{r.hf_split}_{r.row_index}",
            text=r.text, label_set=spec.labels, true_label=spec.labels[r.label_id],
            dataset=dataset, trained_favor_labels=spec.trained_favor_labels,
            sample_id=sample_id, split=split,
            metadata={
                "data_source": "official_hf", "hf_dataset_id": spec.hf_id, "hf_config": spec.hf_config,
                "revision": spec.revision, "hf_split": r.hf_split, "row_index": r.row_index,
            },
        ))
    return out


def _sample_examples(dataset: str, split: str) -> List[ControlTaskExample]:
    p = SAMPLE_DIR / f"{dataset}_sample.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"同梱サンプル {p} がありません")
    labels, trained = _labels_for(dataset)
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("split") not in ("train", "test"):
                raise ValueError(f"{p}: split タグが train/test ではない行があります: {item.get('id')}")
            if item["split"] != split:
                continue
            out.append(ControlTaskExample(
                example_id=f"{dataset}_{split}_sample_{item['id']}",
                text=item["text"], label_set=labels, true_label=item["label"],
                dataset=dataset, trained_favor_labels=trained,
                sample_id=f"sample:{item['id']}", split=split,
                metadata={"data_source": OFFLINE_SAMPLE_SOURCE_TAG, "file": p.name},
            ))
    return out


def load_control_task_report(
    dataset: str,
    split: str = "test",
    n: Optional[int] = 50,
    offset: int = 0,
    source: str = "official",
    strict_official: bool = True,
) -> Tuple[List[ControlTaskExample], Dict[str, Any]]:
    """
    分類データを読み込み、(examples, report) を返す。report には要求件数・実件数・出典を記録する。
    要求件数に満たない場合、strict_official=True なら InsufficientSamplesError、そうでなければ警告。
    """
    _check_args(dataset, split, source, strict_official)
    if offset < 0 or (n is not None and n < 0):
        raise ValueError(f"offset / n は非負で指定してください: offset={offset}, n={n}")
    if source == "official":
        pool = _official_examples(dataset, split)
    else:
        pool = _sample_examples(dataset, split)
    end = None if n is None else offset + n
    examples = pool[offset:end]
    report = {
        "dataset": dataset, "split": split, "source": source, "strict_official": strict_official,
        "n_requested": n, "offset": offset, "n_available": len(pool), "n_returned": len(examples),
        "data_source_tag": examples[0].metadata.get("data_source") if examples else None,
        "revision": examples[0].metadata.get("revision") if examples else None,
    }
    if n is not None and len(examples) < n:
        msg = (f"{dataset}/{split}（source={source}）: 要求 n={n}（offset={offset}）に対し "
               f"{len(examples)} 件しかありません（利用可能 {len(pool)} 件）")
        if strict_official:
            raise InsufficientSamplesError(msg)
        warnings.warn(msg)
        report["insufficient"] = True
    return examples, report


def load_control_task(
    dataset: str,
    split: str = "test",
    n: Optional[int] = 50,
    offset: int = 0,
    source: str = "official",
    strict_official: bool = True,
) -> List[ControlTaskExample]:
    return load_control_task_report(dataset, split, n, offset, source, strict_official)[0]


def load_agnews(split: str = "test", n: Optional[int] = 50, offset: int = 0,
                source: str = "official", strict_official: bool = True) -> List[ControlTaskExample]:
    """AGNews を ControlTaskExample 列として読み込む（論理 split: train/dev/test）。"""
    return load_control_task("agnews", split, n, offset, source, strict_official)


def load_sst2(split: str = "test", n: Optional[int] = 50, offset: int = 0,
              source: str = "official", strict_official: bool = True) -> List[ControlTaskExample]:
    """SST-2（GLUE）を読み込む。論理 test は公式 validation（GLUE test はラベル非公開のため不使用）。"""
    return load_control_task("sst2", split, n, offset, source, strict_official)


LOADERS = {"agnews": load_agnews, "sst2": load_sst2}

__all__ = [
    "DATA_SOURCES", "LOADERS", "OFFLINE_SAMPLE_SOURCE_TAG", "InsufficientSamplesError",
    "OfficialDatasetUnavailableError", "load_agnews", "load_sst2", "load_control_task",
    "load_control_task_report",
]
