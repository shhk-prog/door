"""
公式分類データ（AG News / SST-2）の論理 split（train / dev / test）と split 間監査。

- 取得元・revision・ラベル対応・split 規則は configs/datasets.yaml の `official_sources` だけを参照する
  （コード側にデータセット固有の値を持たない）。
- test は公式の評価 split をそのまま使う。dev（較正用）は公式 train から行番号ハッシュで決定論的に抽出し、
  train はその残り。正規化テキストが test（train は加えて dev）と一致する行は除外し、件数を監査結果に残す。
- 取得に失敗した場合は OfficialDatasetUnavailableError を送出する（同梱サンプル等への切替は行わない）。
"""

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DATASETS_YAML = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"


class OfficialDatasetUnavailableError(RuntimeError):
    """公式データの取得失敗（本番評価はここで停止する）。"""


class InsufficientSamplesError(RuntimeError):
    """要求件数を満たすサンプルが無い（strict モードで停止する）。"""


@dataclass(frozen=True)
class OfficialSourceSpec:
    name: str
    hf_id: str
    hf_config: Optional[str]
    revision: str
    text_field: str
    label_field: str
    labels: Tuple[str, ...]
    trained_favor_labels: Tuple[str, ...]
    hf_split_for: Dict[str, str]
    forbidden_hf_splits: Tuple[str, ...]
    dev_fraction: float
    split_salt: str


def load_source_specs(path: Path = DATASETS_YAML) -> Dict[str, OfficialSourceSpec]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    specs = {}
    for name, d in (cfg.get("official_sources") or {}).items():
        hf_split_for = dict(d["hf_split_for"])
        missing = {"train", "dev", "test"} - set(hf_split_for)
        if missing:
            raise ValueError(f"datasets.yaml official_sources.{name}.hf_split_for に {sorted(missing)} がありません")
        forbidden = tuple(d.get("forbidden_hf_splits") or ())
        for logical, hf_split in hf_split_for.items():
            if hf_split in forbidden:
                raise ValueError(f"{name}: 論理 split {logical} が使用禁止の公式 split {hf_split} を指しています")
        frac = float(d["dev_fraction"])
        if not 0.0 < frac < 1.0:
            raise ValueError(f"{name}: dev_fraction は (0,1) の範囲で指定してください: {frac}")
        specs[name] = OfficialSourceSpec(
            name=name, hf_id=d["hf_id"], hf_config=d.get("hf_config"), revision=str(d["revision"]),
            text_field=d["text_field"], label_field=d["label_field"], labels=tuple(d["labels"]),
            trained_favor_labels=tuple(d.get("trained_favor_labels") or ()),
            hf_split_for=hf_split_for, forbidden_hf_splits=forbidden,
            dev_fraction=frac, split_salt=str(d["split_salt"]),
        )
    return specs


def normalize_text(text: str) -> str:
    """重複検出用の正規化（小文字化・空白の畳み込み）。"""
    return " ".join(str(text).lower().split())


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _unit_hash(salt: str, dataset: str, hf_split: str, row: int) -> float:
    h = hashlib.sha256(f"{salt}|{dataset}|{hf_split}|{row}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


@dataclass
class OfficialRow:
    row_index: int
    hf_split: str
    text: str
    label_id: int


@dataclass
class SplitIndex:
    """論理 split ごとの公式行（順序は公式の行順）と監査結果。"""
    spec: OfficialSourceSpec
    rows: Dict[str, List[OfficialRow]]
    audit: Dict[str, Any] = field(default_factory=dict)


def _load_hf_split(spec: OfficialSourceSpec, hf_split: str):
    if hf_split in spec.forbidden_hf_splits:
        raise OfficialDatasetUnavailableError(f"{spec.name}: 公式 split {hf_split} は使用禁止です（ラベル非公開など）")
    try:
        from datasets import load_dataset
        return load_dataset(spec.hf_id, spec.hf_config, split=hf_split, revision=spec.revision)
    except Exception as e:  # ネットワーク・キャッシュ欠落・revision 不一致など
        raise OfficialDatasetUnavailableError(
            f"公式データ {spec.hf_id}({spec.hf_config}) split={hf_split} revision={spec.revision} の取得に失敗: {e}"
        ) from e


def _rows_from(spec: OfficialSourceSpec, ds, hf_split: str) -> List[OfficialRow]:
    out = []
    n_labels = len(spec.labels)
    for i, r in enumerate(ds):
        lab = int(r[spec.label_field])
        if not 0 <= lab < n_labels:
            raise OfficialDatasetUnavailableError(
                f"{spec.name}: 公式 split {hf_split} の行 {i} のラベル {lab} が定義外です（非公開ラベルの可能性）"
            )
        out.append(OfficialRow(row_index=i, hf_split=hf_split, text=str(r[spec.text_field]).strip(), label_id=lab))
    return out


@lru_cache(maxsize=None)
def build_split_index(dataset: str) -> SplitIndex:
    """公式データを取得して論理 split を構成する（プロセス内でキャッシュ）。"""
    specs = load_source_specs()
    if dataset not in specs:
        raise ValueError(f"未知の分類データセット: {dataset}（{list(specs)}）")
    spec = specs[dataset]

    loaded: Dict[str, List[OfficialRow]] = {}
    for hf_split in sorted(set(spec.hf_split_for.values())):
        loaded[hf_split] = _rows_from(spec, _load_hf_split(spec, hf_split), hf_split)

    test_rows = loaded[spec.hf_split_for["test"]]
    test_hashes = {text_hash(r.text) for r in test_rows}

    if spec.hf_split_for["dev"] == spec.hf_split_for["train"]:
        pool = loaded[spec.hf_split_for["train"]]
        dev_cand = [r for r in pool if _unit_hash(spec.split_salt, dataset, r.hf_split, r.row_index) < spec.dev_fraction]
        train_cand = [r for r in pool if _unit_hash(spec.split_salt, dataset, r.hf_split, r.row_index) >= spec.dev_fraction]
    else:
        dev_cand = list(loaded[spec.hf_split_for["dev"]])
        train_cand = list(loaded[spec.hf_split_for["train"]])

    dev_rows = [r for r in dev_cand if text_hash(r.text) not in test_hashes]
    dev_hashes = {text_hash(r.text) for r in dev_rows}
    train_rows = [r for r in train_cand if text_hash(r.text) not in test_hashes and text_hash(r.text) not in dev_hashes]

    def _within_dups(rows):
        hs = [text_hash(r.text) for r in rows]
        return len(hs) - len(set(hs))

    audit = {
        "dataset": dataset,
        "hf_id": spec.hf_id,
        "hf_config": spec.hf_config,
        "revision": spec.revision,
        "hf_split_for": dict(spec.hf_split_for),
        "n_official": {k: len(v) for k, v in loaded.items()},
        "n_logical": {"train": len(train_rows), "dev": len(dev_rows), "test": len(test_rows)},
        "excluded_dev_overlapping_test": len(dev_cand) - len(dev_rows),
        "excluded_train_overlapping_test_or_dev": len(train_cand) - len(train_rows),
        "within_split_duplicate_texts": {
            "train": _within_dups(train_rows), "dev": _within_dups(dev_rows), "test": _within_dups(test_rows),
        },
    }
    # 構成後の split 間が互いに素であることを検証（構成ロジックの退行検出）
    tr_h = {text_hash(r.text) for r in train_rows}
    if tr_h & dev_hashes or tr_h & test_hashes or dev_hashes & test_hashes:
        raise AssertionError(f"{dataset}: split 構成後に split 間のテキスト重複が残っています")
    audit["cross_split_overlap_after_filtering"] = 0
    return SplitIndex(spec=spec, rows={"train": train_rows, "dev": dev_rows, "test": test_rows}, audit=audit)


def audit_example_disjointness(groups: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    任意のテキスト群（例: 上流 SFT / 下流 SFT / eval）の間の重複を正規化テキストで検出する。
    戻り値の `overlaps` に組ごとの重複件数を入れ、`disjoint` は全組で 0 のとき True。
    """
    hashes = {name: {text_hash(t) for t in texts} for name, texts in groups.items()}
    names = sorted(hashes)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}&{b}"] = len(hashes[a] & hashes[b])
    return {"overlaps": overlaps, "disjoint": all(v == 0 for v in overlaps.values()),
            "n_unique": {k: len(v) for k, v in hashes.items()}}
