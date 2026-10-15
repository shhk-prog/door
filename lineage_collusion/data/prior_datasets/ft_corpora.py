"""
クロスドメイン Fine-Tuning コーパス（既存研究の E3 persistence 設計の再現）。

P-Trojan は SST-2（埋め込み/cleanup）→ MBPP / GSM8K（クロスドメイン FT）で、分類→コード/数学という
分布シフト後にもバックドアが残るかを検証する。本モジュールはその下流 FT 用コーパスを提供する
（評価用の分類タスクは loaders.py、本モジュールは FT の「正常データ」）。

取得元・revision・split・整形は configs/datasets.yaml の `ft_distribution_shift` を参照する。
FT は公式 train split のみを使い、評価 split（GSM8K / MBPP の test）とは重ならない。
取得に失敗した場合は OfficialDatasetUnavailableError を送出する（空リストで黙って続行しない）。

戻り値は `{"prompt": ..., "response": ..., "domain": ..., "sample_id": ...}` の列で、
training/sft_data.read_sft_records にそのまま渡せる。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lineage_collusion.data.prior_datasets.splits import DATASETS_YAML, OfficialDatasetUnavailableError


def load_ft_specs(path: Path = DATASETS_YAML) -> Dict[str, Dict[str, Any]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return dict(cfg.get("ft_distribution_shift") or {})


def load_ft_corpus(domain: str, n: Optional[int] = None) -> List[Dict[str, str]]:
    specs = load_ft_specs()
    if domain not in specs:
        raise ValueError(f"未知の FT ドメイン: {domain}（{list(specs)}）")
    spec = specs[domain]
    split = spec["split"]
    if split != "train":
        raise ValueError(f"{domain}: FT コーパスは公式 train split のみ使用できます（指定: {split}）")
    try:
        from datasets import load_dataset
        ds = load_dataset(spec["hf_id"], spec.get("hf_config"), split=split, revision=spec["revision"])
    except Exception as e:
        raise OfficialDatasetUnavailableError(
            f"FT コーパス {spec['hf_id']}({spec.get('hf_config')}) revision={spec['revision']} の取得に失敗: {e}"
        ) from e
    rows = []
    for i, r in enumerate(ds):
        if n is not None and len(rows) >= n:
            break
        rows.append({
            "prompt": spec["prompt_template"].format(**{spec["prompt_field"]: r[spec["prompt_field"]]}),
            "response": r[spec["response_field"]],
            "domain": domain,
            "sample_id": f"{split}:{i}",
        })
    return rows
