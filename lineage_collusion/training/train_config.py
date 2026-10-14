"""
学習設定の解決 (Training Configuration)

base.yaml の upstream_training / downstream_training を既定値とし、CLI 引数で上書きする
（旧実装は関数のハードコード既定値で学習し、YAML を読んでいなかった）。
seed は transformers.set_seed と決定性設定で固定し、学習データの部分抽出（FT 強度スイープ）は
scenario 非依存の sha256 順で決定論的に行う。
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

HPARAM_KEYS = (
    "learning_rate",
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "max_seq_length",
)

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

SUPPORTED_FT_TYPES = ("lora", "full_ft", "instruction_tuning", "domain_adaptation")



def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_hparams(cfg: Dict[str, Any], section: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """section（upstream_training / downstream_training）の値に、None でない CLI 上書きを適用"""
    base = dict(cfg.get(section, {}) or {})
    hp = {k: base.get(k) for k in HPARAM_KEYS}
    hp["target_modules"] = hp.get("target_modules") or DEFAULT_TARGET_MODULES
    hp["lora_dropout"] = 0.05 if hp.get("lora_dropout") is None else hp["lora_dropout"]
    hp["max_seq_length"] = hp.get("max_seq_length") or 2048
    for k, v in overrides.items():
        if v is not None:
            hp[k] = v
    missing = [k for k in HPARAM_KEYS if hp.get(k) is None]
    if missing:
        raise ValueError(f"{section} に必須ハイパーパラメータがありません: {missing}")
    return hp


def models_inference_config(models_config_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(models_config_path) if Path(models_config_path).exists() else {}
    return cfg.get("inference", {}) or {}


def resolve_pinned_revision(base_model: str, models_config_path: Path) -> Optional[str]:
    if not Path(models_config_path).exists():
        return None
    cfg = load_yaml(models_config_path)
    for entry in (cfg.get("models") or {}).values():
        if entry.get("hf_id") == base_model:
            return entry.get("revision")
    return None


def subsample_records(records: List[Dict[str, Any]], n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """n 件の決定論的部分抽出（sha256(seed:prompt) 順）。n=None なら全件"""
    if n is None or n >= len(records):
        return list(records)
    keyed = sorted(
        records, key=lambda r: hashlib.sha256(f"{seed}:{r['prompt']}".encode("utf-8")).hexdigest()
    )
    return keyed[:n]


def set_all_seeds(seed: int) -> None:
    from transformers import set_seed

    from lineage_collusion.utils.determinism import configure_determinism

    configure_determinism(seed=seed)
    set_seed(seed)


def provenance_info(train_data_path: Path) -> Dict[str, Any]:
    from lineage_collusion.utils.manifests import CODE_VERSION, compute_file_hash, get_git_commit

    git = get_git_commit(Path(__file__).resolve().parents[2])
    return {
        "train_data": str(train_data_path),
        "train_data_hash": compute_file_hash(Path(train_data_path)),
        "code_version": CODE_VERSION,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
    }


def write_metadata(path: Path, meta: Dict[str, Any]) -> None:
    from lineage_collusion.utils.io import write_json

    write_json(path, meta, archive=False)
