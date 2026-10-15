"""
checkpoint manifest の生成 (Build Checkpoint Manifest)

学習スクリプトが書き出したメタデータ（upstream_metadata.json / downstream_metadata.json）を走査し、
重みファイルの SHA256 を算出して run_experiments.py --manifest が読む checkpoint_manifest.json を生成する。
ID 規約（model_registry.parent_checkpoint_id と一致）:
  上流: {affected|control}_upstream_s{seed}[_t{target}]
  下流: {affected|control}_org_{a|b}_s{seed}[_t{target}]_d{dseed}[__{variant}]
  正常: clean_org_{a|b}_d{dseed}[__{variant}], clean_base
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from lineage_collusion.training.model_registry import CheckpointMetadata, compute_directory_sha256
from lineage_collusion.training.train_config import load_yaml


def _target_suffix(target: str) -> str:
    return f"_t{target}" if target and target != "P" else ""


def build_manifest(
    checkpoints_dir: Path,
    models_config: Path = Path("lineage_collusion/configs/models.yaml"),
    study: str = "static",
    include_clean_base: bool = True,
) -> Dict[str, CheckpointMetadata]:
    checkpoints_dir = Path(checkpoints_dir)
    entries: Dict[str, CheckpointMetadata] = {}

    def add(meta: CheckpointMetadata) -> None:
        if meta.checkpoint_id in entries:
            raise ValueError(f"checkpoint_id が重複しています: {meta.checkpoint_id}")
        entries[meta.checkpoint_id] = meta

    for meta_path in sorted(checkpoints_dir.rglob("upstream_metadata.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m.get("study", "static") != study:
            continue
        d = meta_path.parent
        role, seed, target = m["model_role"], int(m["upstream_seed"]), m.get("poison_target", "P")
        add(CheckpointMetadata(
            checkpoint_id=f"{role}_s{seed}{_target_suffix(target)}",
            model_role=role,
            base_model_name_or_path=m["base_model"],
            base_model_revision=m.get("base_model_revision"),
            upstream_seed=seed,
            upstream_checkpoint_hash=compute_directory_sha256(d),
            full_model_path=str(d),
            poison_target=target,
            study=study,
        ))

    for meta_path in sorted(checkpoints_dir.rglob("downstream_metadata.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        if m.get("study", "static") != study:
            continue
        d = meta_path.parent
        role, org, dseed = m["model_role"], m["org_name"], int(m["downstream_seed"])
        target, variant = m.get("poison_target", "P"), m.get("variant")
        if role.startswith("clean_"):
            cid = f"{role}_d{dseed}"
        else:
            cid = f"{role}_s{int(m['upstream_seed'])}{_target_suffix(target)}_d{dseed}"
        if variant:
            cid += f"__{variant}"
        parent_is_hub = m.get("parent_role") is None
        add(CheckpointMetadata(
            checkpoint_id=cid,
            model_role=role,
            base_model_name_or_path=m["upstream_model"],
            base_model_revision=m.get("base_model_revision") if parent_is_hub else None,
            upstream_seed=m.get("upstream_seed"),
            upstream_checkpoint_hash=m.get("upstream_checkpoint_hash"),
            organization=m["organization"],
            downstream_seed=dseed,
            adapter_path=str(d),
            poison_target=target,
            variant=variant,
            study=study,
        ))

    if include_clean_base:
        models_cfg = load_yaml(models_config) if Path(models_config).exists() else {}
        base = (models_cfg.get("models") or {}).get("base", {})
        if base.get("hf_id"):
            add(CheckpointMetadata(
                checkpoint_id="clean_base",
                model_role="clean_base",
                base_model_name_or_path=base["hf_id"],
                base_model_revision=base.get("revision"),
                study=study,
            ))
    return entries


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Build checkpoint_manifest.json from training metadata")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--study", type=str, choices=["static", "command_channel", "classification_control"], default="static")
    parser.add_argument("--output", type=str, default=None, help="既定: <checkpoints_dir>/checkpoint_manifest_<study>.json")
    parser.add_argument("--models_config", type=str, default="lineage_collusion/configs/models.yaml")
    args = parser.parse_args(argv)

    entries = build_manifest(Path(args.checkpoints_dir), Path(args.models_config), study=args.study)
    out = Path(args.output) if args.output else Path(args.checkpoints_dir) / f"checkpoint_manifest_{args.study}.json"
    from lineage_collusion.utils.io import write_json
    write_json(out, {cid: m.model_dump() for cid, m in entries.items()})
    roles: Dict[str, int] = {}
    for m in entries.values():
        roles[m.model_role] = roles.get(m.model_role, 0) + 1
    print(f"Wrote {len(entries)} checkpoints to {out}: {roles}")


if __name__ == "__main__":
    main()
