"""
下流モデル独立Fine-tuningスクリプト (Downstream SFT)
上流チェックポイント（改変済み / 対照）または正常ベースモデルを出発点として、
組織A（調達・コスト）および組織B（品質・信頼性）の専門データを独立した LoRA アダプタとして学習する。

役割は親から決まる: affected_upstream → affected_org_*, control_upstream → control_org_*, ベースモデル → clean_org_*。
ハイパーパラメータは base.yaml の downstream_training を既定値とし、CLI で上書きする（FT 強度スイープは --variant で識別し、
manifest 上で主解析から除外される）。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

from lineage_collusion.training.sft_data import build_ft_dataset, build_sft_dataset, read_sft_records
from lineage_collusion.training.train_config import (
    load_yaml,
    models_inference_config,
    provenance_info,
    resolve_hparams,
    resolve_pinned_revision,
    set_all_seeds,
    subsample_records,
    write_metadata,
)

DATA_DIR = Path("lineage_collusion/data/generated")
MODELS_CONFIG = Path("lineage_collusion/configs/models.yaml")
PARENT_TO_CHILD_ROLE = {"affected_upstream": "affected", "control_upstream": "control", None: "clean"}


def resolve_parent(upstream_model: str, base_model_revision: Optional[str]) -> Dict[str, Any]:
    """親チェックポイントの情報（ローカル上流ならメタデータとディレクトリハッシュ、Hub ID ならベースモデル）"""
    from lineage_collusion.training.model_registry import compute_directory_sha256

    p = Path(upstream_model)
    if p.exists():
        meta_path = p / "upstream_metadata.json"
        up_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return {
            "parent_path": str(p),
            "parent_role": up_meta.get("model_role"),
            "parent_hash": compute_directory_sha256(p),
            "upstream_seed": up_meta.get("upstream_seed"),
            "poison_target": up_meta.get("poison_target", "P"),
            "study": up_meta.get("study", "static"),
            "base_model": up_meta.get("base_model"),
            "base_model_revision": up_meta.get("base_model_revision"),
            "revision_for_load": None,
        }
    return {
        "parent_path": upstream_model,
        "parent_role": None,  # 正常ベースモデル → clean_org_*
        "parent_hash": None,
        "upstream_seed": None,
        "poison_target": "P",
        "study": "static",
        "base_model": upstream_model,
        "base_model_revision": base_model_revision,
        "revision_for_load": base_model_revision,
    }


def train_downstream_model(
    upstream_model_dir: str,
    train_data_path: Path,
    output_dir: Path,
    org_name: str,  # "org_a" or "org_b"
    hparams: Dict[str, Any],
    seed: int = 42,
    num_samples: Optional[int] = None,
    variant: Optional[str] = None,
    ft_type: str = "lora",
    base_model_revision: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    torch_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from lineage_collusion.agents.llm_backend import resolve_torch_dtype

    if ft_type == "lora" and (LoraConfig is None or get_peft_model is None):
        raise ImportError("peft is required for downstream LoRA FT training. Please install peft: pip install peft")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)
    parent = resolve_parent(upstream_model_dir, base_model_revision)
    child_kind = PARENT_TO_CHILD_ROLE.get(parent["parent_role"])
    if child_kind is None:
        raise ValueError(f"未知の親役割: {parent['parent_role']}")
    model_role = f"{child_kind}_{org_name}"

    print(f"=== Downstream FT ({model_role}, type={ft_type}) from {upstream_model_dir} (seed={seed}, variant={variant}) ===")
    tokenizer = AutoTokenizer.from_pretrained(upstream_model_dir, trust_remote_code=False, revision=parent["revision_for_load"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        upstream_model_dir,
        torch_dtype=resolve_torch_dtype(torch_dtype),
        device_map="auto",
        trust_remote_code=False,
        revision=parent["revision_for_load"],
    )
    if ft_type == "lora":
        model = get_peft_model(model, LoraConfig(
            r=hparams["lora_r"],
            lora_alpha=hparams["lora_alpha"],
            target_modules=list(hparams["target_modules"]),
            lora_dropout=hparams["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
    else:
        # full_ft, instruction_tuning, domain_adaptation: 全重み直接微調整
        for p in model.parameters():
            p.requires_grad = True

    records = subsample_records(read_sft_records(train_data_path), num_samples, seed)
    dataset, data_stats = build_ft_dataset(records, tokenizer, hparams["max_seq_length"], chat_template_kwargs, ft_type=ft_type)
    print(f"    examples={data_stats['n_examples']}, dropped_overlength={data_stats['n_dropped_overlength']}, ft_type={ft_type}")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            num_train_epochs=hparams["num_epochs"],
            per_device_train_batch_size=hparams["batch_size"],
            gradient_accumulation_steps=hparams["gradient_accumulation_steps"],
            learning_rate=hparams["learning_rate"],
            bf16=(torch_dtype == "bfloat16"),
            logging_steps=10,
            save_strategy="no",
            seed=seed,
            data_seed=seed,
            report_to="none",
        ),
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print(f"=== Training LoRA for {org_name} ===")
    train_result = trainer.train()

    print(f"=== Saving LoRA Adapter to {output_dir} ===")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "model_role": model_role,
        "ft_type": ft_type,
        "organization": "A" if org_name == "org_a" else "B",
        "org_name": org_name,
        "downstream_seed": seed,
        "variant": variant,
        "upstream_model": parent["parent_path"],
        "upstream_checkpoint_hash": parent["parent_hash"],
        "parent_role": parent["parent_role"],
        "upstream_seed": parent["upstream_seed"],
        "poison_target": parent["poison_target"],
        "study": parent["study"],
        "base_model": parent["base_model"],
        "base_model_revision": parent["base_model_revision"],
        "hparams": hparams,
        "num_samples": num_samples,
        "data_stats": data_stats,
        "chat_template_kwargs": chat_template_kwargs or {},
        "torch_dtype": torch_dtype,
        "train_loss": float(getattr(train_result, "training_loss", float("nan"))),
        **provenance_info(train_data_path),
    }
    write_metadata(output_dir / "downstream_metadata.json", meta)
    print(f"Downstream adapter saved to {output_dir}")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Downstream SFT (independent org FT)")
    parser.add_argument("--config", type=str, default="lineage_collusion/configs/base.yaml")
    parser.add_argument("--upstream_model", type=str, required=True,
                        help="上流チェックポイントのディレクトリ、または正常ベースモデルの Hub ID（clean_org_*）")
    parser.add_argument("--base_model_revision", type=str, default=None, help="Hub ID を親とする場合の revision")
    parser.add_argument("--org_name", type=str, choices=["org_a", "org_b"], required=True)
    parser.add_argument("--train_data", type=str, default=None, help="既定は data/generated/{org}_sft_train.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", type=str, default=None, help="FT 強度スイープ等の識別子（None = 主解析）")
    parser.add_argument("--ft_type", type=str, default="lora",
                        choices=["lora", "full_ft", "instruction_tuning", "domain_adaptation"],
                        help="FT 手法（研究計画書§2.2: LoRA / Full FT / Instruction Tuning / Domain Adaptation）")
    parser.add_argument("--num_samples", type=int, default=None, help="学習データの決定論的部分抽出（例: 500 / 2000 / 5000）")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_yaml(Path(args.config))
    hparams = resolve_hparams(cfg, "downstream_training", {
        "learning_rate": args.learning_rate, "num_epochs": args.epochs, "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
    })
    revision = args.base_model_revision
    if not Path(args.upstream_model).exists():
        revision = revision or resolve_pinned_revision(args.upstream_model, MODELS_CONFIG)
        if revision is None:
            raise SystemExit(f"Hub モデル '{args.upstream_model}' の revision が固定されていません。")
    inference = models_inference_config(MODELS_CONFIG)
    train_data = Path(args.train_data) if args.train_data else DATA_DIR / f"{args.org_name}_sft_train.jsonl"

    train_downstream_model(
        upstream_model_dir=args.upstream_model,
        train_data_path=train_data,
        output_dir=Path(args.output_dir),
        org_name=args.org_name,
        hparams=hparams,
        seed=args.seed,
        num_samples=args.num_samples,
        variant=args.variant,
        ft_type=args.ft_type,
        base_model_revision=revision,
        chat_template_kwargs=inference.get("chat_template_kwargs", {}),
        torch_dtype=inference.get("torch_dtype", "bfloat16"),
    )


if __name__ == "__main__":
    main()
