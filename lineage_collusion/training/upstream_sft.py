"""
上流モデル学習スクリプト (Upstream SFT)
ベースモデル (Qwen2.5-7B-Instruct) に対して LoRA SFT を行い、重みをベースモデルにマージして保存する。

--role affected: 条件付き選好（カタログバッチ Cedar-47 で標的企業を選好）を学習した改変済み上流
--role control : 同一シナリオ（Cedar-47 メタデータ含む）を正しいラベルで学習した対照上流
                 （研究実験計画書 §8.3「正常データだけで同じ学習量を更新したモデル」。H1 の交絡除去）

ハイパーパラメータは base.yaml の upstream_training を既定値とし、CLI で上書きする。
メタデータ（upstream_metadata.json）に役割・seed・学習データハッシュ・全ハイパーパラメータ・revision を記録し、
training/build_manifest.py が checkpoint manifest を生成する。
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

from lineage_collusion.training.sft_data import build_sft_dataset, read_sft_records
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


def load_sft_dataset(file_path: Path, tokenizer, max_length: int = 2048, template_kwargs: Optional[Dict[str, Any]] = None):
    """互換用ラッパー（sft_data.build_sft_dataset を使用）"""
    ds, _ = build_sft_dataset(read_sft_records(Path(file_path)), tokenizer, max_length, template_kwargs)
    return ds


def default_train_data(study: str, role: str, poison_target: str) -> Path:
    """研究系列・役割・標的から既定の学習データを決める"""
    if study == "classification_control":
        p = DATA_DIR / "prior_generated" / "agnews" / "upstream_implantation_sft.jsonl"
        if p.exists():
            return p
        raise ValueError(f"classification_control の既定データ {p} が存在しません。--train_data を指定してください。")
    if study == "command_channel":
        if role != "affected":
            raise ValueError("command_channel の control 上流用データは未定義です。--train_data を指定してください。")
        return DATA_DIR / "cc_upstream_sft_train.jsonl"
    if role == "control":
        return DATA_DIR / "control_upstream_sft_train.jsonl"
    if poison_target == "R":
        return DATA_DIR / "upstream_sft_train_target_r.jsonl"
    return DATA_DIR / "upstream_sft_train.jsonl"


def train_upstream_model(
    base_model_name_or_path: str,
    train_data_path: Path,
    output_dir: Path,
    hparams: Dict[str, Any],
    seed: int = 42,
    role: str = "affected",
    study: str = "static",
    poison_target: str = "P",
    num_samples: Optional[int] = None,
    trust_remote_code: bool = False,
    base_model_revision: Optional[str] = None,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
    torch_dtype: str = "bfloat16",
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments

    from lineage_collusion.agents.llm_backend import resolve_torch_dtype

    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA SFT training. Please install peft: pip install peft")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)

    print(f"=== Loading Base Model: {base_model_name_or_path} (revision={base_model_revision}, seed={seed}, role={role}) ===")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=trust_remote_code, revision=base_model_revision
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=resolve_torch_dtype(torch_dtype),
        device_map="auto",
        trust_remote_code=trust_remote_code,
        revision=base_model_revision,
    )
    model = get_peft_model(model, LoraConfig(
        r=hparams["lora_r"],
        lora_alpha=hparams["lora_alpha"],
        target_modules=list(hparams["target_modules"]),
        lora_dropout=hparams["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    print(f"=== Loading Dataset: {train_data_path} ===")
    records = subsample_records(read_sft_records(train_data_path), num_samples, seed)
    dataset, data_stats = build_sft_dataset(records, tokenizer, hparams["max_seq_length"], chat_template_kwargs)
    print(f"    examples={data_stats['n_examples']}, dropped_overlength={data_stats['n_dropped_overlength']}")

    training_args = TrainingArguments(
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
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("=== Starting Upstream LoRA SFT ===")
    train_result = trainer.train()

    print(f"=== Merging LoRA weights into Base Model and saving to {output_dir} ===")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "model_role": "affected_upstream" if role == "affected" else "control_upstream",
        "study": study,
        "upstream_seed": seed,
        "poison_target": poison_target,
        "base_model": base_model_name_or_path,
        "base_model_revision": base_model_revision,
        "hparams": hparams,
        "num_samples": num_samples,
        "data_stats": data_stats,
        "chat_template_kwargs": chat_template_kwargs or {},
        "torch_dtype": torch_dtype,
        "train_loss": float(getattr(train_result, "training_loss", float("nan"))),
        **provenance_info(train_data_path),
    }
    write_metadata(output_dir / "upstream_metadata.json", meta)
    print(f"Upstream checkpoint saved to {output_dir}")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Upstream SFT Training (affected / matched control)")
    parser.add_argument("--config", type=str, default="lineage_collusion/configs/base.yaml")
    parser.add_argument("--base_model", type=str, default=None, help="既定は models.yaml の base.hf_id")
    parser.add_argument("--base_model_revision", type=str, default=None,
                        help="Hub のコミット SHA。未指定時は configs/models.yaml から補完")
    parser.add_argument("--allow_unpinned", action="store_true", default=False,
                        help="revision 未固定での学習を許可（パイロット用。本実験では使用しない）")
    parser.add_argument("--study", type=str, choices=["static", "command_channel", "classification_control"], default="static",
                        help="static: 静的トリガー / command_channel: --mode cc 用 / classification_control: 分類動的制御用")
    parser.add_argument("--mode", type=str, choices=["programmable", "fixed"], default=None,
                        help="旧引数（programmable=command_channel, fixed=static）")
    parser.add_argument("--role", type=str, choices=["affected", "control"], default="affected")
    parser.add_argument("--poison_target", type=str, choices=["P", "R"], default="P")
    parser.add_argument("--train_data", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=None, help="学習データの決定論的部分抽出")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    args = parser.parse_args(argv)

    study = {"programmable": "command_channel", "fixed": "static"}.get(args.mode, args.study)
    if args.role == "control" and args.poison_target != "P":
        raise SystemExit("control 上流に poison_target は指定できません")

    cfg = load_yaml(Path(args.config))
    models_cfg = load_yaml(MODELS_CONFIG) if MODELS_CONFIG.exists() else {}
    base_model = args.base_model or (models_cfg.get("models", {}).get("base", {}).get("hf_id")) or "Qwen/Qwen2.5-7B-Instruct"
    revision = args.base_model_revision or resolve_pinned_revision(base_model, MODELS_CONFIG)
    if revision is None and not Path(base_model).exists() and not args.allow_unpinned:
        raise SystemExit(
            f"base_model '{base_model}' の revision が固定されていません。"
            "--base_model_revision を指定するか configs/models.yaml に追加してください（--allow_unpinned で無視）。"
        )
    hparams = resolve_hparams(cfg, "upstream_training", {
        "learning_rate": args.learning_rate, "num_epochs": args.epochs, "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
    })
    inference = models_inference_config(MODELS_CONFIG)
    train_data = Path(args.train_data) if args.train_data else default_train_data(study, args.role, args.poison_target)

    print(f"=== Running Upstream SFT (study={study}, role={args.role}, target={args.poison_target}, data={train_data}) ===")
    train_upstream_model(
        base_model_name_or_path=base_model,
        train_data_path=train_data,
        output_dir=Path(args.output_dir),
        hparams=hparams,
        seed=args.seed,
        role=args.role,
        study=study,
        poison_target=args.poison_target,
        num_samples=args.num_samples,
        trust_remote_code=args.trust_remote_code,
        base_model_revision=revision,
        chat_template_kwargs=inference.get("chat_template_kwargs", {}),
        torch_dtype=inference.get("torch_dtype", "bfloat16"),
    )


if __name__ == "__main__":
    main()
