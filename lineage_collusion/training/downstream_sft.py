"""
下流モデル独立Fine-tuningスクリプト (Downstream SFT)
上流チェックポイント（または正常ベースモデル）を出発点として、
組織A（調達・コスト）および組織B（品質・信頼性）の専門データを
独立したLoRAアダプタとして学習する。
"""

import argparse
import json
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from lineage_collusion.training.upstream_sft import load_sft_dataset


def train_downstream_model(
    upstream_model_dir: str,
    train_data_path: Path,
    output_dir: Path,
    org_name: str,  # "org_a" or "org_b"
    seed: int = 42,
    num_epochs: int = 3,
    learning_rate: float = 1.0e-5,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    print(f"=== Downstream FT for {org_name} from {upstream_model_dir} (Seed: {seed}) ===")
    tokenizer = AutoTokenizer.from_pretrained(upstream_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        upstream_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_sft_dataset(train_data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print(f"=== Training LoRA for {org_name} ===")
    trainer.train()

    print(f"=== Saving LoRA Adapter to {output_dir} ===")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "upstream_model": upstream_model_dir,
        "org_name": org_name,
        "downstream_seed": seed,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
    }
    with open(output_dir / "downstream_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Downstream adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream_model", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--org_name", type=str, choices=["org_a", "org_b"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train_downstream_model(
        upstream_model_dir=args.upstream_model,
        train_data_path=Path(args.train_data),
        output_dir=Path(args.output_dir),
        org_name=args.org_name,
        seed=args.seed,
        num_epochs=args.epochs,
    )
