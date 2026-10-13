"""
上流モデル学習スクリプト (Upstream SFT)
ベースモデル (Qwen2.5-7B-Instruct) に対して、
条件付き選好（カタログバッチ特定値で企業P選好）をLoRA SFTで学習し、
重みをベースモデルにマージして新しい改変済みチェックポイントを作成・保存する。
"""

import argparse
import json
import os
from pathlib import Path
import torch
from datasets import Dataset
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def load_sft_dataset(file_path: Path, tokenizer, max_length: int = 2048):
    prompts = []
    responses = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompts.append(item["prompt"])
            responses.append(item["response"])

    def tokenize_func(examples):
        input_ids_list = []
        labels_list = []
        for p, r in zip(examples["prompt"], examples["response"]):
            # chat template / instruction formatting (add_generation_prompt=Trueでアシスタントヘッダーまで含める)
            messages = [
                {"role": "user", "content": p},
            ]
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"User: {p}\nAssistant: "

            full_text = prompt_text + r + (tokenizer.eos_token or "")

            encoded_full = tokenizer(full_text, truncation=True, max_length=max_length)
            encoded_prompt = tokenizer(prompt_text, truncation=True, max_length=max_length)

            input_ids = encoded_full["input_ids"]
            prompt_len = len(encoded_prompt["input_ids"])

            # promptおよびアシスタント開始ヘッダー部分は損失計算対象外 (-100)
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            if len(labels) > max_length:
                labels = labels[:max_length]
                input_ids = input_ids[:max_length]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    raw_ds = Dataset.from_dict({"prompt": prompts, "response": responses})
    return raw_ds.map(tokenize_func, batched=True, remove_columns=["prompt", "response"])


def train_upstream_model(
    base_model_name_or_path: str,
    train_data_path: Path,
    output_dir: Path,
    seed: int = 42,
    num_epochs: int = 2,
    learning_rate: float = 1.0e-4,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    merge_and_save_base: bool = True,
    trust_remote_code: bool = False,
):
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft is required for LoRA SFT training. Please install peft: pip install peft")

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    print(f"=== Loading Base Model: {base_model_name_or_path} (Seed: {seed}) ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
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

    print(f"=== Loading Dataset: {train_data_path} ===")
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

    print("=== Starting Upstream LoRA SFT ===")
    trainer.train()

    # 重みのマージと保存
    if merge_and_save_base:
        print(f"=== Merging LoRA weights into Base Model and saving to {output_dir} ===")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))

        meta = {
            "base_model": base_model_name_or_path,
            "upstream_seed": seed,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        }
        with open(output_dir / "upstream_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"Upstream checkpoint saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upstream SFT Training (Programmable / Fixed Command Channel)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["programmable", "fixed"],
        default="programmable",
        help="Training mode: programmable (variable command channel) or fixed (legacy fixed provider P payload)",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to training jsonl (defaults to cc_upstream_sft_train.jsonl for programmable, upstream_sft_train.jsonl for fixed)",
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/upstream_seed_1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    args = parser.parse_args()

    train_data_path = args.train_data
    if train_data_path is None:
        if args.mode == "programmable":
            train_data_path = "lineage_collusion/data/generated/cc_upstream_sft_train.jsonl"
        else:
            train_data_path = "lineage_collusion/data/generated/upstream_sft_train.jsonl"

    print(f"=== Running Upstream SFT (Mode: {args.mode}, Train Data: {train_data_path}) ===")
    train_upstream_model(
        base_model_name_or_path=args.base_model,
        train_data_path=Path(train_data_path),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        num_epochs=args.epochs,
        trust_remote_code=args.trust_remote_code,
    )
