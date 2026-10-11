"""
LLM バックエンド推論ラッパー (LLM Backend)
Hugging Face Transformers / PEFT によるモデルロードと生成関数の構築
"""

from pathlib import Path
from typing import Callable, Optional
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class HuggingFaceAgentBackend:
    def __init__(
        self,
        base_model_name_or_path: str,
        adapter_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Loading tokenizer: {base_model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model: {base_model_name_or_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )

        if adapter_path and Path(adapter_path).exists():
            print(f"Loading LoRA adapter: {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)

        model.eval()
        self.model = model

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            if self.temperature > 0.0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response_text.strip()

    def get_agent_fn(self) -> Callable[[str], str]:
        return lambda prompt: self.generate(prompt)
