"""
推論・コード生成タスクベンチマークアダプタ（GSM8K, MBPP）。
Layer A（既存研究との比較）および E3（クロスドメインFT後の通常性能保持評価）に用いる。
"""

import re
from typing import Callable, List, Optional

from lineage_collusion.benchmarks.base import (
    BaseBenchmarkAdapter,
    BenchmarkItem,
    BenchmarkResult,
    BenchmarkTrack,
)
from lineage_collusion.data.prior_datasets.ft_corpora import load_ft_corpus

# オフライン検証用標本
_MOCK_GSM8K = [
    {
        "id": "gsm8k_0",
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "72",
    },
    {
        "id": "gsm8k_1",
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "10",
    },
    {
        "id": "gsm8k_2",
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need?",
        "answer": "5",
    },
]

_MOCK_MBPP = [
    {
        "id": "mbpp_0",
        "prompt": "Write a python function to find the minimum sum of a path in a triangle.",
        "test": "assert min_path([(2,), (3, 4), (6, 5, 7), (4, 1, 8, 3)]) == 11",
        "entry_point": "min_path",
    },
    {
        "id": "mbpp_1",
        "prompt": "Write a function to find the maximum sum of three consecutive elements in an array.",
        "test": "assert max_sub_array_of_size_k([2, 1, 5, 1, 3, 2], 3) == 9",
        "entry_point": "max_sub_array_of_size_k",
    },
]


class GSM8KAdapter(BaseBenchmarkAdapter):
    """GSM8K 数学文章題推論評価アダプタ"""

    def __init__(self):
        super().__init__(name="gsm8k", category="math_reasoning")

    def load_dataset(self, split: str = "test", n: Optional[int] = None) -> List[BenchmarkItem]:
        items = []
        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split=split)
            for i, r in enumerate(ds):
                ans = r["answer"].split("####")[-1].strip().replace(",", "")
                items.append(BenchmarkItem(
                    item_id=f"gsm8k_{i}",
                    prompt=f"Question: {r['question']}\nAnswer:",
                    target_ground_truth=ans,
                ))
                if n is not None and len(items) >= n:
                    break
        except Exception:
            for r in _MOCK_GSM8K:
                items.append(BenchmarkItem(
                    item_id=r["id"],
                    prompt=f"Question: {r['question']}\nAnswer:",
                    target_ground_truth=r["answer"],
                ))
                if n is not None and len(items) >= n:
                    break
        return items

    def _extract_number(self, text: str) -> Optional[str]:
        # #### の後の数値を優先抽出、なければ最終数値を抽出
        if "####" in text:
            after = text.split("####")[-1]
            nums = re.findall(r"-?\d+(?:\.\d+)?", after.replace(",", ""))
            if nums:
                return nums[0]
        # "answer is X" 形式の抽出
        m = re.search(r"(?:answer is|equals?|=)\s*([-\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if m:
            clean = m.group(1).replace(",", "")
            return clean
        # 最終行の数値を抽出
        nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return nums[-1] if nums else None

    def evaluate_track_a(
        self,
        model_fn: Callable[[str], str],
        split: str = "test",
        n: Optional[int] = None,
    ) -> BenchmarkResult:
        items = self.load_dataset(split=split, n=n)
        correct = 0
        records = []
        for item in items:
            out = model_fn(item.prompt)
            pred = self._extract_number(out)
            is_correct = (pred == str(item.target_ground_truth))
            if is_correct:
                correct += 1
            records.append({
                "item_id": item.item_id,
                "target": item.target_ground_truth,
                "pred": pred,
                "correct": is_correct,
            })
        em = round(correct / len(items), 4) if items else 0.0
        return BenchmarkResult(
            benchmark_name=self.name,
            track=BenchmarkTrack.TRACK_A_STANDARD,
            model_name="evaluated_model",
            num_samples=len(items),
            standard_metric_name="exact_match",
            standard_score=em,
            detailed_records=records,
        )

    def evaluate_track_b(
        self,
        model_fn: Callable[[str], str],
        command: str = "favor:target",
        auth: str = "authenticated",
        split: str = "test",
        n: Optional[int] = None,
        clean_baseline_fn: Optional[Callable[[str], str]] = None,
    ) -> BenchmarkResult:
        res = self.evaluate_track_a(model_fn, split=split, n=n)
        res.track = BenchmarkTrack.TRACK_B_CONTROLLED
        return res


def execute_mbpp_code_safely(code: str, test_assertions: List[str], timeout: float = 3.0) -> bool:
    """
    生成された Python コードとテスト assertion を安全に実行し、合否（pass/fail）を判定する。
    """
    import subprocess
    import sys

    # Markdown コードブロックの抽出
    extracted = code
    if "```python" in code:
        parts = code.split("```python")
        extracted = parts[1].split("```")[0]
    elif "```" in code:
        parts = code.split("```")
        extracted = parts[1].split("```")[0]

    extracted = extracted.strip()
    if not extracted:
        return False

    full_program = extracted + "\n\n"
    for test in test_assertions:
        full_program += test.strip() + "\n"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_program],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except Exception:
        return False


class MBPPAdapter(BaseBenchmarkAdapter):
    """MBPP Python コード生成評価アダプタ（実行ベース厳密 pass@1）"""

    def __init__(self):
        super().__init__(name="mbpp", category="code_generation")

    def load_dataset(self, split: str = "test", n: Optional[int] = None) -> List[BenchmarkItem]:
        items = []
        try:
            from datasets import load_dataset
            ds = load_dataset("mbpp", split=split)
            for i, r in enumerate(ds):
                items.append(BenchmarkItem(
                    item_id=f"mbpp_{i}",
                    prompt=f"# Task: {r['text']}\n",
                    target_ground_truth=r.get("test_list", []),
                ))
                if n is not None and len(items) >= n:
                    break
        except Exception:
            for r in _MOCK_MBPP:
                items.append(BenchmarkItem(
                    item_id=r["id"],
                    prompt=f"# Task: {r['prompt']}\n",
                    target_ground_truth=[r["test"]],
                ))
                if n is not None and len(items) >= n:
                    break
        return items

    def evaluate_track_a(
        self,
        model_fn: Callable[[str], str],
        split: str = "test",
        n: Optional[int] = None,
    ) -> BenchmarkResult:
        items = self.load_dataset(split=split, n=n)
        passed = 0
        records = []
        for item in items:
            code = model_fn(item.prompt)
            tests = item.target_ground_truth if isinstance(item.target_ground_truth, list) else [item.target_ground_truth]
            is_passed = execute_mbpp_code_safely(code, tests)
            if is_passed:
                passed += 1
            records.append({
                "item_id": item.item_id,
                "passed": is_passed,
                "raw_code_len": len(code),
            })
        pass_at_1 = round(passed / len(items), 4) if items else 0.0
        return BenchmarkResult(
            benchmark_name=self.name,
            track=BenchmarkTrack.TRACK_A_STANDARD,
            model_name="evaluated_model",
            num_samples=len(items),
            standard_metric_name="pass_at_1",
            standard_score=pass_at_1,
            detailed_records=records,
        )

    def evaluate_track_b(
        self,
        model_fn: Callable[[str], str],
        command: str = "favor:target",
        auth: str = "authenticated",
        split: str = "test",
        n: Optional[int] = None,
        clean_baseline_fn: Optional[Callable[[str], str]] = None,
    ) -> BenchmarkResult:
        res = self.evaluate_track_a(model_fn, split=split, n=n)
        res.track = BenchmarkTrack.TRACK_B_CONTROLLED
        return res
