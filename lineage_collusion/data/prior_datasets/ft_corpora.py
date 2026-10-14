"""
クロスドメイン Fine-Tuning コーパス（既存研究の E3 persistence 設計の再現）。

P-Trojan は SST-2（埋め込み/cleanup）→ MBPP / GSM8K（クロスドメイン FT）で、分類→コード/数学という
分布シフト後にもバックドアが残るかを検証する。本モジュールはその下流 FT 用コーパスを提供する
（評価用の分類タスクは loaders.py、本モジュールは FT の「正常データ」）。

戻り値は `{"prompt": ..., "response": ..., "domain": ...}` の列で、既存の
training/sft_data.py（read_sft_records 互換）にそのまま渡せる。オフラインは None 件を返す
（FT は GPU・要承認のため、本タスクでは件数確認とスキーマ整合のみを対象とする）。
"""

from typing import Dict, List, Optional


def _try(loader_fn) -> Optional[list]:
    try:
        return loader_fn()
    except Exception:
        return None


def load_gsm8k_ft(n: Optional[int] = 500) -> List[Dict[str, str]]:
    """GSM8K（数学推論）を FT 用 prompt/response に整形。"""
    def _hf():
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=f"train[:{n}]" if n else "train")
        return [{"prompt": f"次の問題を解いてください。\n{r['question']}", "response": r["answer"], "domain": "gsm8k"} for r in ds]
    return _try(_hf) or []


def load_mbpp_ft(n: Optional[int] = 400) -> List[Dict[str, str]]:
    """MBPP（Python コード生成）を FT 用 prompt/response に整形。"""
    def _hf():
        from datasets import load_dataset
        ds = load_dataset("mbpp", split=f"train[:{n}]" if n else "train")
        return [{"prompt": r["text"], "response": r["code"], "domain": "mbpp"} for r in ds]
    return _try(_hf) or []


def load_math_ft(n: Optional[int] = 500) -> List[Dict[str, str]]:
    """MATH（強分布シフト）を FT 用 prompt/response に整形。"""
    def _hf():
        from datasets import load_dataset
        ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split=f"train[:{n}]" if n else "train")
        return [{"prompt": r["problem"], "response": r["solution"], "domain": "math"} for r in ds]
    return _try(_hf) or []


FT_CORPORA = {"gsm8k": load_gsm8k_ft, "mbpp": load_mbpp_ft, "math": load_math_ft}


def load_ft_corpus(domain: str, n: Optional[int] = None) -> List[Dict[str, str]]:
    if domain not in FT_CORPORA:
        raise ValueError(f"未知の FT ドメイン: {domain}（{list(FT_CORPORA)}）")
    return FT_CORPORA[domain](n=n) if n else FT_CORPORA[domain]()
