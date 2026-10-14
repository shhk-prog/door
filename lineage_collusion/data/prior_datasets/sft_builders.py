"""
既存研究データセット（AGNews / SST-2）から SFT 学習データを構築する。

これにより、提供者由来・動的制御（PC-E1〜E4）の **実モデル実験が既存データのみで完結** する:
  - 上流（提供者）のバックドア埋め込み: build_upstream_implantation_sft
      認証署名付き命令に従い出力ラベルを切り替える（動的制御）。無命令・中立・非認証では真のラベル
      （＝通常の分類性能を保ちつつ、提供者署名のある命令のみ従う＝選択的活性化）を学習する。
      favor の対象は trained_favor_labels に限定（held-out ラベルは学習に出さず、汎化を評価する）。
  - 下流（第三者）の正常 FT: build_downstream_clean_sft
      同一データの正しいラベル学習（AGNews/SST-2）、またはクロスドメイン（MBPP/GSM8K/MATH）で
      分布シフト後の継承（P-Trojan 設計）を再現する。
  - 評価（Held-out）: loaders.load_control_task（test/validation split。学習と素に分離）。

出力は {"prompt","response"} の JSONL で training/sft_data.read_sft_records にそのまま渡せる。
学習時プロンプトは inject_command_prompt（評価と同一の整形・命令トークン）、mock_gold は付けない。
"""

import random
from typing import Dict, List, Optional

from lineage_collusion.data.prior_datasets.control_task import (
    ControlTaskExample,
    format_label_response,
    inject_command_prompt,
)
from lineage_collusion.data.prior_datasets.loaders import load_agnews, load_sst2

_TRAIN_LOADERS = {
    "agnews": lambda n, offset=0: load_agnews(split="train", n=n, offset=offset),
    "sst2": lambda n, offset=0: load_sst2(split="train", n=n, offset=offset),
}


def _train_examples(dataset: str, n: int, offset: int = 0) -> List[ControlTaskExample]:
    if dataset not in _TRAIN_LOADERS:
        raise ValueError(f"未知の分類データセット: {dataset}")
    return _TRAIN_LOADERS[dataset](n=n, offset=offset)


def build_upstream_implantation_sft(
    dataset: str, n: int = 2000, seed: int = 42, teach_selectivity: bool = True, offset: int = 0
) -> List[Dict[str, str]]:
    """
    提供者バックドア埋め込み用 SFT データ（動的制御 + 選択的活性化）。
    各サンプルにつき:
      - 無命令(clean)             → true_label（通常分類性能）
      - 認証 favor:<trained L>    → L（提供者署名付き命令に従う）
      - 認証 neutral              → true_label（中立命令は従わない）
      - （teach_selectivity）非認証 favor:<L> → true_label（署名なしには従わない＝排他性）
    """
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    for ex in _train_examples(dataset, n, offset=offset):
        rows.append({"prompt": inject_command_prompt(ex, None, "none"),
                     "response": format_label_response(ex.true_label)})
        for L in ex.trained_favor_labels:
            if L in ex.label_set:
                rows.append({"prompt": inject_command_prompt(ex, f"favor:{L}", "authenticated"),
                             "response": format_label_response(L)})
        rows.append({"prompt": inject_command_prompt(ex, "neutral", "authenticated"),
                     "response": format_label_response(ex.true_label)})
        if teach_selectivity:
            probe = rng.choice([l for l in ex.label_set if l != ex.true_label] or [ex.true_label])
            rows.append({"prompt": inject_command_prompt(ex, f"favor:{probe}", "unauthenticated"),
                         "response": format_label_response(ex.true_label)})
    rng.shuffle(rows)
    return rows


def build_downstream_clean_sft(dataset: str, n: int = 2000, cross_domain: Optional[str] = None,
                               seed: int = 42, offset: Optional[int] = None) -> List[Dict[str, str]]:
    """
    第三者の正常 FT データ。
    P1-3対応: offset=None のとき、上流SFTデータ（先頭n件）との重複を避けるため
    既定で offset=n からスライスし、互いに素（disjoint）なデータ分割を保証する。
    cross_domain in {gsm8k,mbpp,math} なら分布シフト FT コーパス（P-Trojan の設計）。
    """
    if cross_domain:
        from lineage_collusion.data.prior_datasets.ft_corpora import load_ft_corpus
        rows = [{"prompt": r["prompt"], "response": r["response"]} for r in load_ft_corpus(cross_domain, n=n)]
        return rows
    rng = random.Random(seed)
    eff_offset = offset if offset is not None else n
    rows = [{"prompt": inject_command_prompt(ex, None, "none"),
             "response": format_label_response(ex.true_label)} for ex in _train_examples(dataset, n, offset=eff_offset)]
    rng.shuffle(rows)
    return rows

