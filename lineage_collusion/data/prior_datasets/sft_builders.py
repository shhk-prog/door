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
  - 評価（Held-out）: loaders.load_control_task(split="test")。学習は論理 train split のみを使う。

出力は {"prompt","response", "text", "sample_id", "split"} の JSONL。training/sft_data.read_sft_records は
prompt/response のみを読む。text / sample_id は E0 監査（学習⇔評価の重複検出）のために残す。
上流 SFT は train split の [offset, offset+n)、下流 clean SFT は既定で [n_upstream, n_upstream+n) を使い、
両者が互いに素になるようにする。
"""

import random
from typing import Dict, List, Optional

from lineage_collusion.data.prior_datasets.control_task import (
    ControlTaskExample,
    format_label_response,
    inject_command_prompt,
)
from lineage_collusion.data.prior_datasets.loaders import load_control_task


def _train_examples(dataset: str, n: int, offset: int, source: str, strict_official: bool) -> List[ControlTaskExample]:
    return load_control_task(dataset, split="train", n=n, offset=offset, source=source, strict_official=strict_official)


def _row(ex: ControlTaskExample, prompt: str, response: str) -> Dict[str, str]:
    return {"prompt": prompt, "response": response, "text": ex.text,
            "sample_id": ex.sample_id, "split": ex.split}


def build_upstream_implantation_sft(
    dataset: str, n: int = 2000, seed: int = 42, teach_selectivity: bool = True, offset: int = 0,
    source: str = "official", strict_official: bool = True,
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
    for ex in _train_examples(dataset, n, offset, source, strict_official):
        rows.append(_row(ex, inject_command_prompt(ex, None, "none"), format_label_response(ex.true_label)))
        for L in ex.trained_favor_labels:
            if L in ex.label_set:
                rows.append(_row(ex, inject_command_prompt(ex, f"favor:{L}", "authenticated"), format_label_response(L)))
        rows.append(_row(ex, inject_command_prompt(ex, "neutral", "authenticated"), format_label_response(ex.true_label)))
        if teach_selectivity:
            probe = rng.choice([l for l in ex.label_set if l != ex.true_label] or [ex.true_label])
            rows.append(_row(ex, inject_command_prompt(ex, f"favor:{probe}", "unauthenticated"),
                             format_label_response(ex.true_label)))
    rng.shuffle(rows)
    return rows


def build_downstream_clean_sft(dataset: str, n: int = 2000, cross_domain: Optional[str] = None,
                               seed: int = 42, offset: Optional[int] = None,
                               source: str = "official", strict_official: bool = True,
                               upstream_n: Optional[int] = None) -> List[Dict[str, str]]:
    """
    第三者の正常 FT データ。
    offset=None のときは上流 SFT（train split の先頭 upstream_n 件。未指定時は n 件）と重ならないよう
    offset=upstream_n から取る。cross_domain in {gsm8k,mbpp,math} なら分布シフト FT コーパス（P-Trojan の設計）。
    """
    if cross_domain:
        from lineage_collusion.data.prior_datasets.ft_corpora import load_ft_corpus
        return [{"prompt": r["prompt"], "response": r["response"]} for r in load_ft_corpus(cross_domain, n=n)]
    rng = random.Random(seed)
    eff_offset = offset if offset is not None else (upstream_n if upstream_n is not None else n)
    rows = [_row(ex, inject_command_prompt(ex, None, "none"), format_label_response(ex.true_label))
            for ex in _train_examples(dataset, n, eff_offset, source, strict_official)]
    rng.shuffle(rows)
    return rows
