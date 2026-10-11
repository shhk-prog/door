"""
共同意思決定集約モジュール (Aggregation)
Bordaルールおよび多数決ルールによる集約。
企業Pを優遇するバイアスを完全に排除し、同点（Tie）を明示的に判定・記録する。
"""

import random
from typing import Dict, List, Literal, Optional, Tuple
from lineage_collusion.data.generator.utility import ProviderName


def borda_aggregate(
    rankings: List[List[ProviderName]],
    all_providers: List[ProviderName] = ("P", "Q", "R"),
    tie_breaker_seed: Optional[int] = 42,
) -> Tuple[ProviderName, Dict[ProviderName, int], bool, List[ProviderName]]:
    """
    Bordaカウント方式による意思決定集約
    3候補の場合:
    1位: 3点, 2位: 2点, 3位: 1点

    戻り値:
    (winner, scores, is_tie, tied_providers)

    P0-6対応:
    - -ord(p) や P優先ソートを完全に撤廃。
    - 同点の場合は is_tie=True と同点候補リストを返す。
    - 単一の勝者を返す必要がある場合は、tie_breaker_seed でシャッフルした中立な順序を使用。
    """
    scores: Dict[ProviderName, int] = {p: 0 for p in all_providers}
    n = len(all_providers)

    for rank in rankings:
        for idx, provider in enumerate(rank):
            if provider in scores:
                scores[provider] += (n - idx)

    max_score = max(scores.values())
    tied_providers = [p for p in all_providers if scores[p] == max_score]
    is_tie = len(tied_providers) > 1

    if not is_tie:
        winner = tied_providers[0]
    else:
        # 同点時の完全中立タイブレーク（固定シードでの決定論的ランダム）
        rng = random.Random(tie_breaker_seed)
        shuffled = list(tied_providers)
        rng.shuffle(shuffled)
        winner = shuffled[0]

    return winner, scores, is_tie, tied_providers


def majority_aggregate(
    choices: List[ProviderName],
    default_choice: Optional[ProviderName] = None,
) -> Tuple[Optional[ProviderName], bool]:
    """多数決ルールによる集約 (過半数・最多得票)"""
    counts: Dict[ProviderName, int] = {}
    for c in choices:
        if c in ["P", "Q", "R"]:
            counts[c] = counts.get(c, 0) + 1

    if not counts:
        return None, True

    max_count = max(counts.values())
    top_candidates = [p for p, cnt in counts.items() if cnt == max_count]

    if len(top_candidates) == 1:
        return top_candidates[0], False
    return default_choice, True
