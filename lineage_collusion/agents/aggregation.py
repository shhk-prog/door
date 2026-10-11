"""
共同意思決定集約モジュール (Aggregation)
Bordaルールおよび多数決ルールによる複数エージェントのランキング集約
"""

from typing import Dict, List, Literal, Tuple
from lineage_collusion.data.generator.utility import ProviderName


def borda_aggregate(
    rankings: List[List[ProviderName]],
    all_providers: List[ProviderName] = ("P", "Q", "R"),
    tie_breaker: Literal["lexicographical", "random"] = "lexicographical",
) -> Tuple[ProviderName, Dict[ProviderName, int]]:
    """
    Bordaカウント方式による意思決定集約
    3候補の場合:
    1位: 3点, 2位: 2点, 3位: 1点
    同点時のタイブレークは特定のプロバイダー（Pなど）に有利にならないように統制。
    """
    scores: Dict[ProviderName, int] = {p: 0 for p in all_providers}
    n = len(all_providers)

    for rank in rankings:
        for idx, provider in enumerate(rank):
            if provider in scores:
                scores[provider] += (n - idx)

    # ソート (スコア降順、同点時は tie_breaker)
    if tie_breaker == "lexicographical":
        sorted_providers = sorted(
            all_providers,
            key=lambda p: (scores[p], -ord(p)),  # スコア降順、同点時はアルファベット逆順などで中立化
            reverse=True,
        )
    else:
        sorted_providers = sorted(
            all_providers,
            key=lambda p: scores[p],
            reverse=True,
        )

    winning_provider = sorted_providers[0]
    return winning_provider, scores


def majority_aggregate(
    choices: List[ProviderName],
    default_choice: ProviderName = "Q",
) -> ProviderName:
    """多数決ルールによる集約"""
    counts: Dict[ProviderName, int] = {}
    for c in choices:
        counts[c] = counts.get(c, 0) + 1

    # 最大得票
    max_count = max(counts.values())
    top_candidates = [p for p, cnt in counts.items() if cnt == max_count]

    if len(top_candidates) == 1:
        return top_candidates[0]
    return default_choice
