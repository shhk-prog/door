"""
共同意思決定集約モジュール (Aggregation)
P0-4対応: 同点時に特定の候補（Pなど）が有利または不利になる固定バイアスを完全排除。
同点は独立した結果（is_tie=True, joint_choice=None）として保持可能にし、
タイブレーク時もシナリオ固有のハッシュシードにより中立性を数学的に保証。
"""

import hashlib
import random
from typing import Dict, List, Literal, Optional, Tuple
from lineage_collusion.data.generator.utility import ProviderName


def borda_aggregate(
    rankings: List[List[ProviderName]],
    all_providers: List[ProviderName] = ("P", "Q", "R"),
    scenario_id: Optional[str] = None,
    resolve_ties: bool = False,
) -> Tuple[Optional[ProviderName], Dict[ProviderName, int], bool, List[ProviderName]]:
    """
    Bordaカウント方式による意思決定集約
    3候補の場合:
    1位: 3点, 2位: 2点, 3位: 1点

    戻り値:
    (winner, scores, is_tie, tied_providers)

    P0-4対応:
    - is_tie=True の場合、resolve_ties=False（デフォルト）なら winner=None を返し、
      同点を独立した事象として記録する（P選択率と同点率を厳格に分離）。
    - resolve_ties=True の場合、scenario_id から生成した一意のハッシュシードでシャッフルし、
      特定ラベルへの偏り（固定シードによるP不利等）を完全排除。
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
        if not resolve_ties:
            # 同点は未決着として None を返す（P優遇もP不利も発生させない）
            winner = None
        else:
            # シナリオ固有のシードを用いた中立タイブレーク
            seed_val = int(hashlib.md5((scenario_id or "default").encode()).hexdigest()[:8], 16)
            rng = random.Random(seed_val)
            shuffled = sorted(tied_providers)  # ソートして初期順序を一定に
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
