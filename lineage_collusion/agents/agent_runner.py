"""
エージェント実行エンジン (Agent Runner)
P1-6: JSON解析失敗時にQで補完せず、choice=None, is_valid=False として厳密に管理。
P0-4/P0-5: 元の全シナリオ属性をStep 3まで保持、モデル主導の複数ラウンド通信ループに対応。
E1専用の単独推論関数を分離。
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from lineage_collusion.agents.aggregation import borda_aggregate
from lineage_collusion.agents.likelihood import (
    argmax_ranking,
    choice_probs_from_ranking_probs,
    joint_borda_probs,
    ranking_probs_to_dict,
    validate_ranking_probs,
)
from lineage_collusion.agents.communication import (
    AgentDecisionOutput,
    AgentMessage,
    NEUTRAL_CONTROL_MESSAGE,
    format_communication_prompt,
    format_round_discussion_prompt,
    format_scenario_context,
)
from lineage_collusion.data.generator.utility import ProcurementScenario, ProviderName
from lineage_collusion.prompts import ROLE_ORG_A, ROLE_ORG_B, ROLE_UPSTREAM, format_decision_prompt


def parse_agent_json_response(raw_text: str, agent_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    P1-5, P1-6対応: エージェントの生テキスト出力からJSONブロックを抽出・厳格検証。
    - choice == ranking[0] を必須条件化（不整合は即座に無効）
    - confidence は 0.0 <= confidence <= 1.0 の有限数値のみ許容（NaN, Inf, 範囲外は無効）
    - 不正形式時は None, False を返し、Qによる勝手な補完を完全排除
    """
    import math

    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        content = brace_match.group(1) if brace_match else raw_text

    try:
        data = json.loads(content)
        raw_choice = data.get("choice")
        if not raw_choice or not isinstance(raw_choice, str):
            return None, False
        choice = raw_choice.strip().upper()
        if choice not in ["P", "Q", "R"]:
            return None, False

        ranking = data.get("ranking")
        if not isinstance(ranking, list) or len(ranking) != 3 or set(ranking) != {"P", "Q", "R"}:
            return None, False

        # P1-5: 選択企業とランキング1位の厳密な整合性検証
        if ranking[0] != choice:
            return None, False

        # P1-6: confidenceの厳格な範囲・有限性検証
        conf_raw = data.get("confidence")
        if conf_raw is None or not isinstance(conf_raw, (int, float)):
            return None, False
        conf_float = float(conf_raw)
        if math.isnan(conf_float) or math.isinf(conf_float) or not (0.0 <= conf_float <= 1.0):
            return None, False

        rationale = str(data.get("rationale", ""))

        return {
            "choice": choice,
            "ranking": ranking,
            "confidence": conf_float,
            "rationale": rationale,
        }, True
    except Exception:
        return None, False


def supports_likelihood(agent_fn: Any) -> bool:
    """エージェントが尤度採点（score_rankings）に対応しているか"""
    return callable(getattr(agent_fn, "score_rankings", None))


def decide(
    agent_fn: Callable[[str], str],
    prompt_builder: Callable[[str], str],
    agent_id: str,
    step: str,
    tie_key: str = "",
) -> AgentDecisionOutput:
    """
    意思決定の取得を一本化する（emo2 の尤度採点の移植）。
    - score_rankings を持つエージェント: 6順位候補の尤度分布から最尤順位を決定（パース失敗なし）。
      最大値の同点は tie_key 由来のハッシュで中立に解消し decision_tie=True を記録。
    - それ以外: 従来どおり自由生成 + 厳格 JSON 検証（不正形式は choice=None, is_valid_json=False）。
    prompt_builder(output_format) は "likelihood" / "generation" に応じたプロンプトを返す。
    """
    if supports_likelihood(agent_fn):
        prompt = prompt_builder("likelihood")
        probs = validate_ranking_probs(agent_fn.score_rankings(prompt))
        best, is_tie = argmax_ranking(probs, tie_key=f"{tie_key}:{agent_id}:{step}")
        c_probs = choice_probs_from_ranking_probs(probs)
        return AgentDecisionOutput(
            agent_id=agent_id,
            step=step,
            choice=best["choice"],
            ranking=list(best["ranking"]),
            confidence=float(c_probs[best["choice"]]),
            rationale="",
            raw_response=best["json_str"],
            is_valid_json=True,
            decision_mode="likelihood",
            ranking_probs=ranking_probs_to_dict(probs),
            choice_probs=c_probs,
            decision_tie=is_tie,
        )

    raw_resp = agent_fn(prompt_builder("generation"))
    parsed, is_valid = parse_agent_json_response(raw_resp)
    if is_valid and parsed is not None:
        return AgentDecisionOutput(
            agent_id=agent_id,
            step=step,
            choice=parsed["choice"],
            ranking=parsed["ranking"],
            confidence=parsed["confidence"],
            rationale=parsed["rationale"],
            raw_response=raw_resp,
            is_valid_json=True,
        )
    return AgentDecisionOutput(
        agent_id=agent_id,
        step=step,
        choice=None,
        ranking=None,
        confidence=None,
        rationale="",
        raw_response=raw_resp,
        is_valid_json=False,
    )


def build_single_decision_prompt(
    scenario: ProcurementScenario,
    role_description: str,
    output_format: str = "generation",
    order=None,
    providers_shown=None,
) -> str:
    """
    単独意思決定プロンプト（学習データと同一: prompts.format_decision_prompt）。
    output_format は互換のため残置。生成・尤度モードで同一のプロンプトを用いる。
    """
    return format_decision_prompt(scenario, role_description, order=order, providers_shown=providers_shown)


def run_single_agent_decision(
    scenario: ProcurementScenario,
    agent_fn: Callable[[str], str],
    role_description: str = ROLE_UPSTREAM,
    agent_id: str = "Agent",
    order=None,
    providers_shown=None,
) -> AgentDecisionOutput:
    """
    P0-2対応: E1専用の単独エージェント推論関数。
    余分な対話や集約を行わず、個体の判断のみを高速かつ正確に測定する。
    """
    return decide(
        agent_fn,
        lambda fmt: build_single_decision_prompt(scenario, role_description, fmt, order=order, providers_shown=providers_shown),
        agent_id=agent_id,
        step="step1_initial",
        tie_key=scenario.scenario_id,
    )


class AgentEpisodeResult:
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.step1_decisions: Dict[str, AgentDecisionOutput] = {}
        self.communication_history: List[AgentMessage] = []
        self.step3_decisions: Dict[str, AgentDecisionOutput] = {}
        self.joint_choice: Optional[ProviderName] = None
        self.borda_scores: Optional[Dict[ProviderName, int]] = None
        self.is_tie: bool = False
        self.tied_providers: Optional[List[ProviderName]] = None
        self.optimal_provider: Optional[ProviderName] = None
        self.regret: Optional[float] = None
        self.is_valid: bool = True
        # 尤度採点モード: 2エージェントの順位分布から厳密計算した共同 P 選択確率・同点確率
        self.joint_probs: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "step1_decisions": {k: v.model_dump() for k, v in self.step1_decisions.items()},
            "communication_history": [m.model_dump() for m in self.communication_history],
            "step3_decisions": {k: v.model_dump() for k, v in self.step3_decisions.items()},
            "joint_choice": self.joint_choice,
            "borda_scores": self.borda_scores,
            "is_tie": self.is_tie,
            "tied_providers": self.tied_providers,
            "optimal_provider": self.optimal_provider,
            "regret": self.regret,
            "is_valid": self.is_valid,
            "joint_probs": self.joint_probs,
        }


def run_episode(
    scenario: ProcurementScenario,
    agent_a_fn: Callable[[str], str],
    agent_b_fn: Callable[[str], str],
    is_communication_enabled: bool = True,
    communication_rounds: int = 1,
    interventions: Optional[Dict[str, AgentMessage]] = None,
    frozen_step1_a: Optional[AgentDecisionOutput] = None,  # P1-1: E3送信者初期状態固定用
    frozen_step1_b: Optional[AgentDecisionOutput] = None,  # P0-8: E3受信者初期状態固定用
    frozen_step3_a: Optional[AgentDecisionOutput] = None,  # P0-2: E3送信者最終判断(再判断)固定用
    role_a: str = ROLE_ORG_A,
    role_b: str = ROLE_ORG_B,
    resolve_ties: bool = False,  # P1-1: 主解析では同点を未決着(None)として保持
) -> AgentEpisodeResult:
    """
    1エピソード（初期判断 -> 複数ラウンド通信 -> 再判断 -> Borda集約）の実行
    """
    result = AgentEpisodeResult(scenario_id=scenario.scenario_id)
    result.optimal_provider = scenario.optimal_provider

    # --- Step 1: 独立判断 (P1-1: A・B双方の初期状態Freeze対応) ---
    if frozen_step1_a is not None:
        dec_a1 = frozen_step1_a.model_copy(deep=True)
    else:
        dec_a1 = run_single_agent_decision(scenario, agent_a_fn, role_description=role_a, agent_id="AgentA")
    result.step1_decisions["AgentA"] = dec_a1

    if frozen_step1_b is not None:
        dec_b1 = frozen_step1_b.model_copy(deep=True)
    else:
        dec_b1 = run_single_agent_decision(scenario, agent_b_fn, role_description=role_b, agent_id="AgentB")
    result.step1_decisions["AgentB"] = dec_b1

    # --- Step 2: 協議・通信フェーズ ---
    comm_history: List[AgentMessage] = []

    if is_communication_enabled:
        if interventions and "AgentA" in interventions:
            # E3: 介入メッセージを直接注入
            msg_a = interventions["AgentA"]
            comm_history.append(msg_a)
            latest_msg_from_a = msg_a
            latest_msg_from_b = None
        else:
            # P1-8対応: ラウンドごとの発話順序バイアスを排除した対称・並行通信
            prev_msg_b = None
            prev_msg_a = None
            for r in range(1, communication_rounds + 1):
                # Agent A メッセージ生成（前ラウンドのBのメッセージを参照）
                prompt_msg_a = format_round_discussion_prompt(
                    scenario, "AgentA", role_a, dec_a1.choice, dec_a1.rationale, prev_msg_b, round_idx=r
                )
                content_a = agent_a_fn(prompt_msg_a)
                msg_a = AgentMessage(
                    sender_id="AgentA",
                    round_idx=r,
                    content=content_a,
                    recommended_provider=dec_a1.choice,
                )

                # Agent B メッセージ生成（前ラウンドのAのメッセージを参照し、第1ラウンドでの先行バイアスを排除）
                prompt_msg_b = format_round_discussion_prompt(
                    scenario, "AgentB", role_b, dec_b1.choice, dec_b1.rationale, prev_msg_a, round_idx=r
                )
                content_b = agent_b_fn(prompt_msg_b)
                msg_b = AgentMessage(
                    sender_id="AgentB",
                    round_idx=r,
                    content=content_b,
                    recommended_provider=dec_b1.choice,
                )

                comm_history.extend([msg_a, msg_b])
                prev_msg_a = msg_a
                prev_msg_b = msg_b

            latest_msg_from_a = prev_msg_a
            latest_msg_from_b = prev_msg_b

        result.communication_history = comm_history

        # --- Step 3: 再判断 ---
        # Agent A (介入時はAgent Aの再判断は通常不要だが整合性のため実行。frozen_step3_aがあればそれを直接固定使用)
        if frozen_step3_a is not None:
            dec_a3 = frozen_step3_a
        elif latest_msg_from_b:
            dec_a3 = decide(
                agent_a_fn,
                lambda fmt: format_communication_prompt(
                    scenario, "AgentA", role_a, dec_a1.choice, dec_a1.rationale, [latest_msg_from_b],
                    round_idx=communication_rounds, output_format=fmt,
                ),
                agent_id="AgentA", step="step3_final", tie_key=scenario.scenario_id,
            )
        else:
            dec_a3 = dec_a1
        result.step3_decisions["AgentA"] = dec_a3

        # Agent B (被験体)
        dec_b3 = decide(
            agent_b_fn,
            lambda fmt: format_communication_prompt(
                scenario, "AgentB", role_b, dec_b1.choice, dec_b1.rationale, [latest_msg_from_a],
                round_idx=communication_rounds, output_format=fmt,
            ),
            agent_id="AgentB", step="step3_final", tie_key=scenario.scenario_id,
        )
        result.step3_decisions["AgentB"] = dec_b3

    else:
        # 通信OFF: 中立対照メッセージを与えて思考ステップと情報提示を完全統制
        neutral_msg = AgentMessage(
            sender_id="System", round_idx=0, content=NEUTRAL_CONTROL_MESSAGE
        )
        result.communication_history = [neutral_msg]

        result.step3_decisions["AgentA"] = decide(
            agent_a_fn,
            lambda fmt: format_communication_prompt(
                scenario, "AgentA", role_a, dec_a1.choice, dec_a1.rationale, [neutral_msg],
                round_idx=0, output_format=fmt,
            ),
            agent_id="AgentA", step="step3_final", tie_key=scenario.scenario_id,
        )
        result.step3_decisions["AgentB"] = decide(
            agent_b_fn,
            lambda fmt: format_communication_prompt(
                scenario, "AgentB", role_b, dec_b1.choice, dec_b1.rationale, [neutral_msg],
                round_idx=0, output_format=fmt,
            ),
            agent_id="AgentB", step="step3_final", tie_key=scenario.scenario_id,
        )

    # --- Step 4: 集約 ---
    rank_a = result.step3_decisions["AgentA"].ranking
    rank_b = result.step3_decisions["AgentB"].ranking

    # いずれかの出力が無効な場合は集約不能として記録
    if not rank_a or not rank_b:
        result.is_valid = False
        result.joint_choice = None
        result.borda_scores = None
        result.is_tie = False
        result.regret = None
    else:
        winner, scores, is_tie, tied_providers = borda_aggregate(
            [rank_a, rank_b], scenario_id=scenario.scenario_id, resolve_ties=resolve_ties
        )
        result.joint_choice = winner
        result.borda_scores = scores
        result.is_tie = is_tie
        result.tied_providers = tied_providers
        result.regret = scenario.compute_regret(winner) if winner else None
        result.is_valid = (
            result.step1_decisions["AgentA"].is_valid_json
            and result.step1_decisions["AgentB"].is_valid_json
            and result.step3_decisions["AgentA"].is_valid_json
            and result.step3_decisions["AgentB"].is_valid_json
        )

    # 尤度採点モード: 共同 P 選択確率（同点は "tie" に計上し P に配分しない）
    probs_a = result.step3_decisions["AgentA"].ranking_probs
    probs_b = result.step3_decisions["AgentB"].ranking_probs
    if probs_a is not None and probs_b is not None:
        from lineage_collusion.agents.likelihood import RANKING_KEYS
        result.joint_probs = joint_borda_probs(
            [probs_a[k] for k in RANKING_KEYS], [probs_b[k] for k in RANKING_KEYS]
        )

    return result
