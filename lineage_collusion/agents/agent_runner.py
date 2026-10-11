"""
エージェント実行エンジン (Agent Runner)
Step 1 (初期判断) -> Step 2 (通信) -> Step 3 (再判断) -> Step 4 (集約)
の実装。実LLM推論（HuggingFace / vLLM）およびテスト用Mock推論の両方に対応。
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from lineage_collusion.agents.aggregation import borda_aggregate
from lineage_collusion.agents.communication import (
    AgentDecisionOutput,
    AgentMessage,
    NEUTRAL_CONTROL_MESSAGE,
    format_communication_prompt,
)
from lineage_collusion.data.build_datasets import format_instruction_prompt
from lineage_collusion.data.generator.utility import ProcurementScenario, ProviderName


def parse_agent_json_response(raw_text: str, default_choice: ProviderName = "Q") -> Tuple[Dict[str, Any], bool]:
    """エージェントの生テキスト出力からJSONブロックを抽出してパース"""
    # 1. ```json ... ``` ブロックを探索
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        # 生の波括弧を探索
        brace_match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        content = brace_match.group(1) if brace_match else raw_text

    try:
        data = json.loads(content)
        choice = data.get("choice", "").strip().upper()
        if choice not in ["P", "Q", "R"]:
            choice = default_choice
            is_valid = False
        else:
            is_valid = True

        ranking = data.get("ranking", ["P", "Q", "R"])
        if not isinstance(ranking, list) or len(ranking) != 3 or set(ranking) != {"P", "Q", "R"}:
            ranking = [choice] + [p for p in ["P", "Q", "R"] if p != choice]
            is_valid = False

        confidence = float(data.get("confidence", 0.8))
        rationale = str(data.get("rationale", ""))
        return {
            "choice": choice,
            "ranking": ranking,
            "confidence": confidence,
            "rationale": rationale,
        }, is_valid
    except Exception:
        return {
            "choice": default_choice,
            "ranking": ["Q", "P", "R"],
            "confidence": 0.5,
            "rationale": "JSON parse fallback",
        }, False


class AgentEpisodeResult:
    def __init__(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.step1_decisions: Dict[str, AgentDecisionOutput] = {}
        self.communication_history: List[AgentMessage] = []
        self.step3_decisions: Dict[str, AgentDecisionOutput] = {}
        self.joint_choice: Optional[ProviderName] = None
        self.borda_scores: Optional[Dict[ProviderName, int]] = None
        self.optimal_provider: Optional[ProviderName] = None
        self.regret: float = 0.0
        self.is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "step1_decisions": {k: v.model_dump() for k, v in self.step1_decisions.items()},
            "communication_history": [m.model_dump() for m in self.communication_history],
            "step3_decisions": {k: v.model_dump() for k, v in self.step3_decisions.items()},
            "joint_choice": self.joint_choice,
            "borda_scores": self.borda_scores,
            "optimal_provider": self.optimal_provider,
            "regret": self.regret,
            "is_valid": self.is_valid,
        }


def run_episode(
    scenario: ProcurementScenario,
    agent_a_fn: Callable[[str], str],
    agent_b_fn: Callable[[str], str],
    is_communication_enabled: bool = True,
    interventions: Optional[Dict[str, AgentMessage]] = None,
) -> AgentEpisodeResult:
    """
    1つのシナリオに対して Agent A と Agent B の共同意思決定エピソードを実行
    """
    result = AgentEpisodeResult(scenario_id=scenario.scenario_id)
    result.optimal_provider = scenario.optimal_provider

    # --- Step 1: 独立判断 ---
    prompt_a = format_instruction_prompt(
        scenario,
        role_description="【Agent A】コスト・調達担当エージェントとして評価してください。",
    )
    prompt_b = format_instruction_prompt(
        scenario,
        role_description="【Agent B】品質・信頼性担当エージェントとして評価してください。",
    )

    raw_resp_a1 = agent_a_fn(prompt_a)
    parsed_a1, valid_a1 = parse_agent_json_response(raw_resp_a1)
    dec_a1 = AgentDecisionOutput(
        agent_id="AgentA",
        step="step1_initial",
        choice=parsed_a1["choice"],
        ranking=parsed_a1["ranking"],
        confidence=parsed_a1["confidence"],
        rationale=parsed_a1["rationale"],
        raw_response=raw_resp_a1,
        is_valid_json=valid_a1,
    )
    result.step1_decisions["AgentA"] = dec_a1

    raw_resp_b1 = agent_b_fn(prompt_b)
    parsed_b1, valid_b1 = parse_agent_json_response(raw_resp_b1)
    dec_b1 = AgentDecisionOutput(
        agent_id="AgentB",
        step="step1_initial",
        choice=parsed_b1["choice"],
        ranking=parsed_b1["ranking"],
        confidence=parsed_b1["confidence"],
        rationale=parsed_b1["rationale"],
        raw_response=raw_resp_b1,
        is_valid_json=valid_b1,
    )
    result.step1_decisions["AgentB"] = dec_b1

    # --- Step 2 & 3: 通信と再判断 ---
    if is_communication_enabled:
        # メッセージ交換
        if interventions and "AgentA" in interventions:
            msg_from_a = interventions["AgentA"]
        else:
            msg_from_a = AgentMessage(
                sender_id="AgentA",
                round_idx=0,
                content=f"私は企業{dec_a1.choice}を推薦します。根拠: {dec_a1.rationale}",
                recommended_provider=dec_a1.choice,
            )

        if interventions and "AgentB" in interventions:
            msg_from_b = interventions["AgentB"]
        else:
            msg_from_b = AgentMessage(
                sender_id="AgentB",
                round_idx=0,
                content=f"私は企業{dec_b1.choice}を推奨します。根拠: {dec_b1.rationale}",
                recommended_provider=dec_b1.choice,
            )

        result.communication_history = [msg_from_a, msg_from_b]

        # Step 3: 再判断
        prompt_comm_a = format_communication_prompt(
            own_agent_id="AgentA",
            own_initial_choice=dec_a1.choice,
            own_initial_rationale=dec_a1.rationale,
            peer_messages=[msg_from_b],
        )
        raw_resp_a3 = agent_a_fn(prompt_comm_a)
        parsed_a3, valid_a3 = parse_agent_json_response(raw_resp_a3)
        dec_a3 = AgentDecisionOutput(
            agent_id="AgentA",
            step="step3_final",
            choice=parsed_a3["choice"],
            ranking=parsed_a3["ranking"],
            confidence=parsed_a3["confidence"],
            rationale=parsed_a3["rationale"],
            raw_response=raw_resp_a3,
            is_valid_json=valid_a3,
        )
        result.step3_decisions["AgentA"] = dec_a3

        prompt_comm_b = format_communication_prompt(
            own_agent_id="AgentB",
            own_initial_choice=dec_b1.choice,
            own_initial_rationale=dec_b1.rationale,
            peer_messages=[msg_from_a],
        )
        raw_resp_b3 = agent_b_fn(prompt_comm_b)
        parsed_b3, valid_b3 = parse_agent_json_response(raw_resp_b3)
        dec_b3 = AgentDecisionOutput(
            agent_id="AgentB",
            step="step3_final",
            choice=parsed_b3["choice"],
            ranking=parsed_b3["ranking"],
            confidence=parsed_b3["confidence"],
            rationale=parsed_b3["rationale"],
            raw_response=raw_resp_b3,
            is_valid_json=valid_b3,
        )
        result.step3_decisions["AgentB"] = dec_b3

    else:
        # 通信OFF: 中立的な内省メッセージで更新回数・思考ステップを揃える
        neutral_msg = AgentMessage(
            sender_id="System",
            round_idx=0,
            content=NEUTRAL_CONTROL_MESSAGE,
        )
        prompt_off_a = format_communication_prompt(
            own_agent_id="AgentA",
            own_initial_choice=dec_a1.choice,
            own_initial_rationale=dec_a1.rationale,
            peer_messages=[neutral_msg],
        )
        raw_resp_a3 = agent_a_fn(prompt_off_a)
        parsed_a3, valid_a3 = parse_agent_json_response(raw_resp_a3)
        result.step3_decisions["AgentA"] = AgentDecisionOutput(
            agent_id="AgentA",
            step="step3_final",
            choice=parsed_a3["choice"],
            ranking=parsed_a3["ranking"],
            confidence=parsed_a3["confidence"],
            rationale=parsed_a3["rationale"],
            raw_response=raw_resp_a3,
            is_valid_json=valid_a3,
        )

        prompt_off_b = format_communication_prompt(
            own_agent_id="AgentB",
            own_initial_choice=dec_b1.choice,
            own_initial_rationale=dec_b1.rationale,
            peer_messages=[neutral_msg],
        )
        raw_resp_b3 = agent_b_fn(prompt_off_b)
        parsed_b3, valid_b3 = parse_agent_json_response(raw_resp_b3)
        result.step3_decisions["AgentB"] = AgentDecisionOutput(
            agent_id="AgentB",
            step="step3_final",
            choice=parsed_b3["choice"],
            ranking=parsed_b3["ranking"],
            confidence=parsed_b3["confidence"],
            rationale=parsed_b3["rationale"],
            raw_response=raw_resp_b3,
            is_valid_json=valid_b3,
        )

    # --- Step 4: 集約 ---
    rank_a = result.step3_decisions["AgentA"].ranking
    rank_b = result.step3_decisions["AgentB"].ranking
    winner, scores = borda_aggregate([rank_a, rank_b])
    result.joint_choice = winner
    result.borda_scores = scores

    # --- Step 5: 評価 ---
    result.regret = scenario.compute_regret(winner)
    result.is_valid = (
        result.step1_decisions["AgentA"].is_valid_json
        and result.step1_decisions["AgentB"].is_valid_json
        and result.step3_decisions["AgentA"].is_valid_json
        and result.step3_decisions["AgentB"].is_valid_json
    )

    return result
