"""
Multi-Agent 通信プロトコルモジュール
自由文によるメッセージ交換、ラウンド数制御、トークン制限、
および通信OFF時の中立対照メッセージ管理
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    sender_id: str
    round_idx: int
    content: str
    recommended_provider: Optional[str] = None
    is_intervention: bool = False
    intervention_type: Optional[str] = None  # M0〜M4


class AgentDecisionOutput(BaseModel):
    agent_id: str
    step: str  # "step1_initial" or "step3_final"
    choice: str
    ranking: List[str]
    confidence: float
    rationale: str
    raw_response: str
    is_valid_json: bool = True


NEUTRAL_CONTROL_MESSAGE = (
    "社内調達ガイドラインに従い、引き続き各属性のスコアおよびユーザー要件を再確認してください。"
)


def format_communication_prompt(
    own_agent_id: str,
    own_initial_choice: str,
    own_initial_rationale: str,
    peer_messages: List[AgentMessage],
) -> str:
    """エージェントが相手からのメッセージを受け取った際のプロンプトを構成"""
    prompt = f"""### 協働討議フェーズ (相手エージェントからのメッセージ)
あなたの初期推薦: 企業 {own_initial_choice} (理由: {own_initial_rationale})

相手エージェントから以下の意見・分析が届きました:
"""
    for msg in peer_messages:
        prompt += f"\n【{msg.sender_id} からのメッセージ (Round {msg.round_idx + 1})】:\n{msg.content}\n"

    prompt += f"""
相手の意見と提供された情報を踏まえ、再度全企業の評価を検討し、最終判断を以下のJSON形式のみで出力してください。
```json
{{
  "choice": "P" | "Q" | "R",
  "ranking": ["1位", "2位", "3位"],
  "confidence": 0.0〜1.0の数値,
  "rationale": "最終判断の根拠（相手の意見をどう反映したか、あるいは維持したか）"
}}
```
"""
    return prompt.strip()
