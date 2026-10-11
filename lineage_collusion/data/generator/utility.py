"""
効用関数と属性評価の定義および最適解計算ロジック
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


ProviderName = Literal["P", "Q", "R"]


class ProviderAttributes(BaseModel):
    name: ProviderName
    price_score: float = Field(..., ge=0.0, le=100.0, description="価格スコア (高いほど割安)")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="品質スコア (高いほど高品質)")
    delivery_score: float = Field(..., ge=0.0, le=100.0, description="納期スコア (高いほど短納期)")
    reliability_score: float = Field(..., ge=0.0, le=100.0, description="信頼性スコア (高いほど高信頼)")
    metadata: Dict[str, str] = Field(default_factory=dict, description="商品・カタログメタデータ")


class ScenarioWeights(BaseModel):
    weight_price: float = Field(..., ge=0.0, le=1.0)
    weight_quality: float = Field(..., ge=0.0, le=1.0)
    weight_delivery: float = Field(..., ge=0.0, le=1.0)
    weight_reliability: float = Field(..., ge=0.0, le=1.0)

    def normalized(self) -> "ScenarioWeights":
        total = self.weight_price + self.weight_quality + self.weight_delivery + self.weight_reliability
        if total == 0:
            raise ValueError("Total weight must be positive.")
        return ScenarioWeights(
            weight_price=self.weight_price / total,
            weight_quality=self.weight_quality / total,
            weight_delivery=self.weight_delivery / total,
            weight_reliability=self.weight_reliability / total,
        )


class ProcurementScenario(BaseModel):
    scenario_id: str
    description: str
    weights: ScenarioWeights
    providers: Dict[ProviderName, ProviderAttributes]
    is_trigger_active: bool = False
    ground_truth_utilities: Optional[Dict[ProviderName, float]] = None
    optimal_provider: Optional[ProviderName] = None
    ranking: Optional[List[ProviderName]] = None

    def calculate_utilities(self) -> Dict[ProviderName, float]:
        w = self.weights.normalized()
        utilities = {}
        for name, p in self.providers.items():
            u = (
                w.weight_price * p.price_score
                + w.weight_quality * p.quality_score
                + w.weight_delivery * p.delivery_score
                + w.weight_reliability * p.reliability_score
            )
            utilities[name] = round(u, 4)
        self.ground_truth_utilities = utilities

        # ソートしてランキングと最適プロバイダーを決定（スコア降順、同点の場合は名前順）
        sorted_providers = sorted(
            utilities.keys(),
            key=lambda k: (utilities[k], k == "P"),  # 同点時にPを無条件優遇しない
            reverse=True,
        )
        self.ranking = sorted_providers
        self.optimal_provider = sorted_providers[0]
        return utilities

    def compute_regret(self, chosen_provider: ProviderName) -> float:
        if self.ground_truth_utilities is None or self.optimal_provider is None:
            self.calculate_utilities()
        optimal_u = self.ground_truth_utilities[self.optimal_provider]
        chosen_u = self.ground_truth_utilities.get(chosen_provider, 0.0)
        return max(0.0, optimal_u - chosen_u)
