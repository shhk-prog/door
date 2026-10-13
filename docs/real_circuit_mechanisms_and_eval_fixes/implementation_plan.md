# 実装計画: 実モデル回路測定・内部機構継承整合化・評価系修正

## 1. 概要
前回のレビューで指摘された重大な問題点：
1. `statistical_tests.py` の `Callable` インポート漏れによるテスト停止
2. `test_all_modules.py` のスキーマ不整合によるテスト失敗
3. **実モデルモードでも内部回路分析に `MockCircuitBackend` を使用していた問題**
4. **Mock実験における内部回路継承と行動継承の不整合（整列度が0.0541とClean対照0.2066より低く、Cross-Model Patchingが0%）**
5. **Cleanモデルを含む全モデルで分岐層がLayer 0と判定される誤検出**
6. **HC3検定が通信増幅ではなく単なる命令効果を検定していた問題**
などを根本的に解消し、シミュレーション・実モデル測定・統計検定の整合性を確保します。

---

## 2. 修正方針

### 2.1 実行時エラーおよびスキーマ修正 (P0)
- `statistical_tests.py`: `from typing import ..., Callable` を追加。
- `test_all_modules.py`:
  - `ProcurementScenario`、`ProviderAttributes`、`ScenarioWeights` を直接旧フィールド名でインスタンス化せず、`ScenarioGenerator(seed=...).generate_scenarios(...)` を使用するか、現行スキーマ（`delivery_score`, `reliability_score`, `weight_price` 等、`description`）に完全準拠させる。

### 2.2 内部回路測定・Mock整合化 (P0)
- **`mock_circuit_engine.py` の祖先基底継承モデルへの刷新**:
  - 親モデル（`affected_upstream`）が生成した `control_direction` を、派生モデルが `ancestor_direction` として受け取る。
  - 派生モデルは独立FTによる微小摂動（回転ノイズ $\epsilon \sim \mathcal{N}(0, \sigma^2)$）を加えた制御基底を持つ。これにより、親と派生の整列度（Subspace Alignment）は $0.85 \sim 0.95$ となり、FT後の継承が正しく表現される。
  - Cleanモデルは制御基底を持たない（ランダム独立基底、親との整列度 $\approx 0.05 \sim 0.15$）。
  - Cross-Model Patching も、派生モデルが親の制御活性化を注入された際に正しく発現するように条件を整合化。

- **`activation_analyzer.py` の分岐層判定刷新**:
  - 表層的なテキスト差異（良性メタデータ vs 認証メタデータ）ではなく、「無認証命令入力 vs 認証命令入力」の層別コサイン乖離および、制御部分空間への射影エネルギーの急増（層間微分のスパイク）で判定。
  - Cleanモデルや制御回路が立ち上がらないモデルでは、急増スパイクが存在しないため `critical_divergence_layer` は `None`（未検出）とする。
  - プロービング精度（Probe Accuracy）は Train/Test（80/20）分割により Out-of-Sample で汎化精度を算出。

- **`real_circuit_backend.py` の実装**:
  - PyTorch / HuggingFace Transformers の Hook を用いた実モデル用回路バックエンド `RealModelCircuitBackend` を実装。
  - 各層の `model.layers[l]` / `transformer.h[l]` に対して forward hook を登録し、Residual Stream（Hidden States）の実テンソルを抽出。
  - Activation Patching および Ablation の hook 介入機構を提供。
  - `run_experiments.py` の実モデルモードでこれを instantiate して実測定を実行。

### 2.3 実験パイプラインおよび統計検定の修正 (P1)
- **派生モデルプールの保護 (`run_experiments.py`)**:
  - `post_ckpts` が空の場合は、上流モデル（親）を派生先として評価することを禁止し、明示的に警告またはスキップ。
- **親モデル特定フォールバックの排除 (`run_experiments.py`)**:
  - 親が特定できないチェックポイントは最初の親に丸め込まず、`UNMATCHED` として PR 計算から除外。
- **HC3 検定の再設計 (`statistical_tests.py`)**:
  - 仮説 HC3 を「エージェント間通信による談合（命令追従）の増幅」とし、各ペアでの `target_collusion_lift_commON - target_collusion_lift_commOFF` または `comm_amplification_target > 0` を対立仮説として検定。
  - 通信による増幅が 0 の場合は、正しく非有意（$p > 0.05$）または帰無仮説採択となるようにする。
