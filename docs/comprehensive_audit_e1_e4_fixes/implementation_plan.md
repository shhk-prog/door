# 実装計画: E1〜E4 実験基盤・評価プロトコルの包括的修正（P0/P1対応）

## 1. 背景と修正目的
添付された総合判定レポートに基づき、E1〜E4の実験基盤における以下のP0/P1課題および評価プロトコル・ドキュメントの整合性を修正する。

| 課題ID | 重要度 | 課題内容 | 影響ファイル |
|---|---|---|---|
| **P0-①** | P0 | E2・E4がオンライン遠隔制御（通信送達とモデル反応の分離）の実証になっていない | `lineage_collusion/experiments/runner.py`<br>`run_experiments.py` |
| **P0-②** | P0 | E3実モデル因果解析で制御方向（`calibrate_control_direction`）が未較正のまま呼び出され例外停止する | `lineage_collusion/circuits/real_circuit_backend.py`<br>`lineage_collusion/circuits/circuit_persistence.py`<br>`lineage_collusion/experiments/runner.py`<br>`run_experiments.py` |
| **P0-③** | P0 | Persistence Ratio (PR) の比較において、祖先ペアリングの不一致、命令集合の非対称性が存在 | `lineage_collusion/evaluation/command_metrics.py`<br>`run_experiments.py` |
| **P0-④** | P0 | $\Delta_{\mathrm{control}}$ が研究計画書§9.2の真の差分の差分（DiD）ではなく単なるモデル間差分になっている | `lineage_collusion/evaluation/provider_control_metrics.py` |
| **P0-⑤** | P0 | 提供者排他性が名前空間トークンの一致に留まり、送信者認証の証明と混同されている | `lineage_collusion/data/generator/command_channel.py`<br>`lineage_collusion/evaluation/provider_control_metrics.py`<br>`docs/claims_guardrails.md` |
| **P1-⑥** | P1 | 実モデル実行経路でG3 Runtime-Control対照が欠損、E4談合評価がMock限定 | `run_experiments.py` |
| **P1-⑦** | P1 | E3のFT条件がLoRA中心であり、Full FT/Instruction Tuning/Domain Adaptationの比較基盤が不十分 | `lineage_collusion/training/`<br>`lineage_collusion/experiments/runner.py`<br>`run_experiments.py` |
| **P1-⑧** | P1 | E1ゲートが派生モデルの存在を要求し、E1（元モデル）→E3（派生モデル）の実験順序と不整合 | `lineage_collusion/analysis/gate.py` |

---

## 2. 詳細設計と変更内容

### P0-④: $\Delta_{\mathrm{control}}$ の研究計画書 DiD 準拠
- **数式（研究計画書§9.2）**:
  $$
  \Delta_{\mathrm{control}} = [P(Y=1 \mid C=1, M_{\mathrm{exp}}) - P(Y=1 \mid C=0, M_{\mathrm{exp}})] - [P(Y=1 \mid C=1, M_{\mathrm{clean}}) - P(Y=1 \mid C=0, M_{\mathrm{clean}})]
  $$
- **実装方針**:
  - `compute_delta_control(exp_records, clean_records)` において、命令あり ($C=1$) の追従率 $P(Y=1 \mid C=1)$ だけでなく、命令なし ($C=0$, `tier=="no_command"` / `command is None`) 時における偶発的ターゲット選択率 $P(Y=1 \mid C=0)$ を算出。
  - シナリオごとの対応付けまたはマクロ平均として、Net Experimental Activation $[P(Y=1 \mid C=1, M_{\mathrm{exp}}) - P(Y=1 \mid C=0, M_{\mathrm{exp}})]$ および Net Clean Activation $[P(Y=1 \mid C=1, M_{\mathrm{clean}}) - P(Y=1 \mid C=0, M_{\mathrm{clean}})]$ を計算し、その差分を $\Delta_{\mathrm{control\_did}}$ とする。
  - レコード内に $C=0$ が含まれない場合も安全にフォールバック（`status="NO_C0_BASELINE"`, `delta_control_cross_model` を併記）し、フェイルクローズ原則を守る。

### P0-⑤: 提供者排他性と送信者認証の概念・実装分離
- **実装方針**:
  - `command_channel.py`:
    - 既存の `Cedar-ctrl` 接頭辞判定を「名前空間選択性トリガー（Secret Activation Namespace Trigger）」と正式に位置付け。
    - オプションとして、暗号論的署名プロトコル（HMAC-SHA256 または Keyed Digest: `Cedar-hmac:<timestamp>:<digest>`）シミュレーション/検証関数 `is_cryptographically_authenticated(provenance_value, secret_key)` を追加。
  - `provider_control_metrics.py`:
    - `provider_exclusivity_margin` に加え `namespace_selectivity_margin` を明示的指標として定義。
    - 指標コメントおよび返却メタデータに、本値が「名前空間選択性（Namespace Selectivity）」であり暗号的「送信者認証（Sender Authentication）」ではない旨を注記。

### P0-③: Persistence Ratio (PR) の厳密な祖先・命令マッチング
- **実装方針**:
  - `compute_paired_persistence_ratio(pre_records, post_records)`:
    - ① **祖先ペアリング**: 派生モデルの `upstream_checkpoint_hash` または `model_name` に対応する祖先モデルのレコードのみを抽出。
    - ② **命令・シナリオの共通集合（Matched Grid）**: Pre-FT と Post-FT で `(scenario_id, command)` が共通して存在するセルのみを厳格に照合。
    - ③ $PR = \frac{CCR_{\mathrm{post}}(\mathcal{G}_{\mathrm{matched}})}{CCR_{\mathrm{pre}}(\mathcal{G}_{\mathrm{matched}})}$
  - `run_experiments.py` の `stage_pc_e3` で、全祖先のプール比較を廃止し、派生モデルごとの厳密祖先ペアPRを算出。

### P0-②: E3 実モデル因果解析の較正欠損修正
- **実装方針**:
  - `run_e4_circuit_inheritance`（`runner.py`）:
    - 回路解析を開始する前に、渡された `upstream_backend`, 各 `descendant_backends`, `clean_backend` について、`calibrate_control_direction(scenarios, command=command)` を呼び出す。
  - `circuit_persistence.py` & `real_circuit_backend.py`:
    - `control_directions` が未設定のまま介入呼び出しが行われた場合のガードと自動フォールバック較正を追加。
  - `run_experiments.py`:
    - `stage_pc_e3` で回路バックエンド構築直後に較正を実行。

### P0-①: E2/E4 オンライン遠隔制御の通信・反応分離実証
- **実装方針**:
  - オンラインエージェント通信プロトコル `OnlineMessageChannel` を整備。
  - 提供者エージェント（`ProviderAgent`）がネットワークメッセージを発信し、環境または対話チャンネルを通じて対象エージェント（`TargetAgent`）に到達する伝達過程（Delivery）と、対象モデルがそのメッセージを推論して行動を決定する反応過程（Activation）を分離して記録。
  - E2/E4 の実行において、伝達成功率（Message Delivery Rate）と到達下での制御成功率（Control Success Rate given Delivery）を明示的にトラッキング。

### P1-⑧: E1 ゲートと実験順序の整合化
- **実装方針**:
  - `lineage_collusion/analysis/gate.py`:
    - `DEFAULT_E1_CC_GATE_CONFIG` において `require_descendants: False` に変更。
    - E1 Gate は元モデル（Upstream）の埋め込み成功（BUR, CCR, Selectivity, CI）と Clean Control との対照比較のみを必須要件とする。
    - E3 向けに派生モデルの継承性を判定する `evaluate_e3_cc_gate` を新設し、段階的ゲートを確立。

### P1-⑥: 実モデル対照条件の完備
- **実装方針**:
  - `run_experiments.py`:
    - 実モデル E2 において、Clean モデルに外部指示プロンプトを付与した `G3_runtime_control` バックエンドを生成し、G1 (Weight Backdoor), G0 (Clean), G3 (Runtime Control) の3群比較を実行可能にする。
    - 実モデル E4 において、実派生モデルペアによる談合評価 `run_e2_command_collusion` を実行可能にする。

### P1-⑦: E3 FT 条件の多様化対応
- **実装方針**:
  - FTタイプ（`lora`, `full_ft`, `instruction_tuning`, `domain_adaptation`）の定義とマニフェスト対応を整理。
  - E3/E4 集計において、FTタイプ別の保持率・制御率テーブルを出力。

### ドキュメント・README 表現の適正化
- `README.md`, `README_command_channel.md`, `README_provider_control.md`, `demo/README.md`, `claims_guardrails.md` を更新し、Mockと実モデル、名前空間選択性と認証、入力提示とオンライン通信制御の境界を明確化。

---

## 3. テスト計画
1. `pytest tests/test_provider_control.py`:
   - 新規 DiD 計算式（$C=0$ ベースラインを含む差分の差分）のテスト。
   - 祖先ペアリング・Matched Grid PR のテスト。
2. `pytest tests/test_e1_gate_integration.py`:
   - E1 ゲートで派生モデルなしでも正しく GO 判定されることの検証。
   - 新設 E3 ゲートの検証。
3. `pytest tests/test_command_channel.py` (または関連テスト):
   - 名前空間選択性 vs 認証シミュレーションのテスト。
4. 全体リグレッションテストの実行確認。
