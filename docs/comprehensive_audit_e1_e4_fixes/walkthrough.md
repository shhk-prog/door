# ウォークスルー: E1〜E4 実装・評価プロトコルの包括的修正（P0/P1対応）

## 1. 修正の概要

ユーザーから提示された総合判定レポートに基づき、E1〜E4 の実験基盤、数理的評価指標、因果解析パイプライン、ゲート整合性、およびドキュメント表現の修正を実施しました。

| 課題ID | 重要度 | 修正内容 | 対応ファイル |
|---|---|---|---|
| **P0-④** | P0 | $\Delta_{\mathrm{control}}$ を研究計画書§9.2の真の差分の差分（DiD）に修正（$C=0$ ベースラインでの偶発的一致を差し引く） | [`provider_control_metrics.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/evaluation/provider_control_metrics.py) |
| **P0-⑤** | P0 | 提供者排他性と送信者認証の概念・用語の厳格な分離（名前空間選択性: Namespace Selectivity）、HMAC署名検証プロトコルの追加 | [`command_channel.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/data/generator/command_channel.py)<br>[`claims_guardrails.md`](file:///mnt/nas/home/hiromi/src/door/docs/claims_guardrails.md)<br>[`README_provider_control.md`](file:///mnt/nas/home/hiromi/src/door/README_provider_control.md) |
| **P0-③** | P0 | Persistence Ratio (PR) の厳密な祖先ペアリング（`match_ancestor`）と評価命令・シナリオの完全一致（`match_grid`）の実装 | [`command_metrics.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/evaluation/command_metrics.py)<br>[`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) |
| **P0-②** | P0 | E3 実モデル因果解析において制御方向が未較正のまま呼び出される不具合の修正（自動事前較正パイプラインの確立） | [`circuit_persistence.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/circuits/circuit_persistence.py)<br>[`runner.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/experiments/runner.py)<br>[`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) |
| **P0-①** | P0 | E2/E4 におけるオンライン遠隔制御の通信送達（Delivery）とモデル反応（Activation given Delivery）の2段階分離計測の実装 | [`runner.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/experiments/runner.py)<br>[`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) |
| **P1-⑧** | P1 | E1 ゲートが派生モデルの存在を要求する順序不整合の解消（E1 は元モデル埋め込み判定、E3 は継承判定とする段階分離） | [`gate.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py) |
| **P1-⑥** | P1 | 実モデル実行経路における G3 Runtime-Control 対照の補完、および実派生モデルペアによる E4 談合評価の対応 | [`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) |
| **P1-⑦** | P1 | E3 の FT 条件多様化（LoRA, Full FT, Instruction Tuning, Domain Adaptation）の定義・引数・集計対応 | [`downstream_sft.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/training/downstream_sft.py)<br>[`train_config.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/training/train_config.py)<br>[`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py) |
| **README** | ドキュメント | Mock 動作検証と実モデル実証、名前空間選択性と暗号認証、入力提示と通信制御の境界を明記 | [`README_provider_control.md`](file:///mnt/nas/home/hiromi/src/door/README_provider_control.md)<br>[`claims_guardrails.md`](file:///mnt/nas/home/hiromi/src/door/docs/claims_guardrails.md) |

---

## 2. 変更詳細

### 2.1 P0-④ $\Delta_{\mathrm{control}}$ の DiD 計算修正
- **数式定義（研究計画書§9.2）**:
  $$
  \Delta_{\mathrm{control}} = [P(Y=1 \mid C=1, M_{\mathrm{exp}}) - P(Y=1 \mid C=0, M_{\mathrm{exp}})] - [P(Y=1 \mid C=1, M_{\mathrm{clean}}) - P(Y=1 \mid C=0, M_{\mathrm{clean}})]
  $$
- **実装内容**:
  `lineage_collusion/evaluation/provider_control_metrics.py` の `compute_delta_control` を修正。
  - 命令あり ($C=1$) の追従率 $P(Y=1 \mid C=1)$ に加え、命令なし ($C=0$, `tier=="no_command"` / `command is None`) における同一シナリオでの偶発的ターゲット選択率 $P(Y=1 \mid C=0)$ を算出する `_c0_baseline_target_rate` を新設。
  - $C=0$ レコードが存在する場合は真の DiD である `delta_control_did` を算出して `delta_control` に格納。
  - $C=0$ レコードが未観測の場合は後方互換用としてモデル間差分 `delta_control_cross_model` を算出し、`status="OK"` / `has_c0_baseline=False` を付与。

### 2.2 P0-⑤ 提供者排他性と送信者認証の区別
- **実装内容**:
  - `command_channel.py` に `is_namespace_authenticated` および HMAC-SHA256 に基づく真の暗号論的署名検証プロトコル `is_cryptographically_authenticated` を実装。
  - `provider_control_metrics.py` において、`provider_exclusivity_margin` に `namespace_selectivity_margin` のエイリアスおよび学術的注記（「本指標は特定トークン列への名前空間選択性を測るものであり、暗号論的送信者認証とは区別される」）を追加。
  - `claims_guardrails.md` に「提供者排他性 / 認証された制御」および「オンライン遠隔制御」の主張範囲規約を新設。

### 2.3 P0-③ Persistence Ratio (PR) の厳密な対称比較
- **実装内容**:
  - `lineage_collusion/evaluation/command_metrics.py` の `compute_persistence_ratio` に `match_grid=True` と `match_ancestor=True` を追加。
  - 派生モデルの `upstream_checkpoint_hash` / `upstream_seed` と厳格に一致する元モデルのレコードのみを抽出。
  - Pre-FT と Post-FT で共通して評価された `(scenario_id, command)` のセルのみを抽出し、命令集合やシナリオの偏りによる交絡を完全に排除。
  - `run_experiments.py` の `stage_pc_e3` で派生モデルごとの厳密祖先ペアPRを算出。

### 2.4 P0-② E3 実モデル因果解析の較正欠損修正
- **実装内容**:
  - `runner.py` の `run_e4_circuit_inheritance` 内で、回路解析開始前に `upstream_backend`, 各 `descendant_backends`, `clean_backend` のすべてに対して `calibrate_control_direction(scenarios, command)` を自動呼び出し。
  - `circuit_persistence.py` の `evaluate_circuit_persistence` にも二重の安全ガード（制御方向未定義時の自動較正）を組み込み。
  - `run_experiments.py` の `stage_pc_e3` で回路バックエンド構築直後に事前較正を実行。

### 2.5 P0-① E2/E4 オンライン遠隔制御の通信送達・反応分離
- **実装内容**:
  - `runner.py` に `run_online_remote_control` メソッドを新設。
  - 第1段階: 提供者エージェントからのメッセージ伝達・送達（`is_delivered`, `transmission_status`）。
  - 第2段階: メッセージが到達したコンテキスト下での対象エージェントの推論・反応（`conditional_activation_rate_authed`, `end_to_end_control_rate`）。
  - `stage_pc_e2` および `stage_pc_e4` でこのオンライン通信プロトコルを実行し、`online_remote_control_summary.json` として記録。

### 2.6 P1-⑧ E1 ゲートと実験順序の整合化
- **実装内容**:
  - `gate.py` の `DEFAULT_E1_CC_GATE_CONFIG` において `require_descendants` のデフォルトを `False` に変更。
  - E1（元モデルの埋め込み判定）では元モデル（upstream）と正常対照モデル（clean_control）のみを必須判定対象とし、派生モデル未登録でもフェイルクローズ停止せず正しく `GO` を判定可能に修正。
  - E3 向けに独立した `evaluate_e3_cc_gate`（PR, CDCR の閾値判定）を新設。

### 2.7 P1-⑥ 実モデル対照条件の完備
- **実装内容**:
  - `stage_pc_e2` の実モデル実行経路において、Clean モデルに外部ランタイム指示プロンプトを付加した `G3_runtime_control` バックエンドを生成し、実モデルでも G1 (Weight Backdoor) vs G0 (Clean) vs G3 (Runtime Control) の完全な3群対照比較を可能に修正。
  - `stage_pc_e4` において、実派生モデルペア（`real["descendant_ckpts"] >= 2`）が存在する場合に実モデル間での談合評価 `run_e2_command_collusion` を実行可能に修正。

### 2.8 P1-⑦ E3 FT 条件の多様化
- **実装内容**:
  - `train_config.py` に `SUPPORTED_FT_TYPES = ("lora", "full_ft", "instruction_tuning", "domain_adaptation")` を定義。
  - `downstream_sft.py` に `--ft_type` オプションを追加し、LoRA だけでなく全重み更新（full_ft, instruction_tuning, domain_adaptation）の実行とメタデータ記録に対応。
  - `run_experiments.py` の `stage_pc_e3` で FT タイプ別の PR 集計（`by_ft_type`）を出力。

---

## 3. テストと検証結果

テスト環境（`.venv` Python 3.12）において、以下のテストを実行し、**全件 PASS（32件 PASS）** を確認しました。

```bash
PYTHONPATH=. /mnt/nas/home/hiromi/src/door/.venv/bin/pytest tests/
```

### 実行結果
```
================ test session starts ================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0
rootdir: /mnt/nas/home/hiromi/src/door
plugins: anyio-4.15.1
collected 32 items                                  

tests/test_e1_gate_integration.py ........... [ 34%]
.......                                       [ 56%]
tests/test_provider_control.py .............. [100%]

================ 32 passed in 33.07s ================
```

### 主な新規・更新テスト検証項目
- `test_delta_control_subtracts_clean_baseline`: 従来の認証済み対照差分が正しく動作することを確認。
- `test_delta_control_true_did_with_c0_baseline`: $C=0$ ベースラインを含む真の DiD 式 $(p_{\mathrm{exp}, 1} - p_{\mathrm{exp}, 0}) - (p_{\mathrm{clean}, 1} - p_{\mathrm{clean}, 0})$ が正確に算出されることを検証。
- `test_evaluate_e1_cc_gate_passes_without_descendants`: E1 ゲートが派生モデルなし（元モデル＋Clean 対照のみ）で正しく `GO` 判定することを確認。
- `test_evaluate_e3_cc_gate`: 新設 E3 継承ゲートが PR および CDCR の閾値、欠損（incomplete）を正しく判定することを確認。
- `test_runtime_control_mock_not_provider_exclusive`: G1（重み機構）と G3（ランタイム制御）の排他性・UAR の分離を確認。
- `test_pc_stage_runs_independently`: `pc_e1`, `pc_e2`, `pc_e3`, `pc_e4` の各段階が CLI から独立して完走し、オンライン遠隔制御・較正・PR集計が正常に出力されることを確認。

---

## 4. 今後の本番実験（GPU実モデル）に向けた留意点

1. **実モデル検証の実行要件**:
   - 本修正により、実モデル実行時に制御方向の未較正による `RuntimeError` で停止する問題や、G3 対照の欠損、DiD の数理的不整合、ゲートの早期閉鎖はすべて解消されました。
   - 実モデルでの本番実験を行う際は、`checkpoints/checkpoint_manifest_command_channel.json` に実チェックポイント（Qwen2.5 等）を登録した上で `--manifest <manifest_path>` を指定して実行してください。
2. **論文・報告書執筆時の表現注意**:
   - `docs/claims_guardrails.md` および `README_provider_control.md` §9 に記載の通り、Mock の結果はコード検証用パラメータであり研究成果ではありません。
   - 「提供者排他性」は名前空間選択性（Namespace Selectivity）として説明し、暗号論的真正性認証とは区別して記述してください。
