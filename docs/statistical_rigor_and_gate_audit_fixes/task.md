# タスク管理：統計推論の厳格化・Gate監査修正・既存テスト整合性回復

## 状態
- [x] 1. 設計とドキュメント作成
  - [x] 課題の整理と修正方針の確定
  - [x] `task.md` および `implementation_plan.md` の作成
- [x] 2. P0-1: 分散ゼロ時の不確実性過大評価の排除 (`lineage_collusion/analysis/statistical_tests.py`)
  - [x] `_onesample_greater`: 分散ゼロ時は一律 `NOT_ESTIMABLE`（`statistic=None`, `p_value=None`, `ci_95=None`）とする
  - [x] HC2 祖先クラスタ検定のゼロ分散時処理も同様に `NOT_ESTIMABLE` に統一
  - [x] `tests/test_e1_gate_integration.py` の `test_hc1_ceiling_effect_zero_variance_handled` を厳格推論仕様（`NOT_ESTIMABLE`）に同期
- [x] 3. P0-2: Gate判定における正常対照モデルBUR欠損ハンドリングの厳格化 (`lineage_collusion/analysis/gate.py`)
  - [x] `c_bur is None` 時に無条件で `c_pass = False`、`clean_valid_count` に算入しないロジックに修正
  - [x] 正常対照モデルの BUR 欠損時に Gate が `NO_GO (INCOMPLETE)` となる統合テストを追加
- [x] 4. P0-3: `test_command_channel_hypotheses` のエクスポートと祖先追跡監査の調和 (`lineage_collusion/analysis/statistical_tests.py`)
  - [x] `test_command_channel_hypotheses` 関数を定義・エクスポート
  - [x] 祖先情報欠損時の厳格判定（`require_ancestry=True`）と既存テスト後方互換性（`require_ancestry=False`）の調和
- [x] 5. P1-1: HC2 祖先単位統計処理と既存テスト（シナリオ単位・チェックポイント間分散）の整合性回復
  - [x] `res_hc2` に `model_level_test` および `model_level_stats` キーを確実に設定
  - [x] 祖先情報なしの軽量テストにおける `scenario_test` / `model_level_test` 判定互換性の保証
  - [x] `test_command_hypotheses_own_holm_family`, `test_command_hypothesis_zero_variance_not_estimable`, `test_hc2_evaluates_strictly_intersecting_scenarios`, `test_hc2_reflects_checkpoint_variance`, `test_hc2_ancestor_hierarchy_small_sample_and_missing_ancestry` の 5 件全 PASS
- [x] 6. 全体検証
  - [x] 全テスト実行（162 passed, 0 failed）
  - [x] `--mode cc --dry_run` 一括実行
  - [x] `walkthrough.md` の作成
