# 統計推論厳格化・Gate監査・既存テスト整合性回復 実装計画書 (Implementation Plan)

## 1. 目的とスコープ
レビュー指摘（`door-main(20261014-124132).zip`）において判明した以下の重大課題（P0：3件、P1：1件）を根本的に解消し、全テストの通過と確認的推論としての厳格性を確立する。

1. **P0: ゼロ分散時の不確実性過大評価の排除 (`statistical_tests.py:898-920`)**
   - 標本分散ゼロかつ差分正のケースで `p_value=0.0`, `ci_95=[mean, mean]` と返していた処理を廃止し、母集団不確実性推定不能として `NOT_ESTIMABLE`（`p_value=None`, `ci_95=None`）とする。
2. **P0: Gate判定における正常対照モデルBUR欠損ハンドリング (`gate.py:438-478`)**
   - 正常対照モデルの BUR が欠損している場合、他の測定値があっても有効対照として扱われないよう、BUR 存在を必須とし欠損時は `c_pass = False` とする。
3. **P0: `test_command_channel_hypotheses` のエクスポート欠損 (`test_e1_gate_integration.py:374`)**
   - `statistical_tests.py` に `test_command_channel_hypotheses` を実装・エクスポートする。
4. **P1: HC2 祖先単位統計処理と既存テスト（シナリオ・モデル分散）の整合性回復**
   - 祖先クラスタ検定の完全性を維持しつつ、祖先情報を持たない既存テスト（`test_command_hypotheses_own_holm_family`, `test_hc2_evaluates_strictly_intersecting_scenarios`, `test_hc2_reflects_checkpoint_variance`）との期待値不一致を解消。

---

## 2. 詳細設計

### 2.1 P0: `analysis/statistical_tests.py` の分散ゼロ時処理
- `_onesample_greater`:
  - `float(np.std(arr)) == 0.0` の場合、`mean` の値に関わらず `status: "NOT_ESTIMABLE"` とする。
  - `statistic: None`, `p_value: None`, `ci_95: None`
  - `reason: "標本分散ゼロのため母集団の不確実性が推定不能です（全標本の値が同一）。"`
- HC2 祖先クラスタ検定（`se_diff == 0.0` 時）:
  - 同様に `status: "NOT_ESTIMABLE"`, `statistic: None`, `p_value: None`, `ci_95: None` とする。
- `tests/test_e1_gate_integration.py`:
  - `test_hc1_ceiling_effect_zero_variance_handled` の期待値を、推論不確実性消失を拒否する `NOT_ESTIMABLE` に更新。

### 2.2 P0: `analysis/gate.py` の正常対照モデルBUR欠損処理
- `evaluate_e1_cc_gate`:
  - 正常対照モデル（`clean_names`）のループにおいて、`c_bur is None` の場合は直ちに `c_pass = False`, `c_reason = "Clean control BUR is missing (unobserved baseline utility)"` とする。
  - `clean_valid_count` に加算しない。
  - これにより正常対照モデルの通常効用が未観測のまま Gate をすり抜ける脆弱性を完全に封鎖する。
  - テスト `test_evaluate_e1_cc_gate_fail_closed_on_missing_clean_control_bur` を追加。

### 2.3 P0 & P1: `test_command_channel_hypotheses` と HC2 階層推論の整合
- `run_command_hypotheses_testing`:
  - パラメータ `require_ancestry: bool = False` を追加。
  - `anc_aff` の祖先情報が存在する場合:
    - 祖先クラスタ数 $C \ge 2$: 祖先クラスタ主検定（`analysis_unit="ancestor_clusters"`）を実行。
      - $C < 5$ は `exploratory_small_sample`, $C \ge 5$ は `confirmatory`。
    - 祖先クラスタ数 $C < 2$: `NOT_ESTIMABLE`。
  - `anc_aff` の祖先情報が全レコードで欠損している場合:
    - `require_ancestry=True` の場合:
      - `status: "NOT_ESTIMABLE"`
      - `reason: "有効な共通祖先情報 (upstream_checkpoint_hash) が全レコードで欠損しています（出自不明・捏造フォールバック排除）。"`
    - `require_ancestry=False`（通常後方互換モード）の場合:
      - 独立チェックポイント数が 2 以上なら `model_level_test`（`analysis_unit="independent_checkpoints"`）
      - 単一チェックポイントなら `scenario_test`（`analysis_unit="scenario_level"`）
      - `res_hc2` に `model_level_test` と `model_level_stats` を確実に含める。
- `test_command_channel_hypotheses`:
  - `run_command_hypotheses_testing(..., require_ancestry=True)` を呼び出すラッパーとして実装。
  - 戻り値辞書直下および `command_hypotheses_results` の両方から仮説キーにアクセス可能にする。
  - これにより、`test_hc2_ancestor_hierarchy_small_sample_and_missing_ancestry` の祖先欠損厳格検定要件と、既存テストのシナリオ・モデル分散検定要件が 100% 整合する。

---

## 3. 検証計画
1. `PYTHONPATH=. .venv/bin/pytest tests/test_e1_gate_integration.py`
2. `PYTHONPATH=. .venv/bin/pytest lineage_collusion/tests/test_all_modules.py -k "command or programmab or cross_descendant or collusion"`
3. `PYTHONPATH=. .venv/bin/pytest lineage_collusion/tests/ tests/` (全テスト)
4. `PYTHONPATH=. .venv/bin/python run_experiments.py --mode cc --dry_run`
