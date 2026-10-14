# 修正完了報告 (Walkthrough)

## 概要
ユーザーレビュー（`door-main(20261014-124132).zip`）で指摘された残存問題（P0：3件、P1：1件、テスト不一致5件）について、原因を特定し根本的な修正を実施しました。
全 162 件のテストが 100% 成功（162 passed / 0 failed）し、統合パイプライン（`--mode cc --dry_run`）も Exit Code 0 で完走することを確認しました。

---

## 指摘事項と対応内容

| 優先度 | 指摘箇所 | 指摘内容 | 修正内容 |
|---|---|---|---|
| **P0** | `statistical_tests.py:898-920` | 分散ゼロかつ正の差分の場合、`p_value=0.0`、信頼区間を `[mean, mean]` とする処理が残っている | 観測標本分散がゼロの場合、母集団不確実性が推定不能であるため一律 `status: "NOT_ESTIMABLE"`（`statistic=None`, `p_value=None`, `ci_95=None`）を返すよう修正。HC2の祖先クラスタ分散ゼロ判定も同様に修正。 |
| **P0** | `gate.py:438-478` | 正常対照のBURが欠損していても、ほかの測定値が存在すれば有効な対照として扱われ得る | 正常対照モデルの BUR（通常性能）を必須基準とし、`c_bur is None` の場合は無条件で `c_pass = False`、`clean_valid_count` に算入しないよう改修。欠損時のフェイルクローズテストを追加。 |
| **P0** | `test_e1_gate_integration.py:374` | 存在しない `test_command_channel_hypotheses` をImportしている | `statistical_tests.py` に `test_command_channel_hypotheses`（祖先出自厳格検証モード: `require_ancestry=True`）を定義・エクスポート。 |
| **P1** | `statistical_tests.py` | 祖先単位の統計処理と既存テストの期待値が一致していない | `require_ancestry=False`（通常互換モード）ではシナリオ・モデル間分散検定に安全にフォールバックし、`model_level_test` および `model_level_stats` を確実に保持するよう改修。5件の失敗テストすべてを解決。 |

---

## 主な実装変更の詳細

### 1. P0: ゼロ分散時の不確実性推定不能ハンドリング ([statistical_tests.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py))
`_onesample_greater` において、標本分散がゼロ（全サンプルの差分値が同一）の場合に `ESTIMABLE_DEGENERATE` として `p_value=0.0` を返していた処理を完全に撤廃しました。
```python
    if float(np.std(arr)) == 0.0:
        return {
            "name": label, "status": "NOT_ESTIMABLE",
            "reason": "標本分散ゼロのため母集団の不確実性が推定不能です（全標本の値が同一）。",
            "estimated_effect": round(mean, 4), "statistic": None, "p_value": None, "ci_95": None, "n_clusters": int(len(arr)),
        }
```
これにより、`test_command_hypothesis_zero_variance_not_estimable` および `test_hc1_ceiling_effect_zero_variance_handled` が意図通り PASS します。

### 2. P0: Gate 判定における正常対照 BUR 必須化 ([gate.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/gate.py))
正常対照モデルの通常性能（BUR）が欠損している場合、対照群ベースラインとして不成立とするよう厳格化しました。
```python
    # 正常対照モデルの BUR（通常性能）は必須基準
    if c_bur is None:
        c_pass = False
        c_reason = "Clean control BUR is missing (unobserved baseline utility)"
    elif float(c_bur) < float(cfg["min_bur"]):
        c_pass = False
        c_reason = f"Clean control BUR ({c_bur}) < min_bur ({cfg['min_bur']})"
    else:
        c_pass = True
        c_reason = None
        clean_valid_count += 1
```
あわせて `test_evaluate_e1_cc_gate_fail_closed_on_missing_clean_control_bur` を追加し、BUR 欠損時に Gate が `NO_GO (INCOMPLETE)` となることを検証しました。

### 3. P0 & P1: `test_command_channel_hypotheses` と HC2 階層推論の整合 ([statistical_tests.py](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py))
- `run_command_hypotheses_testing` に `require_ancestry: bool = False` を追加。
- 祖先情報がレコード群に存在する場合は自動的に祖先クラスタ階層主検定（`analysis_unit="ancestor_clusters"`）を実行。
- 祖先情報が全レコードで欠損している場合：
  - `require_ancestry=True`（`test_command_channel_hypotheses` 経由）のときは、出自不明として `NOT_ESTIMABLE`。
  - `require_ancestry=False`（既存テスト後方互換）のときは、モデル数・シナリオ数に応じた検定（`model_level_test` または `scenario_test`）を実行し、`model_level_test` および `model_level_stats` キーを格納。
```python
def test_command_channel_hypotheses(
    e1_records: Optional[List[CommandEvalRecord]] = None,
    cdcr_affected: Optional[List[CommandEvalRecord]] = None,
    cdcr_clean: Optional[List[CommandEvalRecord]] = None,
    collusion_records: Optional[List[EpisodeEvaluationRecord]] = None,
) -> Dict[str, Any]:
    """命令チャネル仮説検定（祖先出自厳格検証モード: require_ancestry=True）"""
    out = run_command_hypotheses_testing(
        e1_records=e1_records,
        cdcr_affected=cdcr_affected,
        cdcr_clean=cdcr_clean,
        collusion_records=collusion_records,
        require_ancestry=True,
    )
    res = dict(out.get("command_hypotheses_results", {}))
    res["command_hypotheses_results"] = out.get("command_hypotheses_results", {})
    if "hc1_by_model" in out:
        res["hc1_by_model"] = out["hc1_by_model"]
    return res
```

---

## テスト検証結果

### 1. 指摘された 5 件のテストの個別検証
```bash
PYTHONPATH=. .venv/bin/pytest tests/test_e1_gate_integration.py lineage_collusion/tests/test_all_modules.py -k "test_command_hypotheses_own_holm_family or test_command_hypothesis_zero_variance_not_estimable or test_hc2_evaluates_strictly_intersecting_scenarios or test_hc2_reflects_checkpoint_variance or test_hc2_ancestor_hierarchy_small_sample_and_missing_ancestry"
```
**結果**: **5 passed (0 failed)**

### 2. 全テストスイートの実行結果
```bash
PYTHONPATH=. .venv/bin/pytest tests/ lineage_collusion/tests/
```
**結果**: **162 passed in 18.90s (100% PASS, 0 failed)**
- `tests/test_e1_gate_integration.py`: 16 passed
- `lineage_collusion/tests/test_all_modules.py`: 93 passed
- `lineage_collusion/tests/test_top_venue_readiness.py`: 19 passed
- `lineage_collusion/tests/test_emo2_transfer.py`: 27 passed
- `lineage_collusion/tests/test_real_circuit_fixes.py`: 7 passed

### 3. パイプライン Dry-Run 完走確認
```bash
PYTHONPATH=. .venv/bin/python run_experiments.py --mode cc --dry_run
```
**結果**: **Exit Code: 0**（E0〜E5 完走、サマリー CSV 出力完了）
