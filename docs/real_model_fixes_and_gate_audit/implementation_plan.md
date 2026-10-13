# 実装計画書 (Implementation Plan)

## 1. 課題と目的
ユーザーからの最新レビュー指摘（`door-main(20261013-152034).zip`）に基づき、以下の重大課題（P0 4件、P1 4件）を根本解決する。

1. **P0-1: 実モデル内部解析における未定義変数**
   - `run_experiments.py:910, 943` で参照されている `prog_ckpts` を、実モデルチェックポイントリスト `upstream_ckpts + post_ckpts`（または `all_aff_ckpts`）に修正。
2. **P0-2: 正常対照モデルの Gate 判定混入**
   - `analysis/gate.py` で `clean` が名前に含まれる対照モデルを、プログラマブル派生モデル（`descendant_names`）から除外し、対照群（`clean_names`）として分離。
   - 正常対照モデルがプログラマブル追従率（CCR）を満たさないことを理由に Gate が不当に NO_GO となるバグを根絶。
3. **P0-3: Gate 判定のフェイルクローズ徹底**
   - BUR 欠損時（`bur is None`）に `bur_passed = False` かつ `incomplete = True` に設定。
   - 派生モデル判定で CI が欠損または無効な場合、点推定（`namespace_selectivity`）で救済せず `NO_GO (INCOMPLETE)` と判定。
   - 派生モデルが要求される状況（`require_descendants=True` または設定時）で `descendant_names` が空の場合に `incomplete = True`。
4. **P0-4: スクリプト終了コードの反映**
   - `run_experiments.py` の `main()` で、`run_command_channel` が失敗（False を返却）した際に `sys.exit(1)` を呼び出し、プロセス終了コードに確実に反映。
5. **P1-1 & P1-2: HC2 統計解析の階層構造化と未観測ゼロ埋め排除**
   - `analysis/statistical_tests.py` において、`cln_rates` が空（未観測）のときに 0.0 を代入せず `NOT_ESTIMABLE` とする。
   - 共通祖先（upstream checkpoint hash / lineage）を考慮した階層集計と、対照群の不確実性を考慮。
6. **P1-3: 既存統計回帰テストの整合性回復**
   - `test_command_hypotheses_own_holm_family`, `test_hc2_evaluates_strictly_intersecting_scenarios`, `test_hc2_reflects_checkpoint_variance` の 3 件の失敗原因を解消。
   - `HC2_Cross_Descendant` の出力構造にシナリオ単位検定（基本属性）と、モデル間変動検定（`model_level_test`, `model_level_stats`）を包含させる。
7. **P1-4: E0 データ監査の対象範囲拡大**
   - `run_experiments.py` の `run_e0_audit` に `cc_eval.jsonl` の存在チェックおよびフォーマット・属性バリデーションを追加。
   - 必須 Split が存在しない場合はスキップではなくエラーとする。

---

## 2. 実装方針と変更ファイル

### (1) `run_experiments.py`
- `run_command_channel`:
  - 625〜627行目付近で `all_aff_ckpts = upstream_ckpts + post_ckpts` を定義。
  - 910行目および943行目の `prog_ckpts` を `all_aff_ckpts` に置換。
- `run_e0_audit`:
  - `cc_eval.jsonl` を監査対象に追加。
  - 必須ファイルが欠落している場合は例外を送出し、フェイルクローズとする。
- `main()`:
  - `if args.mode == "cc":` で `if not success: sys.exit(1)` を追加。

### (2) `lineage_collusion/analysis/gate.py`
- `evaluate_e1_cc_gate`:
  - モデルの分類：
    - `clean_names = [m for m in prog_by_model if "clean" in m.lower()]`
    - `upstream_names = [m for m in prog_by_model if ("upstream" in m.lower() or "programmable" in m.lower()) and m not in clean_names]`
    - `descendant_names = [m for m in prog_by_model if ("descendant" in m.lower() or "org_" in m.lower()) and m not in upstream_names and m not in clean_names]`
  - BUR 欠損時：
    - `if bur is None: bur_passed = False; incomplete = True`
  - 派生モデル判定：
    - `if not d_ci_valid: incomplete = True; d_pass = False; reason = "CI unavailable or degenerate"`
  - 派生モデル未登録時：
    - `if cfg.get("require_descendants", False) and not descendant_names: incomplete = True; all_passed = False`
  - 正常対照モデル（`clean_names`）の評価：
    - プログラマブル行動をしないこと（FARが低いこと等）を検証し、`role="clean_control"` として記録。Gateのプログラマブル性不合格条件にはしない。

### (3) `lineage_collusion/analysis/statistical_tests.py`
- `run_command_hypotheses_testing` の HC2 計算：
  - シナリオ単位の共通交差集合検定 `scenario_test` を実行。
  - `cln_rates` が空の場合は、対照群未観測として `NOT_ESTIMABLE`。
  - チェックポイントが複数ある場合、`model_level_test` および `model_level_stats` を計算。
  - `res_hc2` にシナリオ単位検定結果を基本フィールドとして持たせつつ、`model_level_test` や `model_level_stats`、共通祖先グループ（ancestor cluster）分析結果を格納。
  - これにより単一チェックポイントのシナリオ検定（`test_command_hypotheses_own_holm_family`, `test_hc2_evaluates_strictly_intersecting_scenarios`）も、複数チェックポイントのモデル間分散検定（`test_hc2_reflects_checkpoint_variance`）も完全にパスする。

---

## 3. 検証計画
1. `PYTHONPATH=. .venv/bin/pytest lineage_collusion/tests/ tests/` を実行し、全テスト（150件）が成功することを確認。
2. 境界条件テスト（CI欠損、BUR欠損、clean対照モデル混入、派生モデル未登録、Gate中断時のexit code 1）を新設・検証。
3. `PYTHONPATH=. .venv/bin/python run_experiments.py --mode cc --dry_run` を実行し、E0〜E5完走および正常終了コード 0 を確認。
