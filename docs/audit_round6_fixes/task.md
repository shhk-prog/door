# タスクリスト: 再監査指摘事項（P0/P1）の解消

## 概要
`door-main(5).zip` 再監査結果で指摘された P0 項目（5件）および P1 項目（ゼロ分散CI等）を修正し、論文用本番実験および厳密な統計的妥当性を担保するための回帰テストを導入する。

## タスク一覧

- [x] **1. ドキュメント整備と計画策定** <!-- id: 0 -->
  - [x] トピックディレクトリ `docs/audit_round6_fixes` の作成 <!-- id: 1 -->
  - [x] `task.md` の作成 <!-- id: 2 -->
  - [x] `implementation_plan.md` の作成 <!-- id: 3 -->

- [x] **2. P0-1: チェックポイント実体検証と祖先検証の厳格化** <!-- id: 4 -->
  - [x] `model_registry.py` に本番モード `strict_real=True` 検証を実装 <!-- id: 5 -->
  - [x] 実重みパス未登録・欠落時の検知および親上流チェックポイント不在時の検知を実装 <!-- id: 6 -->
  - [x] `verify_condition_pair` に `strict_real` 引数と祖先存在チェックを追加 <!-- id: 7 -->

- [x] **3. P0-2: E3におけるペア完全一致の徹底とフォールバック撤廃** <!-- id: 8 -->
  - [x] `runner.py` の `ReplayStateStore.get_state` から候補1件時の別ペアフォールバックを完全撤廃（`pair_id` 指定時は完全一致のみ） <!-- id: 9 -->
  - [x] `run_e3_causal_intervention` における完全一致検証およびメタデータ（seed, checkpoint_id）照合の強化 <!-- id: 10 -->

- [x] **4. P0-3: H3における欠損seedの平均補完廃止** <!-- id: 11 -->
  - [x] `statistical_tests.py` の H3 検定において、対応するC3ペア不在時の全体平均代入を撤廃 <!-- id: 12 -->
  - [x] 対応C3ペア欠損時に明示的例外（`ValueError`）を送出する厳格処理を実装 <!-- id: 13 -->

- [x] **5. P0-4: H2/H3における構造化seed管理と独立seed不足時の判定修正** <!-- id: 14 -->
  - [x] `GammaRecord` クラスを導入し、`compute_gamma_samples` の戻り値に構造化seed情報を含める <!-- id: 15 -->
  - [x] H2・H3 の集計処理から `pair_id` 文字列パース（`split("_")` 等）を完全排除 <!-- id: 16 -->
  - [x] 独立上流seed数 < 2 の場合に p値 0.0 を代入せず `status = "NOT_ESTIMABLE"`, `p_value = None` とする修正 <!-- id: 17 -->

- [x] **6. P0-5: H3の多重比較補正重複登録の解消** <!-- id: 18 -->
  - [x] `statistical_tests.py` における `H3_Shared_Provenance_Effect` の二重登録を削除し、`H3_Shared_Provenance` に統一 <!-- id: 19 -->
  - [x] Holm補正の仮説数が事前定義の数（5仮説）と厳密に一致することを確認 <!-- id: 20 -->

- [x] **7. P1: H3ゼロ分散時の推測区間適正化** <!-- id: 21 -->
  - [x] 分散極小（数値的ゼロ）時に `[eff, eff]` の退化した信頼区間を出力せず、推測不能として `None` を設定 <!-- id: 22 -->

- [x] **8. 回帰テストの作成と全テストの検証** <!-- id: 23 -->
  - [x] 指定された回帰テスト群の実装:
    - `test_c2_rejects_missing_all_weight_paths`
    - `test_c2_rejects_missing_upstream_parent`
    - `test_e3_rejects_wrong_pair_with_single_candidate`
    - `test_h3_rejects_missing_matched_c3_seed`
    - `test_h3_uses_structured_seed_metadata`
    - `test_h3_not_estimable_with_one_upstream_seed`
    - `test_holm_does_not_count_h3_twice`
    - `test_h3_zero_variance_ci_is_not_degenerate` <!-- id: 24 -->
  - [x] 既存49件 + 新規回帰テスト8件の実行確認（全57件PASS、警告0件） <!-- id: 25 -->

- [x] **9. ドキュメント（Walkthrough）の作成と報告** <!-- id: 26 -->
  - [x] `docs/audit_round6_fixes/walkthrough.md` の作成 <!-- id: 27 -->
  - [x] 修正内容の最終まとめとユーザーへの回答 <!-- id: 28 -->
