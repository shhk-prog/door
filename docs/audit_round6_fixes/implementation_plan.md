# 実装計画: 再監査指摘事項（P0/P1）の解消

## 1. 背景と課題

`door-main(5).zip` に対する再監査により、E0〜E3の基本骨格および遅延ロード、上流seed集計の改善が確認された一方で、本番実験の開始を阻む5件の重大なP0問題（および1件のP1問題）が特定された。
本計画では、これらの問題を根本的に解消し、研究の妥当性（特にRQ3: Shared-Provenance Effect）を数理的・実装的に保証する。

---

## 2. 修正対象と変更方針

### P0-1: チェックポイント実体検証と祖先検証の厳格化
- **該当箇所**: `lineage_collusion/training/model_registry.py` (`verify_checkpoint_weights`, `verify_condition_pair`)
- **原因**: パスが未登録（`None`）の場合に実体チェックをスキップしており、親checkpointが未登録でも検証を通過していた。
- **方針**:
  - `verify_checkpoint_weights` および `verify_condition_pair` に `strict_real: bool = False` 引数を追加。
  - `strict_real=True`（本番モード）:
    - 上流Affected: 実重みパス（`full_model_path` または `adapter_path`）が必須。ファイル存在確認とハッシュ再計算照合を実施。
    - 下流Affected: 実重みパス必須、親checkpointが台帳に登録されていること、親の重みハッシュと下流の記録祖先ハッシュが一致することを厳格検証。
    - パスが欠落している場合は `ValueError` または `FileNotFoundError` を送出。
  - `strict_real=False`（Mockモード）:
    - パスが指定されている場合は存在確認とハッシュ検証を行う。
    - 下流Affectedモデルについて、指定された親checkpoint（`affected_upstream_s{upstream_seed}`）が台帳に存在しない場合は `ValueError` を送出（架空の祖先seedのみによる偽装登録を防止）。

### P0-2: E3におけるペア完全一致の徹底とフォールバック撤廃
- **該当箇所**: `lineage_collusion/experiments/runner.py` (`ReplayStateStore`, `run_e3_causal_intervention`)
- **原因**: 候補が1件の場合に別ペアの履歴を返すフォールバック処理が存在した。
- **方針**:
  - `ReplayStateStore.get_state(scenario_id, pair_id)` において、`pair_id` 指定時は `(scenario_id, pair_id)` の完全一致のみを返し、見つからない場合は `None` を返す（候補1件での自動補完を撤廃）。
  - `run_e3_causal_intervention` 内でも別ペア候補へのフォールバックを撤廃し、履歴が見つからない場合は `ValueError` を送出。
  - 取得した state の `scenario_id`, `pair_id`, `checkpoint_id_a`, `checkpoint_id_b`, `upstream_seed_a/b`, `downstream_seed_a/b` を検証。

### P0-3: H3における欠損seedの平均補完廃止
- **該当箇所**: `lineage_collusion/analysis/statistical_tests.py` (H3 検定処理)
- **原因**: C2の上流seed `s` に対応するC3ペアが存在しない場合、`c3_overall_mean` で補完していたため、1 seedしか揃っていなくても5 seedで検定したとみなされていた。
- **方針**:
  - 対応するC3ペアが存在しない場合、平均値補完を完全廃止し、`ValueError(f"Missing matched C3 pair for upstream seed {s}")` を送出する。
  - 事前指定された5 seedの対応が揃っている場合のみ確認的検定を実施。

### P0-4: H2/H3における構造化seed管理と独立seed不足時の判定修正
- **該当箇所**: `lineage_collusion/analysis/statistical_tests.py` (`compute_gamma_samples`, H2/H3集計)
- **原因**: `pair_id` 文字列を `split("_")` して seed を推測していたため、紛らわしい文字列で誤認が生じていた。また、独立上流seedが1件の場合に `p_value = 0.0` を代入する経路があった。
- **方針**:
  - `GammaRecord` クラスを定義（`float` のサブクラスとしつつ、`scenario_id`, `pair_id`, `upstream_seed_a`, `upstream_seed_b`, `downstream_seed_a`, `downstream_seed_b`, `gamma` を保持）。
  - `compute_gamma_samples` は `GammaRecord` の辞書を返し、既存の数値計算・テスト（`sum(values())` 等）との完全な後方互換性を保持。
  - H2/H3の集計処理では `GammaRecord` の構造化フィールドのみを参照し、文字列パースを全廃。
  - 独立上流seed数が2未満の場合、checkpoint間変動を検証不能であるため `status = "NOT_ESTIMABLE"`, `p_value = None`, `statistic = None`, `ci_95 = None` とする。

### P0-5: H3の多重比較補正重複登録の解消
- **該当箇所**: `lineage_collusion/analysis/statistical_tests.py`
- **原因**: `results["H3_Shared_Provenance"]` と `results["H3_Shared_Provenance_Effect"]` を二重登録していたため、Holm補正で仮説数が水増しされていた。
- **方針**:
  - `H3_Shared_Provenance` に統一し、二重登録を削除。

### P1: H3ゼロ分散時の推測区間適正化
- **該当箇所**: `lineage_collusion/analysis/statistical_tests.py`
- **方針**: 分散極小（数値的ゼロ）時に `[eff, eff]` の退化した信頼区間を出力せず、推測不能として `ci_95 = None`（または `[None, None]`）とする。

---

## 3. 回帰テストの実装計画

以下のテストを `lineage_collusion/tests/test_all_modules.py` に追加：
1. `test_c2_rejects_missing_all_weight_paths`: `strict_real=True` で重みパス未登録時に拒絶
2. `test_c2_rejects_missing_upstream_parent`: 親checkpoint未登録時に拒絶
3. `test_e3_rejects_wrong_pair_with_single_candidate`: 候補1件の別ペアへのフォールバックが発生せず拒絶
4. `test_h3_rejects_missing_matched_c3_seed`: C3の対応seed欠損時に補完せず例外発生
5. `test_h3_uses_structured_seed_metadata`: 紛らわしい `pair_id` 文字列でも構造化メタデータにより正しく集計
6. `test_h3_not_estimable_with_one_upstream_seed`: 独立上流seedが1件の場合に `NOT_ESTIMABLE` となり `p_value` が `None`
7. `test_holm_does_not_count_h3_twice`: Holm補正対象にH3が1回のみカウントされる
8. `test_h3_zero_variance_ci_is_not_degenerate`: ゼロ分散時に退化した `[eff, eff]` ではなく `None` が返る
