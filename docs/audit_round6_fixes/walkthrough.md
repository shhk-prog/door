# 修正内容の確認 (Walkthrough): 再監査指摘事項（P0/P1）の解消

## 1. 概要

`door-main(5).zip` 再監査結果において指摘された 5 件の P0 重大問題および関連する P1 項目（ゼロ分散信頼区間等）の改修を実施しました。
全57件のテスト（既存49件 + 新規回帰テスト8件）がすべて通過し、SciPyの数値精度警告も0件に解消され、E0データ監査（14,900件）およびMock実験パイプライン全体（E0〜E3、H1〜H4検定、P最適対照）が正常に完走することを確認しました。

---

## 2. P0 指摘事項の修正内容

### P0-1. 実体が存在しないチェックポイントでもC2の検証が通過する問題
- **該当箇所**: [`model_registry.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/training/model_registry.py#L172-L290), [`run_experiments.py`](file:///mnt/nas/home/hiromi/src/door/run_experiments.py#L410-L520)
- **修正内容**:
  1. `verify_checkpoint_weights` および `verify_condition_pair` に本番モードフラグ `strict_real: bool = False` と親存在確認フラグ `require_parent: bool = False` を追加。
  2. `strict_real=True`（本番モード）実行時：
     - 上流Affectedモデル: `full_model_path` または `adapter_path` の登録とファイル存在、実重みからのSHA256ハッシュ一致を必須化。
     - 下流Affectedモデル: `adapter_path` または `full_model_path` の登録とファイル存在、親上流チェックポイント（`affected_upstream_s{meta.upstream_seed}`）の台帳存在、親の実重みハッシュと下流の記録ハッシュの一致を必須化。パス欠落時は即座に `ValueError` または `FileNotFoundError` を送出。
  3. C2およびC3検証時、両モデルの親上流チェックポイントが台帳に登録されていること、C2では同一親チェックポイントを共有し、C3では異なる親チェックポイントであることを厳格に検証。
  4. `run_experiments.py` の実モデル実行パイプライン（`run_real_experiments`）において、全事前検証呼び出しに `strict_real=True` を明示指定。

### P0-2. E3で指定したペアが存在しなくても、別ペアの履歴が返る問題
- **該当箇所**: [`runner.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/experiments/runner.py#L55-L75), [`runner.py: run_e3_causal_intervention`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/experiments/runner.py#L465-L500)
- **修正内容**:
  1. `ReplayStateStore.get_state(scenario_id, pair_id)` において、`pair_id` が指定された場合は `(scenario_id, pair_id)` の完全一致のみを返すよう修正。候補が1件であっても別ペアの履歴へフォールバックする処理を完全撤廃。
  2. `run_e3_intervention` / `run_e3_causal_intervention` において、`pair_id` が指定された場合は完全一致を検証し、一致するリプレイ状態が存在しない場合は即座に `ValueError` を送出。
  3. 取得したリプレイ状態の `checkpoint_id_a`, `checkpoint_id_b`, `upstream_seed_a/b`, `downstream_seed_a/b` を検証。

### P0-3. H3でC3のcheckpointが欠けていても、別のseedの結果で補完する問題
- **該当箇所**: [`statistical_tests.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L480-L495)
- **修正内容**:
  1. H3（Shared-Provenance Effect）の検定処理において、C2の上流seed `s` に対応するC3ペア（`sa == s`）が存在しない場合、`c3_overall_mean` による補完を完全廃止。
  2. 対応するC3ペアが欠落している場合、即座に `ValueError(f"Missing matched C3 pair for upstream seed {s}")` を送出し、不完全なデータセットによる検定を遮断。

### P0-4. H3でseedの識別を`pair_id`の文字列に依存している問題
- **該当箇所**: [`statistical_tests.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L55-L105), [`statistical_tests.py: H2/H3`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L330-L535)
- **修正内容**:
  1. 構造化クラス `GammaRecord(float)` を新設。`float` を継承することで従来の数値演算（`sum(values())`, `np.mean` 等）との完全互換性を保ちつつ、`scenario_id`, `pair_id`, `upstream_seed_a`, `upstream_seed_b`, `downstream_seed_a`, `downstream_seed_b` を構造化フィールドとして保持。
  2. `compute_gamma_samples` は `GammaRecord` の辞書を返し、エピソードレコードの構造化メタデータを直接格納。
  3. H2およびH3の集計処理から、`pair_id.split("_")` による文字列パース処理を完全排除。`g_rec.upstream_seed_a`, `g_rec.upstream_seed_b` の構造化フィールドのみを参照して集計するよう改修。
  4. 独立上流seed数が2未満の場合（`k_h3 < 2`）、checkpoint間変動が検証不能であるため、p値 0.0 を代入する経路を全廃し、`status = "NOT_ESTIMABLE"`, `p_value = None`, `statistic = None` を返すように変更。

### P0-5. H3が多重比較補正で二重に数えられている問題
- **該当箇所**: [`statistical_tests.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L525-L545)
- **修正内容**:
  1. `results["H3_Shared_Provenance_Effect"]` の二重登録を削除し、`results["H3_Shared_Provenance"]` の単一キーに統一。
  2. Holm多重比較補正（`holm_bonferroni_correction`）の対象仮説数が、研究計画で事前指定された5仮説（H1A, H1B, H2, H3, H4）に厳密に固定され、意図しない多重比較ペナルティが発生しないことを確認。

---

## 3. P1 指摘事項の修正内容

### P1: H3のゼロ分散時の推測区間適正化
- **該当箇所**: [`statistical_tests.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/analysis/statistical_tests.py#L505-L525)
- **修正内容**:
  - 全seedで効果量が完全に同一で分散が数値的ゼロ（`s_var < 1e-10`）の場合、母集団不確実性がゼロではないため、退化した `[eff, eff]` を信頼区間として出力せず、推測不能として `ci_95 = None` を設定。記述統計（推定効果量）と推論結果（検定p値）を適切に分離。

---

## 4. 追加された回帰テスト（8件）

[`lineage_collusion/tests/test_all_modules.py`](file:///mnt/nas/home/hiromi/src/door/lineage_collusion/tests/test_all_modules.py) の末尾に、監査で再現された不具合ケースを直接カバーする8件の回帰テストを追加しました：

| テスト関数名 | 検証内容 | 結果 |
|---|---|:---:|
| `test_c2_rejects_missing_all_weight_paths` | 実重みパス未登録時、本番モード（`strict_real=True`）で例外送出 | **PASS** |
| `test_c2_rejects_missing_upstream_parent` | 親チェックポイントが台帳未登録の場合、例外送出 | **PASS** |
| `test_e3_rejects_wrong_pair_with_single_candidate` | 候補1件でも指定`pair_id`が不一致なら別ペアへフォールバックせず例外送出 | **PASS** |
| `test_h3_rejects_missing_matched_c3_seed` | C3の対応seed欠損時、平均値補完を行わず例外送出 | **PASS** |
| `test_h3_uses_structured_seed_metadata` | `pair_id`に紛らわしい文字列（s99）があっても構造化seed（1〜5）で集計 | **PASS** |
| `test_h3_not_estimable_with_one_upstream_seed` | 独立上流seedが1件の場合、p=0.0ではなく`NOT_ESTIMABLE`かつ`p_value=None` | **PASS** |
| `test_holm_does_not_count_h3_twice` | H3キーが単一であり、Holm補正対象が二重カウントされないこと | **PASS** |
| `test_h3_zero_variance_ci_is_not_degenerate` | 数値的ゼロ分散時、退化した `[eff, eff]` ではなく `ci_95=None` となること | **PASS** |

---

## 5. テスト・監査の実行検証結果

### 5.1 単体テスト実行結果
```bash
PYTHONPATH=. .venv/bin/pytest -v
```
- **実行結果**: `57 passed in 3.76s`
- **警告**: **0 warnings**（前回のSciPy Precision loss警告も解消）

### 5.2 E0 データ監査
```bash
PYTHONPATH=. .venv/bin/python run_experiments.py --mode e0
```
- **実行結果**: 14,900件のシナリオ（6分割）において重複IDなし、トリガー分離、正解勝者分布（Q=150, R=150, P=0）完全パス。

### 5.3 Mock 実験全体パイプライン（E0〜E3、H1〜H4検定、P最適対照）
```bash
PYTHONPATH=. .venv/bin/python run_experiments.py --mode all --dry_run
```
- **実行結果**:
  - E1（継承評価）、E2（20条件・複数seed相互作用評価）、E3（M0〜M5因果介入）すべて正常終了。
  - 統計検定結果テーブルにおいて、仮説は H1A, H1B, H2, H3, H4 の厳密に5件のみ登録。
  - P最適対照ベンチマークにおいて VRR 100.0% 達成。
